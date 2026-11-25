"""DataUpdateCoordinator for ha_predictions."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from homeassistant.core import callback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.recorder import get_instance
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from sqlalchemy import text

from .const import (
    CONF_ADDITIONAL_SETTINGS,
    CONF_ADDITIONAL_SETTINGS_IMPORT_FROM_RECORDER,
    CONF_FEATURE_ENTITY,
    CONF_PREDICTION_HORIZON,
    CONF_TARGET_ENTITY,
    ENTITY_KEY_OPERATION_MODE,
    ENTITY_KEY_SAMPLING_STRATEGY,
    LOGGER,
    MIN_DATASET_SIZE,
    MSG_DATASET_CHANGED,
    MSG_PREDICTION_MADE,
    MSG_TRAINING_DONE,
    MSG_TRAINING_SETTINGS_CHANGED,
    OFF_STATES,
    SAMPLING_NONE,
    SAMPLING_RANDOM,
    SAMPLING_SMOTE,
    OperationMode,
)
from .ml.const import F_SCORE, PRECISION, RECALL
from .ml.exceptions import ModelNotTrainedError
from .ml.model import Model, SamplingStrategy

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import NoneType

    from homeassistant.core import Event, EventStateChangedData
    from sqlalchemy.orm import Session

    from .data import HAPredictionConfigEntry
    from .entity import HAPredictionEntity


class HAPredictionUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage training/testing the model."""

    config_entry: HAPredictionConfigEntry

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the coordinator."""
        super().__init__(*args, **kwargs)
        # Initialize instance variables to avoid sharing between coordinator instances
        self.scores: tuple[float, dict[Any, dict[str, float]]] | NoneType = None
        self.entity_registry: list[HAPredictionEntity] = []
        self.dataset: pd.DataFrame | NoneType = None
        self.dataset_size: int = 0
        self.model: Model = Model(self.logger)
        self.operation_mode: OperationMode = OperationMode.TRAINING
        self.training_ready: bool = False
        self.last_trained: datetime | NoneType = None
        self.trained_dataset_size: int = 0
        self.current_prediction: tuple[str, float] | NoneType = None
        self._pending: list[dict[str, Any]] = []

    async def initialize(self) -> NoneType:
        """Initialize the coordinator."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.read_table)
        if self.dataset is None:
            self.logger.debug("No dataset found on disk.")
            if self.prediction_horizon == 0 and self.config_entry.options.get(
                CONF_ADDITIONAL_SETTINGS, {}
            ).get(CONF_ADDITIONAL_SETTINGS_IMPORT_FROM_RECORDER, True):
                self.logger.debug("Importing initial dataset from recorder database.")
                self.hass.async_create_task(
                    self._extract_initial_dataset_from_recorder()
                )
                await loop.run_in_executor(None, self.store_table, self.dataset)
            else:
                self._initialize_dataframe()

    async def _async_update_data(self) -> Any:
        """Retrain periodically once enough new data points have been collected."""
        new_rows = self.dataset_size - self.trained_dataset_size
        threshold = max(MIN_DATASET_SIZE // 2, self.trained_dataset_size // 10)
        if self.training_ready and (self.last_trained is None or new_rows >= threshold):
            self.logger.info("Retraining automatically (%d new data points)", new_rows)
            await self.train()

    def register(self, entity: HAPredictionEntity) -> None:
        """Register an entity to be notified on changes."""
        self.entity_registry.append(entity)

    def remove_listeners(self) -> None:
        """Remove all listeners."""
        self.entity_registry.clear()
        for pending in self._pending:
            pending["cancel"]()
        self._pending.clear()

    @property
    def prediction_horizon(self) -> int:
        """Seconds between a feature snapshot and its target label."""
        value = self.config_entry.options.get(CONF_PREDICTION_HORIZON)
        if value is None:
            value = self.config_entry.data.get(CONF_ADDITIONAL_SETTINGS, {}).get(
                CONF_PREDICTION_HORIZON, 0
            )
        return int(value)

    def _start_snapshot(self) -> None:
        """Snapshot the features now; the label is filled in after the horizon."""
        target = self.config_entry.data[CONF_TARGET_ENTITY]
        features = self._get_states_for_entities(include_target=False)
        pending: dict[str, Any] = {
            "features": features,
            "seen": [self._get_state_for_entity(target)],
        }

        @callback
        def _expired(_now: Any) -> None:
            self._pending.remove(pending)
            self.hass.async_create_task(self._async_finish_snapshot(pending))

        pending["cancel"] = async_call_later(
            self.hass, self.prediction_horizon, _expired
        )
        self._pending.append(pending)

    async def _async_finish_snapshot(self, pending: dict[str, Any]) -> None:
        """Label a snapshot with what the target did during the horizon."""
        seen = [s for s in pending["seen"] if s is not None]
        active = [s for s in seen if s not in OFF_STATES]
        # Any activity inside the window counts, so short motion pulses are not missed
        label = active[-1] if active else (seen[-1] if seen else None)
        if label in ("unavailable", "unknown"):
            return
        await self.hass.async_add_executor_job(
            self._append_row, [*pending["features"], label]
        )
        for entity in self.entity_registry:
            entity.notify(MSG_DATASET_CHANGED)

    def set_zscores(self, *, value: bool) -> NoneType:
        """En- or disable the use of zscores for training."""
        if value and "zscores" not in self.model.transformations:
            self.model.transformations["zscores"] = {}
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("Z-Score normalization enabled.")
        elif not value and "zscores" in self.model.transformations:
            self.model.transformations.pop("zscores", None)
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("Z-Score normalization disabled.")

    def get_option(self, key: str) -> str | NoneType:
        """Get the current option for a given key."""
        if key == ENTITY_KEY_OPERATION_MODE:
            return self.operation_mode.name
        if key == ENTITY_KEY_SAMPLING_STRATEGY:
            if "sampling" not in self.model.transformations:
                return SAMPLING_NONE
            sampling_type = self.model.transformations["sampling"]["type"]
            if sampling_type == SamplingStrategy.RANDOM_OVER:
                return SAMPLING_RANDOM
            if sampling_type == SamplingStrategy.SMOTE:
                return SAMPLING_SMOTE
            self.logger.warning(
                "Unknown sampling strategy '%s', defaulting to '%s'",
                sampling_type,
                SAMPLING_NONE,
            )
            return SAMPLING_NONE
        return None

    def select_option(self, key: str, value: str) -> NoneType:
        """Change the selected option."""
        if key == ENTITY_KEY_OPERATION_MODE:
            self._set_operation_mode(OperationMode[value])
        elif key == ENTITY_KEY_SAMPLING_STRATEGY:
            self._set_sampling_strategy(value)

    def _set_operation_mode(self, mode: OperationMode) -> None:
        """Set the operation mode."""
        if mode != self.operation_mode:
            self.operation_mode = mode
            self.logger.info("Operation mode has been changed to %s", mode)

    def _set_sampling_strategy(self, strategy: str) -> NoneType:
        """Set the sampling strategy for handling imbalanced datasets."""
        if strategy == SAMPLING_RANDOM:
            self.model.transformations["sampling"] = {
                "type": SamplingStrategy.RANDOM_OVER
            }
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("Random oversampling enabled.")
        elif strategy == SAMPLING_SMOTE:
            self.model.transformations["sampling"] = {
                "type": SamplingStrategy.SMOTE,
                "k_neighbors": 5,
            }
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("SMOTE enabled.")
        else:
            self.model.transformations.pop("sampling", None)
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("Sampling disabled.")

    def state_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle state changes of monitored entities."""
        new_state = event.data["new_state"]
        old_state = event.data["old_state"]

        if (
            new_state
            and event.data["entity_id"] == self.config_entry.data[CONF_TARGET_ENTITY]
        ):
            for pending in self._pending:
                pending["seen"].append(new_state.state)

        # Only act if state actually changed
        if old_state and new_state and old_state.state != new_state.state:
            self.logger.info("Detecting changed states, storing current instance.")
            # Schedule data collection and prediction in executor to avoid
            # blocking event loop
            self.hass.async_create_task(self._async_collect_and_predict())

    async def _async_collect_and_predict(self) -> None:
        """
        Collect data and make prediction.

        Runs blocking operations in executor to avoid blocking event loop.
        """
        await self._async_collect_once()

        # Make prediction if model is ready
        if self.model.prediction_ready and self.dataset is not None:
            self.logger.info("Making new prediction after state change.")
            await self._async_make_prediction()

    async def async_collect(self) -> None:
        """
        Collect the current state as a new data point.

        This is an async wrapper that runs blocking operations in executor.
        """
        await self._async_collect_once()

    async def _async_collect_once(self) -> None:
        """Collect one data point now, or schedule it when a horizon is set."""
        if self.prediction_horizon > 0:
            self._start_snapshot()
            return
        await self.hass.async_add_executor_job(self._collect_data)
        for entity in self.entity_registry:
            entity.notify(MSG_DATASET_CHANGED)

    def _collect_data(self) -> None:
        """
        Collect the current state (blocking operations).

        Runs in executor to avoid blocking event loop.
        """
        xy = self._get_states_for_entities()
        if xy is None:
            self.logger.info(
                "Target entity not available or not known, skipping data collection."
            )
            return
        self._append_row(xy)

    def _append_row(self, xy: list[Any]) -> None:
        """Append one row to the dataset and persist it (blocking)."""
        if self.dataset is None:
            self._initialize_dataframe()
        if self.dataset is not None and xy[-1] not in [
            "unavailable",
            "unknown",
            "Unavailable",
            "Unknown",
            None,
        ]:
            try:
                self.dataset.loc[len(self.dataset)] = xy
                self.dataset_size = self.dataset.shape[0]
            except ValueError:
                self.logger.exception(
                    "Error adding data (%s) to dataset: %s", xy, self.dataset
                )
            self.store_table(self.dataset)
        self.training_ready = self.dataset_size >= MIN_DATASET_SIZE

    async def _async_make_prediction(self) -> None:
        """
        Make a prediction.

        Runs blocking operations in executor to avoid blocking event loop.
        """
        # Run blocking operations in executor
        prediction = await self.hass.async_add_executor_job(self._compute_prediction)

        # Update state and notify entities on main event loop
        if prediction is not None:
            self.current_prediction = prediction
            self.logger.info("New prediction: %s", str(self.current_prediction))
            for entity in self.entity_registry:
                entity.notify(MSG_PREDICTION_MADE)

    def _compute_prediction(self) -> tuple[str, float] | None:
        """
        Compute prediction (blocking operations).

        Runs in executor to avoid blocking event loop.
        """
        try:
            return self.model.predict(
                np.array(self._get_states_for_entities(include_target=False)).reshape(
                    1, -1
                )
            )
        except ModelNotTrainedError as e:
            self.logger.warning(e)
            return None

    def _initialize_dataframe(self) -> NoneType:
        """Initialize empty dataframe for dataset."""
        self.dataset = pd.DataFrame(
            columns=[
                *list(self.config_entry.data[CONF_FEATURE_ENTITY]),
                self.config_entry.data[CONF_TARGET_ENTITY],
            ]
        )
        self.logger.debug("Initialized new dataframe: %s", str(self.dataset))

    def _get_state_for_entity(self, entity_id: str) -> str | float | NoneType:
        """Get the state for a given entity_id."""
        if state := self.hass.states.get(entity_id=entity_id):
            try:
                return float(state.state)
            except ValueError:
                return state.state

        return None

    def _get_states_for_entities(
        self, *, include_target: bool | NoneType = True
    ) -> list[str | float | NoneType] | NoneType:
        """Get the states for all monitored entities."""
        features = [
            self._get_state_for_entity(e)
            for e in self.config_entry.data[CONF_FEATURE_ENTITY]
        ]
        if include_target:
            target_entity_state = self._get_state_for_entity(
                entity_id=self.config_entry.data[CONF_TARGET_ENTITY]
            )
            if target_entity_state not in [
                "unavailable",
                "unknown",
                None,
                "Unavailable",
                "Unknown",
            ]:
                return [*features, target_entity_state]
            return None
        return features

    def read_table(self) -> NoneType:
        """Read dataset from file."""
        self.logger.info(
            "Reading dataset from file: %s",
            str(self.config_entry.runtime_data.datafile),
        )
        if Path.exists(self.config_entry.runtime_data.datafile):
            try:
                self.dataset = pd.read_csv(
                    self.config_entry.runtime_data.datafile, header=0
                )
                self.dataset_size = self.dataset.shape[0]
                self.training_ready = self.dataset_size >= MIN_DATASET_SIZE
                self.logger.debug("Read dataset with %d instances.", self.dataset_size)
                [e.notify(MSG_DATASET_CHANGED) for e in self.entity_registry]
            except (OSError, PermissionError):
                self.logger.exception(
                    "Failed to read dataset from file %s",
                    str(self.config_entry.runtime_data.datafile),
                )
            except pd.errors.ParserError:
                self.logger.exception(
                    "Failed to parse CSV file %s",
                    str(self.config_entry.runtime_data.datafile),
                )

    def store_table(self, df: pd.DataFrame | NoneType) -> None:
        """Store dataset to file."""
        try:
            self.config_entry.runtime_data.datafile.parent.mkdir(
                parents=True, exist_ok=True
            )
            if df is not None:
                df.to_csv(self.config_entry.runtime_data.datafile, index=False)
        except (OSError, PermissionError):
            self.logger.exception(
                "Failed to store dataset to file %s",
                str(self.config_entry.runtime_data.datafile),
            )

    async def train(self) -> NoneType:
        """Run the training process."""
        self.logger.info("training")

        # Store and read table on/from disk
        await self.hass.async_add_executor_job(self.store_table, self.dataset)
        await self.hass.async_add_executor_job(self.read_table)

        # Run actual training
        if self.dataset is None or not self.training_ready:
            self.logger.warning(
                "Not enough data points collected yet, need at least %i, have %d",
                MIN_DATASET_SIZE,
                self.dataset_size,
            )
            return

        # Convert DataFrame to numpy array
        data_numpy = self.dataset.copy().to_numpy()

        if self.operation_mode == OperationMode.TRAINING:
            await self.hass.async_add_executor_job(self.model.train_eval, data_numpy)
            if self.model.scores is not None:
                self.scores = self.model.scores
                self.logger.info(
                    "Training complete, accuracy: %.02f",
                    self.scores[0],
                )
                for cls in self.model.scores[1]:
                    self.logger.debug(
                        "Class %s: p=%f, r=%f, f=%f",
                        cls,
                        self.model.scores[1][cls][PRECISION],
                        self.model.scores[1][cls][RECALL],
                        self.model.scores[1][cls][F_SCORE],
                    )
        # The prediction model is trained on all data in either mode
        await self.hass.async_add_executor_job(self.model.train_final, data_numpy)
        self.last_trained = datetime.now(tz=UTC)
        self.trained_dataset_size = self.dataset_size
        [e.notify(MSG_TRAINING_DONE) for e in self.entity_registry]
        await self._async_make_prediction()

    async def _extract_initial_dataset_from_recorder(self) -> NoneType:
        """Extract initial dataset from recorder history."""
        dburl = get_instance(self.hass).db_url
        self.logger.info("Extracting initial dataset from recorder database: %s", dburl)

        entities = [
            *self.config_entry.data[CONF_FEATURE_ENTITY],
            self.config_entry.data[CONF_TARGET_ENTITY],
        ]

        # Run in executor since database operations are blocking
        def _extract() -> tuple[int, pd.DataFrame]:
            # Get recorder instance
            recorder = get_instance(self.hass)
            extractor = HomeAssistantStateExtractor(recorder.get_session())

            raw_data = extractor.extract_states(entities)
            pivot_data = extractor.pivot_to_wide_format(
                raw_data, entities, self.config_entry.data[CONF_TARGET_ENTITY]
            )
            return (len(pivot_data), pd.DataFrame(pivot_data))

        # Extract data in executor
        number_of_rows, array = await get_instance(self.hass).async_add_executor_job(
            _extract
        )

        self.logger.info("Extracted %d rows from recorder database.", number_of_rows)
        self.dataset = array
        self.dataset_size = self.dataset.shape[0]
        self.training_ready = self.dataset_size >= MIN_DATASET_SIZE
        for entity in self.entity_registry:
            entity.notify(MSG_DATASET_CHANGED)


class HomeAssistantStateExtractor:
    """Extract and transform Home Assistant state data."""

    def __init__(self, session: Session) -> NoneType:
        """
        Initialize extractor with existing scoped_session from Home Assistant.

        Args:
            session: scoped_session instance from Home Assistant

        """
        self.session = session

    @contextmanager
    def session_scope(self) -> Generator[Session, Any, Any]:
        """
        Provide a transactional scope for database operations.

        Usage:
            with extractor.session_scope() as session:
                result = session.execute(text(query), params)
        """
        session = self.session
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def extract_states(
        self,
        entities: list[str],
        limit: int = 5000,
    ) -> list[tuple]:
        """
        Extract state data from database.

        Args:
            entities: List of entity IDs to extract
            limit: Optional limit on number of rows returned

        Returns:
            List of tuples: (timestamp, entity_id, state)

        """
        # Build query conditions
        params = {"entities": tuple(entities)}

        limit_clause = f"LIMIT {limit}" if limit else ""

        query = text(f"""
        SELECT
            states.last_updated_ts,
            states_meta.entity_id,
            states.state
        FROM states
        INNER JOIN states_meta ON states.metadata_id = states_meta.metadata_id
        WHERE states_meta.entity_id IN ({",".join(["'" + e + "'" for e in entities])})
        ORDER BY states.last_updated_ts, states_meta.entity_id
        {limit_clause}
        """)  # noqa: S608

        LOGGER.debug("Executing query: %s", query)

        with self.session_scope() as session:
            result = session.execute(query, params)
            return [tuple(row) for row in result]

    def pivot_to_wide_format(
        self, rows: list[tuple], entities: list[str], target_entity: str
    ) -> list[dict]:
        """
        Convert long format data to wide/pivot format with forward fill.

        Args:
            rows: List of (timestamp, entity_id, state) tuples
            entities: List of entity IDs
            target_entity: The target entity ID (must be present in all rows)

        Returns:
            List of dicts with timestamp + all entity states

        """
        if not rows:
            return []

        # Group states by timestamp
        timestamp_data = defaultdict(dict)
        for timestamp, entity_id, state in rows:
            timestamp_data[timestamp][entity_id] = state

        # Get sorted timestamps
        all_timestamps = sorted(timestamp_data.keys())

        # Track last known state for forward fill
        last_state = dict.fromkeys(entities, "")

        # Build pivot rows
        pivot_rows = []
        for timestamp in all_timestamps:
            # Update last known states with new values
            for entity in entities:
                if entity in timestamp_data[timestamp]:
                    last_state[entity] = timestamp_data[timestamp][entity]

            # Create row with all current states
            row = {}
            row.update({entity: last_state[entity] for entity in entities})

            # Filter weird rows
            if last_state[target_entity] == "":
                continue

            # Add only rows that have at least 2 actual values
            if sum(last_state[e] != "" for e in entities) >= 2:  # noqa: PLR2004
                pivot_rows.append(row)

        return pivot_rows
