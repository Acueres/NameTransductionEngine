"""Models learned from the datasets: built after them by `nte init`, stale when
their source data or their training rules change.

Each model registers a builder and a state check here. Models are files under
`data/models/` so that the code reading them needs no database connection.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from name_transduction_engine.paths import PERSIAN_MODEL_PATH

from .persian_romanization.data_provision import (
    ModelState,
    ensure_persian_model,
    persian_model_state,
)

__all__ = ["ensure_models", "collect_model_status", "ModelStatus"]


@dataclass(frozen=True)
class _Model:
    name: str
    description: str
    path: Path
    ensure: Callable[[bool], None]
    state: Callable[[], ModelState]


MODELS: tuple[_Model, ...] = (
    _Model(
        "persian",
        "Persian display romanization, learned from GeoNames",
        PERSIAN_MODEL_PATH,
        ensure_persian_model,
        persian_model_state,
    ),
)


def ensure_models(force: bool = False) -> None:
    """Build every model that is missing or stale (all of them with `force`).
    Runs after the datasets, which the models are learned from"""
    for model in MODELS:
        model.ensure(force)


@dataclass(frozen=True)
class ModelStatus:
    name: str
    description: str
    path: Path
    ready: bool
    reason: str | None  # why it is not ready, or a caveat when it is
    size_bytes: int | None
    meta: dict


def collect_model_status() -> list[ModelStatus]:
    statuses = []
    for model in MODELS:
        state = model.state()
        size = model.path.stat().st_size if model.path.is_file() else None
        statuses.append(
            ModelStatus(
                model.name,
                model.description,
                model.path,
                state.ready,
                state.reason,
                size,
                state.meta,
            )
        )
    return statuses
