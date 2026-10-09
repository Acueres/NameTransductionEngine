"""Models learned from the datasets: built after them by `nte init`, stale when
their source data or their training rules change.

Each model registers a builder and a state check here. Models are files under
`data/models/` so that the code reading them needs no database connection.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .arabic_romanization.data_provision import SPEC as ARABIC_SPEC
from .geonames_model import GeoNamesModelSpec, ModelState, ensure_model, model_state
from .persian_romanization.data_provision import SPEC as PERSIAN_SPEC

__all__ = ["ensure_models", "collect_model_status", "ModelStatus"]


@dataclass(frozen=True)
class _Model:
    name: str
    description: str
    path: Path
    ensure: Callable[[bool], None]
    state: Callable[[], ModelState]


def _geonames_model(spec: GeoNamesModelSpec, description: str) -> _Model:
    return _Model(
        spec.name,
        description,
        spec.path,
        lambda force: ensure_model(spec, force),
        lambda: model_state(spec),
    )


MODELS: tuple[_Model, ...] = (
    _geonames_model(
        PERSIAN_SPEC, "Persian display romanization, learned from GeoNames"
    ),
    _geonames_model(ARABIC_SPEC, "Arabic display romanization, learned from GeoNames"),
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
