"""Optional MLflow tracking that never blocks offline experiments."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def tracked_run(name: str, params: dict | None = None, metrics: dict | None = None, artifacts: list[str] | None = None) -> Iterator[object | None]:
    try:
        import mlflow
    except ImportError:  # optional dependency
        yield None
        return
    with mlflow.start_run(run_name=name) as run:
        if params:
            mlflow.log_params({key: str(value) for key, value in params.items()})
        if metrics:
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        for artifact in artifacts or []:
            mlflow.log_artifact(artifact)
        yield run

