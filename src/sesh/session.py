from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any

from ._optional import HAS_MLFLOW, HAS_TORCH
from .cards.dataset import DatasetCard
from .cards.model import ModelCard
from .logger import configure_logger
from .seed import set_seed

if HAS_TORCH:
    import torch

if HAS_MLFLOW:
    import mlflow

DEFAULT_OUTPUT_ROOT = Path("outputs")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def _resolve_device(device: str) -> str:
    """Normalise a device string, using torch if it's installed.

    Args:
        device: Target hardware accelerator string (e.g., 'cpu', 'cuda').

    Returns:
        The normalised device string, e.g. torch expands 'cuda' to
        'cuda:0'. If torch isn't installed, the input is returned as-is.
    """
    if HAS_TORCH:
        return str(torch.device(device))
    return device


class Session:
    """The master workspace conductor for independent scientific research.

    This class serves as the sole public interface for experiment tracking.
    It encapsulates directory routing, reproducibility seeding, local text-logging,
    scientific card generation, and invisible background MLflow telemetry.
    """

    def __init__(
        self,
        name: str,
        *,
        seed: int = 42,
        device: str = "cpu",
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        deterministic_seed: bool = True,
        enable_mlflow: bool = True,
        mlflow_tracking_uri: str | None = None,
    ) -> None:
        """
        Initialise and align local outputs, logging streams, and MLflow targets.

        Args:
            name: The distinct name of the experimental run.
            seed: Base numerical seed used to initialise deterministic engines.
            device: Target hardware accelerator string (e.g., 'cpu', 'cuda').
            output_root: Root directory path where session outputs are written.
            deterministic_seed: If True, forces strict hardware algorithm determinism.
            enable_mlflow: If True, activates parallel background server tracking.
            mlflow_tracking_uri: Optional explicit MLflow tracking URI (e.g. a
                shared team server). If omitted, defaults to a SQLite store
                scoped inside this session's own output directory.
        """
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        self.name = name
        self.seed = seed
        self.device = _resolve_device(device)

        # Resolve unified output directory with incremental collision handling
        base_dir = output_root / f"{timestamp}_{name}"
        counter = 0
        self.output_dir = base_dir

        while self.output_dir.exists():
            counter += 1
            self.output_dir = Path(f"{base_dir}_{counter}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialise the underlying text logger encapsulated within the session
        self._logger = configure_logger(
            name=name,
            log_file=self.output_dir / "run.log",
        )

        # Enforce multi-engine reproducibility conditions via the seed package
        set_seed(seed, deterministic=deterministic_seed)

        self._logger.info(f"Experiment session: {self.name} initialised.")
        self._logger.info(f"Seeding engine completed using base reference: {seed}")
        self._logger.info(
            f"Local workflow runtime tracking directed to: {self.output_dir}"
        )

        # Configure background MLflow tracking parameters securely
        self._use_mlflow = HAS_MLFLOW and enable_mlflow
        if self._use_mlflow:
            artifact_location: str | None = None
            if mlflow_tracking_uri is not None:
                tracking_uri = mlflow_tracking_uri
            else:
                # MLflow's file-based tracking store is in maintenance mode and
                # can raise MlflowException, so default to a per-session SQLite
                # store instead, keeping artifacts scoped inside this session's
                # own directory.
                tracking_uri = f"sqlite:///{(self.output_dir / 'mlflow.db').resolve()}"
                artifact_location = (self.output_dir / "mlruns").resolve().as_uri()

            mlflow.set_tracking_uri(tracking_uri)

            if artifact_location is not None:
                with contextlib.suppress(mlflow.MlflowException):
                    # Raises if the experiment already exists in this store.
                    mlflow.create_experiment(
                        self.name, artifact_location=artifact_location
                    )

            mlflow.set_experiment(self.name)
            self._mlflow_run = mlflow.start_run(run_name=f"{timestamp}_{name}")

            # Log structural baseline parameters instantly
            mlflow.log_params(
                {
                    "session_seed": self.seed,
                    "session_device": self.device,
                    "deterministic_seed": deterministic_seed,
                }
            )
        elif enable_mlflow and not HAS_MLFLOW:
            self._logger.warning(
                "MLflow tracking requested, but package is missing from environment."
            )

    def __enter__(self) -> Session:
        """Enable contextual resource wrapping via the 'with' statement layout."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Automatically close logging conduits and disconnect active backend MLflow
        tracks.
        """
        if exc_type is not None:
            self._logger.error(
                f"Execution boundary broken by exception: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb),
            )
            if self._use_mlflow:
                mlflow.end_run(status="FAILED")
        else:
            if self._use_mlflow:
                mlflow.end_run(status="FINISHED")

        self._logger.info("Session workspace locked cleanly.")

    def path(self, *parts: str | Path) -> Path:
        """
        Resolve an absolute path targeting a destination inside this session container.

        Args:
            *parts: A sequence of string fragments or Path components.

        Returns:
            The combined and absolute Path object pointing inside the session directory.
        """
        resolved_path = Path(self.output_dir, *parts)
        resolved_path.mkdir(parents=True, exist_ok=True)
        return resolved_path

    def info(self, msg: str) -> None:
        """
        Log a standard informational string message to the local session log file.

        Args:
            msg: The textual message string to record.
        """
        self._logger.info(msg)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Record numerical metrics across terminal console targets and MLflow dashboards.

        Args:
            metrics: Dictionary mapping metric names to numerical values.
            step: Optional training step or epoch index.
        """
        step_prefix = f"[Step {step:04d}] " if step is not None else ""
        metric_elements = [
            f"{name}: {value:.5f}" if isinstance(value, float) else f"{name}: {value}"
            for name, value in metrics.items()
        ]

        self._logger.info(f"{step_prefix}" + " | ".join(metric_elements))

        if self._use_mlflow:
            mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log architectural attributes or pipeline flags to logs and server engines.

        Args:
            params: Dictionary containing configuration metadata parameters.
        """
        self._logger.info(f"Parameters registered: {params}")
        if self._use_mlflow:
            mlflow.log_params(params)

    def log_dataset_card(
        self,
        *,
        name: str,
        parameters: dict[str, Any],
        description: str | tuple[str] | list[str],
        sub_folder: str = "data_artifacts",
    ) -> None:
        """
        Mint a DatasetCard and automatically mirror the structural fields to MLflow.

        Args:
            generator: Name of the dataset generator algorithm.
            parameters: Data configurations or pipeline weights used.
            description: Summary details detailing data features or criteria.
            sub_folder: Target subdirectory inside the session directory.
        """
        target_dir = self.path(sub_folder)

        card = DatasetCard(
            name=name,
            parameters=parameters,
            seed=self.seed,
            device=self.device,
            description=description,
        )
        markdown_path, _config_path = card.save(target_dir)
        self._logger.info(f"Dataset card artifact serialised to: {markdown_path}")

        if self._use_mlflow:
            mlflow.log_params({f"data_{k}": v for k, v in parameters.items()})
            mlflow.log_artifact(str(markdown_path), artifact_path=sub_folder)

    def log_model_card(
        self,
        *,
        name: str,
        architecture: str,
        parameters: dict[str, Any],
        intended_use: list[str],
        limitations: list[str],
        training_metadata: dict[str, Any] | None = None,
        description: str | tuple[str] | list[str] = "",
        sub_folder: str = "model_artifacts",
    ) -> None:
        """Mint a ModelCard and automatically mirror the structural fields to MLflow.

        Args:
            name: Name of the trained model architecture.
            architecture: Class structural string name of the network framework.
            parameters: Key-value hyperparameters of model structure.
            intended_use: Planned list contexts for model usage.
            limitations: Known parameters under which model accuracy degrades.
            training_metadata: Final loss figures, epochs, or performance markers.
            description: Informational textual summary text blocks.
            sub_folder: Target subdirectory inside the session directory.
        """
        target_dir = self.path(sub_folder)
        full_training_meta = training_metadata.copy() if training_metadata else {}

        card = ModelCard(
            name=name,
            description=description,
            architecture=architecture,
            parameters=parameters,
            intended_use=intended_use or [],
            limitations=limitations,
            training=full_training_meta,
        )
        saved_path = card.save(target_dir)
        self._logger.info(f"Model card artifact serialised to: {saved_path}")

        if self._use_mlflow:
            mlflow.log_params({f"model_{k}": v for k, v in parameters.items()})
            mlflow.log_params({f"train_{k}": v for k, v in full_training_meta.items()})
            mlflow.log_param("model_architecture", architecture)
            mlflow.log_artifact(str(saved_path), artifact_path=sub_folder)


if __name__ == "__main__":
    print("=== Showcasing Complete Integrated Public Session API ===")

    # 1. Pipeline initialisation context block
    # Provisions the directory, configures loggers, sets deterministic seeds,
    # and safely sets up isolated internal MLflow tracks under the hood.
    target_hardware = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    with Session(
        name="comprehensive_scientific_run", seed=8888, device=target_hardware
    ) as session:
        # 2. Plain-text information logging via the public wrapper shortcut
        session.info("Commencing execution showcase pipeline step 1: Setup validation.")

        # 3. Parameter logging
        # Synchronises configurations effortlessly across local run.log files
        # and the MLflow instance
        session.log_params(
            {
                "learning_rate_initial": 0.0005,
                "weight_decay_coefficient": 1e-4,
                "dataset_split_ratio": [0.8, 0.1, 0.1],
            }
        )

        # 4. Dataset Card generation
        # Generates structured documentation and registers parameters directly
        # into active session space
        session.info("Commencing execution showcase pipeline step 2: Data synthesis.")
        session.log_dataset_card(
            name="GaussianNoiseFieldSynthesiser",
            parameters={"spatial_resolution": 2048, "noise_floor_db": -60.0},
            description=[
                "High-resolution synthetic validation grid.",
                "Generated to evaluate convergence under extreme boundary conditions.",
            ],
        )

        # 5. Metric streaming loop
        # Sequentially prints to the terminal console, logs to file,
        # and updates MLflow real-time graphs
        session.info(
            "Commencing execution showcase pipeline step 3: Training emulation loops."
        )
        for epoch in range(1, 4):
            simulated_loss = 0.85 / (epoch**0.5)
            simulated_metric = 0.72 + (0.06 * epoch)

            session.log_metrics(
                metrics={
                    "objective_loss": simulated_loss,
                    "validation_accuracy": simulated_metric,
                },
                step=epoch,
            )

        # 6. Model Card generation
        # Finalises documentation, packages metadata, and links absolute environment
        # tracking states
        session.info(
            "Commencing execution pipeline step 4: Model preservation tracking."
        )
        session.log_model_card(
            name="AnomalyClassifierNet",
            architecture="ResNetBackbone_S3",
            parameters={"hidden_channels": 256, "dropout_probability": 0.3},
            intended_use=[
                "Predictive anomaly classification within non-uniform tensor grids."
            ],
            limitations=[
                "Degrades rapidly when processing input domains outside trained bounds."
            ],
            training_metadata={"epochs_executed": 3, "optimal_loss_attained": 0.490},
            description="Trained model optimised for structural matrix parsing.",
        )

    # Context manager automatically triggers clean resource cleanup
    # and finalises connections
    print(
        "\nExecution loop finalised successfully. "
        + f"Review all local run outputs at: {session.output_dir}"
    )
    print("=== Showcasing Completed ===")
