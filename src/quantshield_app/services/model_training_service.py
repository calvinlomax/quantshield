"""Qt service for validating, launching, and monitoring model-training jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
import runpy
import shlex
import shutil
import sys

import psutil
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal
import torch

from quantshield.replay_durations import DEFAULT_REPLAY_DURATION_KEY, get_replay_duration_profile
from quantshield.training_logging import EVENT_PREFIX
from quantshield_app.services.input_parser import parse_ticker_input


TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")
ALLOWED_MODEL_SIZES = (10, 50)
TRAINING_MODES = ("portfolio_fit", "experiment", "rl_policy")
OUTPUT_CATEGORIES = (
    "portfolio_model_fits",
    "model_experiments",
    "model_experiments_50_suite",
    "rl_policy",
    "replay_checkpoint_suites",
)
DEFAULT_RL_OBJECTIVES = ("min_variance", "mean_variance", "risk_parity", "equal_weight")
MODE_OUTPUT_CHOICES = {
    "portfolio_fit": ("portfolio_model_fits", "model_experiments"),
    "experiment": ("model_experiments", "model_experiments_50_suite", "replay_checkpoint_suites"),
    "rl_policy": ("rl_policy", "model_experiments", "replay_checkpoint_suites"),
}


@dataclass(slots=True)
class ModelTrainingRequest:
    """Resolved UI inputs for a new training run."""

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    model_size: int = 10
    training_mode: str = "portfolio_fit"
    universe_source: str = "current_portfolio"
    tickers: list[str] = field(default_factory=list)
    start_date: str = "2018-01-01"
    end_date: str | None = None
    duration_key: str = DEFAULT_REPLAY_DURATION_KEY
    rebalance_frequency: str = "ME"
    benchmark_mode: str = "ticker"
    benchmark_ticker: str = "SPY"
    equal_weight_scope: str = "training_universe"
    output_category: str = "portfolio_model_fits"
    output_dir_override: str | None = None
    hyperparameters: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ResolvedTrainingLaunch:
    """Concrete command and metadata used for one training process."""

    script_path: Path
    python_executable: str
    arguments: list[str]
    command_text: str
    output_dir: Path
    benchmark_value: str
    benchmark_label: str
    tickers: list[str]
    resolved_hyperparameters: dict[str, object]
    compute_plan: dict[str, object]


class ModelTrainingService(QObject):
    """Validate requests, run scripts asynchronously, and stream training telemetry."""

    state_changed = Signal(str)
    log_received = Signal(str, str)
    event_received = Signal(object)
    run_started = Signal(object)
    run_finished = Signal(object)

    _active_service: "ModelTrainingService | None" = None

    def __init__(self, root: str | Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self._script_defaults_cache: dict[str, dict[str, object]] = {}
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._drain_stdout)
        self._process.readyReadStandardError.connect(self._drain_stderr)
        self._process.finished.connect(self._on_finished)
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._state = "idle"
        self._launch: ResolvedTrainingLaunch | None = None
        self._cancel_requested = False
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.setInterval(4000)
        self._kill_timer.timeout.connect(self._kill_process)
        self._utilization_timer = QTimer(self)
        self._utilization_timer.setInterval(1000)
        self._utilization_timer.timeout.connect(self._sample_utilization)
        self._utilization_sample_index = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def default_hyperparameters(self, *, mode: str, duration_key: str) -> dict[str, object]:
        profile = get_replay_duration_profile(duration_key)
        defaults = dict(self._load_script_defaults(mode))
        defaults["lookback_window"] = int(profile.lookback_window)
        defaults.setdefault("device", "auto")
        defaults.setdefault("optimizer", "adamw")
        defaults.setdefault("checkpoint_frequency", 0)
        defaults.setdefault("early_stopping_patience", 0)
        defaults["reward_comparison_mode"] = "best_of_selected"
        return defaults

    def output_categories_for_mode(self, mode: str) -> tuple[str, ...]:
        return MODE_OUTPUT_CHOICES.get(mode, ("model_experiments",))

    def build_temporary_output_dir(self, *, mode: str, duration_key: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = re.sub(r"[^A-Za-z0-9]+", "_", mode.strip().lower()).strip("_") or "training"
        safe_duration = re.sub(r"[^A-Za-z0-9]+", "_", duration_key.strip().lower()).strip("_") or DEFAULT_REPLAY_DURATION_KEY
        return self.root / "outputs" / "app_state" / "model_training_sessions" / safe_mode / safe_duration / timestamp

    @staticmethod
    def load_tickers_from_file(path: str | Path) -> list[str]:
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".csv":
            first_lines = text.splitlines()
            if first_lines and ("," in first_lines[0]):
                headers = [column.strip().lower() for column in first_lines[0].split(",")]
                if "ticker" in headers or "symbol" in headers:
                    target_index = headers.index("ticker") if "ticker" in headers else headers.index("symbol")
                    values = [row.split(",")[target_index] for row in first_lines[1:] if row.strip()]
                    return parse_ticker_input("\n".join(values))
        return parse_ticker_input(text)

    def resolve_request(self, request: ModelTrainingRequest) -> ResolvedTrainingLaunch:
        errors = self.validate_request(request)
        if errors:
            raise ValueError("\n".join(errors))

        tickers = [ticker.strip().upper() for ticker in request.tickers if ticker.strip()]
        defaults = self.default_hyperparameters(mode=request.training_mode, duration_key=request.duration_key)
        resolved_hyperparameters = {**defaults, **request.hyperparameters}
        compute_plan = self._evaluate_compute_capability(request, resolved_hyperparameters)
        resolved_hyperparameters = self._apply_compute_plan(
            request=request,
            resolved_hyperparameters=resolved_hyperparameters,
            compute_plan=compute_plan,
        )
        benchmark_value, benchmark_label = self._resolve_benchmark(request, tickers)
        output_dir = self._build_output_dir(request)
        script_path = self._script_path_for_mode(request.training_mode)
        arguments = self._build_arguments(
            request=request,
            tickers=tickers,
            benchmark_value=benchmark_value,
            resolved_hyperparameters=resolved_hyperparameters,
            output_dir=output_dir,
        )
        python_executable = self._python_executable()
        command_text = shlex.join([python_executable, script_path.as_posix(), *arguments])
        return ResolvedTrainingLaunch(
            script_path=script_path,
            python_executable=python_executable,
            arguments=arguments,
            command_text=command_text,
            output_dir=output_dir,
            benchmark_value=benchmark_value,
            benchmark_label=benchmark_label,
            tickers=tickers,
            resolved_hyperparameters=resolved_hyperparameters,
            compute_plan=compute_plan,
        )

    def validate_request(self, request: ModelTrainingRequest) -> list[str]:
        errors: list[str] = []
        name = request.name.strip()
        if not name:
            errors.append("Model name is required.")
        if request.training_mode not in TRAINING_MODES:
            errors.append("Training mode is invalid.")
        if int(request.model_size) not in ALLOWED_MODEL_SIZES:
            errors.append("Model size must be 10 or 50.")

        tickers = [ticker.strip().upper() for ticker in request.tickers if ticker.strip()]
        if len(tickers) < 5:
            errors.append("Select at least 5 tickers for the training universe.")
        if len(tickers) > int(request.model_size):
            errors.append(f"The selected universe has {len(tickers)} tickers but model size is {request.model_size}.")
        invalid_tickers = [ticker for ticker in tickers if not TICKER_PATTERN.fullmatch(ticker)]
        if invalid_tickers:
            errors.append(f"Invalid ticker symbols: {', '.join(invalid_tickers)}")

        if request.benchmark_mode == "ticker":
            ticker = request.benchmark_ticker.strip().upper()
            if not ticker:
                errors.append("Benchmark ticker is required.")
            elif not TICKER_PATTERN.fullmatch(ticker):
                errors.append(f"Benchmark ticker '{ticker}' is invalid.")
        elif request.benchmark_mode == "equal_weight":
            if not request.equal_weight_scope.strip():
                errors.append("Equal-weight benchmark scope must be defined.")
        elif request.benchmark_mode != "markowitz":
            errors.append("Benchmark selection is invalid.")

        if request.end_date and request.start_date > request.end_date:
            errors.append("Start date must be on or before end date.")
        if request.output_dir_override:
            override_path = Path(request.output_dir_override)
            if override_path.exists() and any(override_path.iterdir()):
                errors.append(f"Output path already exists and is not empty: {override_path.as_posix()}")
            elif not override_path.parent.exists():
                override_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            if request.output_category not in OUTPUT_CATEGORIES:
                errors.append("Output category is invalid.")
            elif request.output_category not in self.output_categories_for_mode(request.training_mode):
                errors.append("Output category is not supported for the chosen training mode.")
            if request.output_category == "model_experiments_50_suite" and int(request.model_size) != 50:
                errors.append("The 50-suite output category requires model size 50.")
        if not self._script_path_for_mode(request.training_mode).exists():
            errors.append("The backing training script does not exist.")

        try:
            output_dir = self._build_output_dir(request)
        except Exception as exc:
            errors.append(str(exc))
        else:
            if output_dir.exists() and any(output_dir.iterdir()):
                errors.append(f"Output path already exists and is not empty: {output_dir.as_posix()}")
            if not output_dir.parent.exists():
                output_dir.parent.mkdir(parents=True, exist_ok=True)

        return errors

    def save_trained_model(
        self,
        *,
        training_output_dir: str | Path,
        training_mode: str,
        model_size: int,
        duration_key: str,
        name: str,
        description: str,
        tags: list[str],
        output_category: str,
    ) -> Path:
        source_dir = Path(training_output_dir)
        if not source_dir.exists():
            raise ValueError(f"Training output directory does not exist: {source_dir.as_posix()}")
        request = ModelTrainingRequest(
            name=name,
            description=description,
            tags=list(tags),
            model_size=int(model_size),
            training_mode=training_mode,
            tickers=["SPY", "QQQ", "GLD", "IVV", "VOO"],
            duration_key=duration_key,
            output_category=output_category,
        )
        target_dir = self._build_output_dir(request)
        if target_dir.exists():
            raise ValueError(f"Target output path already exists: {target_dir.as_posix()}")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source_dir.as_posix(), target_dir.as_posix())
        metadata_path = target_dir / "model_metadata.json"
        metadata: dict[str, object] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "name": name,
                "description": description,
                "tags": list(tags),
                "output_dir": target_dir.as_posix(),
            }
        )
        model_path = target_dir / "actor_critic_policy.pt"
        if model_path.exists():
            metadata["model_path"] = model_path.as_posix()
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return model_path if model_path.exists() else target_dir

    def start_training(self, request: ModelTrainingRequest) -> ResolvedTrainingLaunch:
        if ModelTrainingService._active_service is not None and ModelTrainingService._active_service is not self:
            raise RuntimeError("Another model-training run is already active.")
        launch = self.resolve_request(request)
        self._launch = launch
        self._cancel_requested = False
        ModelTrainingService._active_service = self
        self._set_state("validating")
        self.run_started.emit(launch)
        self.log_received.emit(self._compute_plan_text(launch.compute_plan), "stdout")
        self.event_received.emit({"event": "compute_plan", "compute_plan": launch.compute_plan})

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONPATH", (self.root / "src").as_posix())
        environment.insert("MPLCONFIGDIR", "/tmp/mpl")
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(self.root.as_posix())
        self._set_state("running")
        psutil.cpu_percent(interval=None)
        self._utilization_sample_index = 0
        self._utilization_timer.start()
        self._process.start(launch.python_executable, [launch.script_path.as_posix(), *launch.arguments])
        return launch

    def cancel(self) -> None:
        if not self.is_running:
            return
        self._cancel_requested = True
        self._set_state("cancelled")
        self._process.terminate()
        self._kill_timer.start()

    def force_cancel(self, timeout_ms: int = 1500) -> bool:
        """Stop a training process immediately enough for UI teardown."""
        if not self.is_running:
            return True
        self._cancel_requested = True
        self._set_state("cancelled")
        self._kill_timer.stop()
        self._utilization_timer.stop()
        self._process.kill()
        return bool(self._process.waitForFinished(timeout_ms))

    def _kill_process(self) -> None:
        if self.is_running:
            self._process.kill()

    def _set_state(self, state: str) -> None:
        self._state = state
        self.state_changed.emit(state)

    def _drain_stdout(self) -> None:
        self._stdout_buffer += bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer = self._process_lines(self._stdout_buffer, stream="stdout")

    def _drain_stderr(self) -> None:
        self._stderr_buffer += bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_buffer = self._process_lines(self._stderr_buffer, stream="stderr")

    def _process_lines(self, buffer: str, *, stream: str) -> str:
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            clean = line.rstrip()
            if not clean:
                continue
            self.log_received.emit(clean, stream)
            event = self.parse_event_line(clean)
            if event is not None:
                self.event_received.emit(event)
        return buffer

    def _on_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._kill_timer.stop()
        self._utilization_timer.stop()
        success = exit_code == 0 and not self._cancel_requested
        state = "done" if success else ("cancelled" if self._cancel_requested else "failed")
        self._set_state(state)
        result = {
            "success": success,
            "cancelled": self._cancel_requested,
            "exit_code": int(exit_code),
            "output_dir": self._launch.output_dir if self._launch is not None else None,
        }
        ModelTrainingService._active_service = None
        self.run_finished.emit(result)

    def _sample_utilization(self) -> None:
        if not self.is_running:
            return
        memory = psutil.virtual_memory()
        event = {
            "event": "utilization_sample",
            "sample_index": self._utilization_sample_index,
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "memory_percent": float(memory.percent),
            "memory_used_gb": float(memory.used / (1024**3)),
            "memory_available_gb": float(memory.available / (1024**3)),
        }
        self._utilization_sample_index += 1
        self.event_received.emit(event)

    @staticmethod
    def parse_event_line(line: str) -> dict[str, object] | None:
        if not line.startswith(EVENT_PREFIX):
            return None
        try:
            return json.loads(line[len(EVENT_PREFIX) :].strip())
        except json.JSONDecodeError:
            return None

    def _load_script_defaults(self, mode: str) -> dict[str, object]:
        cached = self._script_defaults_cache.get(mode)
        if cached is not None:
            return dict(cached)

        script_path = self._script_path_for_mode(mode)
        namespace = runpy.run_path(script_path.as_posix())
        default_factory = namespace.get("default_cli_options")
        if not callable(default_factory):
            raise RuntimeError(f"{script_path.name} does not expose default_cli_options().")
        defaults = default_factory()
        if not isinstance(defaults, dict):
            raise RuntimeError(f"{script_path.name} returned invalid defaults.")
        self._script_defaults_cache[mode] = dict(defaults)
        return dict(defaults)

    def _build_output_dir(self, request: ModelTrainingRequest) -> Path:
        if request.output_dir_override:
            return Path(request.output_dir_override)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", request.name.strip().lower()).strip("_") or "new_model"
        duration = request.duration_key or DEFAULT_REPLAY_DURATION_KEY
        category = request.output_category
        if category == "portfolio_model_fits":
            return self.root / "outputs" / "portfolio_model_fits" / duration / slug
        if category == "model_experiments":
            return self.root / "outputs" / "model_experiments" / duration / slug
        if category == "model_experiments_50_suite":
            return self.root / "outputs" / "model_experiments_50_suite" / "portfolio_size_50" / duration / slug
        if category == "replay_checkpoint_suites":
            return self.root / "outputs" / "replay_checkpoint_suites" / duration / slug
        if category == "rl_policy":
            return self.root / "outputs" / "rl_policy" / slug
        raise ValueError("Unsupported output category.")

    def _build_arguments(
        self,
        *,
        request: ModelTrainingRequest,
        tickers: list[str],
        benchmark_value: str,
        resolved_hyperparameters: dict[str, object],
        output_dir: Path,
    ) -> list[str]:
        common = [
            "--name",
            request.name.strip(),
            "--description",
            request.description.strip(),
            "--duration-key",
            request.duration_key,
            "--start-date",
            request.start_date,
            "--output-dir",
            output_dir.as_posix(),
            "--model-size",
            str(request.model_size),
            "--benchmark",
            benchmark_value,
            "--rebalance-frequency",
            request.rebalance_frequency,
            "--lookback-window",
            str(resolved_hyperparameters["lookback_window"]),
            "--epochs",
            str(resolved_hyperparameters["epochs"]),
            "--batch-size",
            str(resolved_hyperparameters["batch_size"]),
            "--seed",
            str(resolved_hyperparameters["seed"]),
            "--learning-rate",
            str(resolved_hyperparameters["learning_rate"]),
            "--weight-decay",
            str(resolved_hyperparameters["weight_decay"]),
            "--dropout",
            str(resolved_hyperparameters["dropout"]),
            "--hidden-dim",
            str(resolved_hyperparameters["hidden_dim"]),
            "--attention-heads",
            str(resolved_hyperparameters["attention_heads"]),
            "--attention-layers",
            str(resolved_hyperparameters["attention_layers"]),
            "--actor-bc-weight",
            str(resolved_hyperparameters["actor_bc_weight"]),
            "--entropy-weight",
            str(resolved_hyperparameters["entropy_weight"]),
            "--validation-fraction",
            str(resolved_hyperparameters["validation_fraction"]),
            "--optimizer",
            str(resolved_hyperparameters["optimizer"]),
            "--checkpoint-frequency",
            str(resolved_hyperparameters["checkpoint_frequency"]),
            "--early-stopping-patience",
            str(resolved_hyperparameters["early_stopping_patience"]),
            "--reward-weight-raw",
            str(resolved_hyperparameters["reward_weight_raw"]),
            "--reward-weight-vs-benchmark",
            str(resolved_hyperparameters["reward_weight_vs_benchmark"]),
            "--reward-weight-vs-equal-weight",
            str(resolved_hyperparameters["reward_weight_vs_equal_weight"]),
            "--reward-weight-vs-restricted-random",
            str(resolved_hyperparameters["reward_weight_vs_restricted_random"]),
            "--reward-weight-vs-markowitz",
            str(resolved_hyperparameters["reward_weight_vs_markowitz"]),
            "--reward-comparison-mode",
            str(resolved_hyperparameters.get("reward_comparison_mode", "best_of_selected")),
        ]
        if request.end_date:
            common.extend(["--end-date", request.end_date])
        if request.tags:
            common.extend(["--tags", *request.tags])
        device_value = str(resolved_hyperparameters.get("device", "auto")).strip().lower()
        if device_value:
            common.extend(["--device", device_value])

        if request.training_mode == "portfolio_fit":
            return [
                *common,
                "--benchmark-mode",
                request.benchmark_mode,
                "--candidate-mode",
                str(resolved_hyperparameters["candidate_mode"]),
                "--tickers",
                *tickers,
            ]
        if request.training_mode == "experiment":
            return [
                *common,
                "--portfolio-size",
                str(request.model_size),
                "--candidate-pool-size",
                str(min(int(resolved_hyperparameters["candidate_pool_size"]), len(tickers))),
                "--random-universes",
                str(resolved_hyperparameters["random_universes"]),
                "--universe-tickers",
                *tickers,
            ]
        return [
            *common,
            "--suite-root",
            (self.root / "outputs" / "ml_tuned_objective_runs").as_posix(),
            "--objectives",
            *[str(item) for item in resolved_hyperparameters.get("objectives", DEFAULT_RL_OBJECTIVES)],
            "--tickers",
            *tickers,
        ]

    def _resolve_benchmark(self, request: ModelTrainingRequest, tickers: list[str]) -> tuple[str, str]:
        if request.benchmark_mode == "equal_weight":
            return "__equal_weight__", f"Equal Weight ({', '.join(tickers)})"
        if request.benchmark_mode == "markowitz":
            return "__markowitz__", f"Markowitz Mean-Variance ({', '.join(tickers)})"
        return request.benchmark_ticker.strip().upper(), request.benchmark_ticker.strip().upper()

    @staticmethod
    def _determine_best_device() -> str:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _evaluate_compute_capability(
        self,
        request: ModelTrainingRequest,
        resolved_hyperparameters: dict[str, object],
    ) -> dict[str, object]:
        memory = psutil.virtual_memory()
        total_ram_gb = float(memory.total / (1024**3))
        available_ram_gb = float(memory.available / (1024**3))
        logical_cores = int(psutil.cpu_count(logical=True) or 1)
        physical_cores = int(psutil.cpu_count(logical=False) or logical_cores)
        requested_device = str(resolved_hyperparameters.get("device", "auto")).strip().lower()
        resolved_device = self._determine_best_device() if requested_device == "auto" else requested_device

        if int(request.model_size) >= 50:
            recommended_batch_size = 16 if available_ram_gb < 8.0 else (32 if available_ram_gb < 16.0 else 64)
        else:
            recommended_batch_size = 32 if available_ram_gb < 8.0 else (64 if available_ram_gb < 16.0 else 96)
        if resolved_device == "cpu":
            recommended_batch_size = min(recommended_batch_size, 64 if int(request.model_size) <= 10 else 32)

        recommended_candidate_pool = None
        recommended_random_universes = None
        if request.training_mode == "experiment":
            if available_ram_gb < 8.0:
                recommended_candidate_pool = 24
                recommended_random_universes = 96
            elif available_ram_gb < 16.0:
                recommended_candidate_pool = 48
                recommended_random_universes = 160
            else:
                recommended_candidate_pool = 80 if resolved_device != "cpu" else 64
                recommended_random_universes = 320 if resolved_device != "cpu" else 224

        notes: list[str] = []
        if requested_device == "auto":
            notes.append(f"auto-selected device={resolved_device}")
        if int(resolved_hyperparameters.get("batch_size", recommended_batch_size)) > recommended_batch_size:
            notes.append(f"batch capped to {recommended_batch_size} for local memory")
        if recommended_candidate_pool is not None and int(resolved_hyperparameters.get("candidate_pool_size", recommended_candidate_pool)) > recommended_candidate_pool:
            notes.append(f"candidate pool capped to {recommended_candidate_pool}")
        if recommended_random_universes is not None and int(resolved_hyperparameters.get("random_universes", recommended_random_universes)) > recommended_random_universes:
            notes.append(f"random universes capped to {recommended_random_universes}")

        return {
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "total_ram_gb": round(total_ram_gb, 2),
            "available_ram_gb": round(available_ram_gb, 2),
            "resolved_device": resolved_device,
            "recommended_batch_size": int(recommended_batch_size),
            "recommended_candidate_pool_size": recommended_candidate_pool,
            "recommended_random_universes": recommended_random_universes,
            "notes": notes,
        }

    def _apply_compute_plan(
        self,
        *,
        request: ModelTrainingRequest,
        resolved_hyperparameters: dict[str, object],
        compute_plan: dict[str, object],
    ) -> dict[str, object]:
        adjusted = dict(resolved_hyperparameters)
        adjusted["device"] = str(compute_plan["resolved_device"])
        adjusted["batch_size"] = min(
            int(adjusted.get("batch_size", compute_plan["recommended_batch_size"])),
            int(compute_plan["recommended_batch_size"]),
        )
        if request.training_mode == "experiment":
            candidate_cap = compute_plan.get("recommended_candidate_pool_size")
            if candidate_cap is not None:
                adjusted["candidate_pool_size"] = min(int(adjusted.get("candidate_pool_size", candidate_cap)), int(candidate_cap))
            random_cap = compute_plan.get("recommended_random_universes")
            if random_cap is not None:
                adjusted["random_universes"] = min(int(adjusted.get("random_universes", random_cap)), int(random_cap))
        return adjusted

    @staticmethod
    def _compute_plan_text(compute_plan: dict[str, object]) -> str:
        notes = compute_plan.get("notes") or []
        notes_text = f" | {', '.join(str(item) for item in notes)}" if notes else ""
        return (
            "Compute plan: "
            f"{compute_plan['physical_cores']} physical / {compute_plan['logical_cores']} logical cores, "
            f"{compute_plan['available_ram_gb']:.2f} GB free of {compute_plan['total_ram_gb']:.2f} GB, "
            f"device={compute_plan['resolved_device']}, "
            f"batch={compute_plan['recommended_batch_size']}"
            f"{notes_text}"
        )

    def _script_path_for_mode(self, mode: str) -> Path:
        filename = {
            "portfolio_fit": "fit_portfolio_model.py",
            "experiment": "train_random_sp500_policy.py",
            "rl_policy": "train_rl_policy.py",
        }[mode]
        return self.root / "scripts" / filename

    def _python_executable(self) -> str:
        venv_python = self.root / ".venv" / "bin" / "python"
        return venv_python.as_posix() if venv_python.exists() else sys.executable
