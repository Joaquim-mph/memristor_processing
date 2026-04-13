"""
Formal pipeline builder for chaining data processing commands.

This module provides a declarative way to define, execute, and manage
multi-step data processing pipelines with error handling, checkpointing,
and rollback capabilities.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


class StepStatus(str, Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    name: str
    command: Callable
    kwargs: Dict[str, Any] = field(default_factory=dict)
    rollback_fn: Optional[Callable] = None
    skip_on_error: bool = False  # Continue pipeline even if this step fails
    retry_count: int = 0
    retry_delay: float = 1.0  # seconds
    checkpoint: bool = True  # Save checkpoint after this step

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    output: Any = None

    @property
    def elapsed_time(self) -> Optional[float]:
        """Calculate elapsed time for this step."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> dict:
        """Serialize step state (for checkpointing)."""
        return {
            "name": self.name,
            "status": self.status.value,
            "elapsed_time": self.elapsed_time,
            "error": str(self.error) if self.error else None,
        }


class PipelineState:
    """Manages pipeline execution state and checkpointing."""

    def __init__(self, name: str, checkpoint_dir: Optional[Path] = None):
        self.name = name
        self.checkpoint_dir = checkpoint_dir or Path("data/.pipeline_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_checkpoint(self, steps: List[PipelineStep], metadata: dict = None):
        """Save current pipeline state to disk."""
        checkpoint_file = self.checkpoint_dir / f"{self.name}_{self.execution_id}.json"

        state = {
            "pipeline_name": self.name,
            "execution_id": self.execution_id,
            "timestamp": datetime.now().isoformat(),
            "steps": [step.to_dict() for step in steps],
            "metadata": metadata or {},
        }

        checkpoint_file.write_text(json.dumps(state, indent=2))
        return checkpoint_file

    def load_checkpoint(self, execution_id: Optional[str] = None) -> Optional[dict]:
        """Load the most recent checkpoint (or specific execution)."""
        if execution_id:
            checkpoint_file = self.checkpoint_dir / f"{self.name}_{execution_id}.json"
            if checkpoint_file.exists():
                return json.loads(checkpoint_file.read_text())
            return None

        # Find most recent checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob(f"{self.name}_*.json"))
        if checkpoints:
            return json.loads(checkpoints[-1].read_text())
        return None

    def clear_checkpoints(self):
        """Remove all checkpoints for this pipeline."""
        for checkpoint in self.checkpoint_dir.glob(f"{self.name}_*.json"):
            checkpoint.unlink()


class Pipeline:
    """
    Formal pipeline builder for chaining commands with error handling and rollback.

    Example:
        ```python
        pipeline = Pipeline("full-pipeline", description="Complete data processing")

        pipeline.add_step(
            "stage-raw-data",
            stage_all_command,
            raw_root=Path("data/01_raw"),
            workers=8,
            rollback_fn=lambda: shutil.rmtree("data/02_stage/raw_measurements"),
        )

        pipeline.add_step(
            "build-histories",
            build_all_histories_command,
            manifest_path=Path("data/02_stage/raw_measurements/_manifest/manifest.parquet"),
        )

        pipeline.add_step(
            "derive-metrics",
            derive_all_metrics_command,
            skip_on_error=True,  # Optional step
        )

        result = pipeline.execute(stop_on_error=True, enable_rollback=True)
        ```
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        self.name = name
        self.description = description
        self.steps: List[PipelineStep] = []
        self.state = PipelineState(name, checkpoint_dir)
        self.verbose = verbose

    def add_step(
        self,
        name: str,
        command: Callable,
        rollback_fn: Optional[Callable] = None,
        skip_on_error: bool = False,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        checkpoint: bool = True,
        **kwargs,
    ) -> "Pipeline":
        """
        Add a step to the pipeline.

        Args:
            name: Human-readable step name
            command: Callable command function to execute
            rollback_fn: Optional function to undo this step's effects
            skip_on_error: If True, pipeline continues even if this step fails
            retry_count: Number of times to retry on failure (0 = no retry)
            retry_delay: Seconds to wait between retries
            checkpoint: Save checkpoint after this step completes
            **kwargs: Arguments to pass to the command function

        Returns:
            self (for method chaining)
        """
        step = PipelineStep(
            name=name,
            command=command,
            kwargs=kwargs,
            rollback_fn=rollback_fn,
            skip_on_error=skip_on_error,
            retry_count=retry_count,
            retry_delay=retry_delay,
            checkpoint=checkpoint,
        )
        self.steps.append(step)
        return self

    def execute(
        self,
        stop_on_error: bool = True,
        enable_rollback: bool = False,
        resume_from: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute the pipeline.

        Args:
            stop_on_error: If True, stop pipeline on first error (unless step has skip_on_error=True)
            enable_rollback: If True, rollback completed steps on failure
            resume_from: Resume from checkpoint (execution_id or "latest")

        Returns:
            PipelineResult with execution details
        """
        start_time = time.time()

        # Handle resume
        start_index = 0
        if resume_from:
            checkpoint = self.state.load_checkpoint(
                None if resume_from == "latest" else resume_from
            )
            if checkpoint:
                console.print(f"[cyan]Resuming from checkpoint: {checkpoint['execution_id']}[/cyan]")
                # Skip already completed steps
                completed_steps = {s["name"]: s for s in checkpoint["steps"] if s["status"] == "success"}
                start_index = len(completed_steps)

        # Display pipeline header
        self._display_header()

        # Execute steps
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            pipeline_task = progress.add_task(
                f"[cyan]Pipeline: {self.name}", total=len(self.steps)
            )

            for i, step in enumerate(self.steps[start_index:], start=start_index):
                progress.update(pipeline_task, description=f"[cyan]Step {i+1}/{len(self.steps)}: {step.name}")

                success = self._execute_step(step, progress)
                progress.advance(pipeline_task)

                if not success:
                    if enable_rollback:
                        console.print(f"\n[yellow]⚠ Rolling back completed steps...[/yellow]")
                        self.rollback()

                    if stop_on_error and not step.skip_on_error:
                        break

                # Save checkpoint after successful step
                if success and step.checkpoint:
                    checkpoint_file = self.state.save_checkpoint(
                        self.steps,
                        metadata={"last_completed_step": step.name}
                    )
                    if self.verbose:
                        console.print(f"[dim]Checkpoint saved: {checkpoint_file}[/dim]")

        elapsed = time.time() - start_time
        result = self._build_result(elapsed)
        self._display_summary(result)

        return result

    def _execute_step(self, step: PipelineStep, progress: Progress) -> bool:
        """Execute a single step with retry logic."""
        # Only print header if step name doesn't start with "STEP" (custom headers)
        if not step.name.startswith("STEP"):
            console.print(f"\n[bold cyan]{'═'*3} {step.name.upper()} {'═'*3}[/bold cyan]\n")

        step.status = StepStatus.RUNNING
        step.start_time = time.time()

        attempts = 0
        max_attempts = step.retry_count + 1

        while attempts < max_attempts:
            try:
                if self.verbose and attempts > 0:
                    console.print(f"[yellow]Retry {attempts}/{step.retry_count}[/yellow]")

                # Execute command
                step.output = step.command(**step.kwargs)

                # Success
                step.status = StepStatus.SUCCESS
                step.end_time = time.time()
                # Only print completion message for non-custom steps
                if not step.name.startswith("STEP"):
                    console.print(f"[green]✓ {step.name} completed in {step.elapsed_time:.1f}s[/green]")
                return True

            except SystemExit as e:
                # Handle Typer command exits
                if e.code != 0:
                    step.error = Exception(f"Command exited with code {e.code}")
                    break
                else:
                    # Successful exit
                    step.status = StepStatus.SUCCESS
                    step.end_time = time.time()
                    return True

            except Exception as e:
                step.error = e
                attempts += 1

                if attempts < max_attempts:
                    console.print(f"[yellow]⚠ Error: {e}. Retrying in {step.retry_delay}s...[/yellow]")
                    time.sleep(step.retry_delay)
                else:
                    break

        # Failed after all retries
        step.status = StepStatus.FAILED
        step.end_time = time.time()
        console.print(f"[red]✗ {step.name} failed: {step.error}[/red]")

        if step.skip_on_error:
            console.print(f"[yellow]⚠ Continuing pipeline (skip_on_error=True)[/yellow]")
            step.status = StepStatus.SKIPPED
            return True

        return False

    def rollback(self):
        """Rollback all completed steps in reverse order."""
        for step in reversed(self.steps):
            if step.status == StepStatus.SUCCESS and step.rollback_fn:
                console.print(f"[yellow]↩ Rolling back: {step.name}[/yellow]")
                try:
                    step.rollback_fn()
                    step.status = StepStatus.ROLLED_BACK
                    console.print(f"[green]✓ Rolled back: {step.name}[/green]")
                except Exception as e:
                    console.print(f"[red]✗ Rollback failed for {step.name}: {e}[/red]")

    def _display_header(self):
        """Display pipeline header."""
        steps_text = "\n".join(f"{i+1}. {step.name}" for i, step in enumerate(self.steps))
        console.print(Panel.fit(
            f"[bold blue]{self.name}[/bold blue]\n"
            f"{self.description}\n\n"
            f"[cyan]Steps:[/cyan]\n{steps_text}",
            border_style="blue"
        ))
        console.print()

    def _build_result(self, elapsed: float) -> "PipelineResult":
        """Build result summary."""
        success_count = sum(1 for s in self.steps if s.status == StepStatus.SUCCESS)
        failed_count = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        skipped_count = sum(1 for s in self.steps if s.status == StepStatus.SKIPPED)

        return PipelineResult(
            pipeline_name=self.name,
            total_steps=len(self.steps),
            successful_steps=success_count,
            failed_steps=failed_count,
            skipped_steps=skipped_count,
            total_time=elapsed,
            steps=self.steps,
        )

    def _display_summary(self, result: "PipelineResult"):
        """Display execution summary."""
        console.print(f"\n{'='*80}\n")

        if result.failed_steps > 0:
            border_style = "red"
            status_text = "[bold red]✗ Pipeline Failed[/bold red]"
        else:
            border_style = "green"
            status_text = "[bold green]✓ Pipeline Complete[/bold green]"

        summary = (
            f"{status_text}\n\n"
            f"Total time: {result.total_time:.1f}s\n"
            f"Steps: {result.successful_steps}/{result.total_steps} successful"
        )

        if result.failed_steps > 0:
            summary += f", {result.failed_steps} failed"
        if result.skipped_steps > 0:
            summary += f", {result.skipped_steps} skipped"

        # Add failed step details
        if result.failed_steps > 0:
            failed_steps = [s for s in self.steps if s.status == StepStatus.FAILED]
            summary += "\n\n[red]Failed steps:[/red]\n"
            summary += "\n".join(f"  • {s.name}: {s.error}" for s in failed_steps)

        console.print(Panel.fit(summary, border_style=border_style))

    def to_yaml(self, path: Path):
        """Export pipeline definition to YAML for reuse."""
        import yaml

        definition = {
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "name": step.name,
                    "command": step.command.__name__,
                    "kwargs": step.kwargs,
                    "skip_on_error": step.skip_on_error,
                    "retry_count": step.retry_count,
                }
                for step in self.steps
            ]
        }

        path.write_text(yaml.dump(definition, indent=2))
        console.print(f"[green]Pipeline definition saved to: {path}[/green]")

    @classmethod
    def from_yaml(cls, path: Path, command_registry: Dict[str, Callable]) -> "Pipeline":
        """Load pipeline definition from YAML."""
        import yaml

        definition = yaml.safe_load(path.read_text())

        pipeline = cls(
            name=definition["name"],
            description=definition.get("description", ""),
        )

        for step_def in definition["steps"]:
            command_name = step_def["command"]
            if command_name not in command_registry:
                raise ValueError(f"Unknown command: {command_name}")

            pipeline.add_step(
                name=step_def["name"],
                command=command_registry[command_name],
                skip_on_error=step_def.get("skip_on_error", False),
                retry_count=step_def.get("retry_count", 0),
                **step_def.get("kwargs", {})
            )

        return pipeline


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_name: str
    total_steps: int
    successful_steps: int
    failed_steps: int
    skipped_steps: int
    total_time: float
    steps: List[PipelineStep]

    @property
    def success(self) -> bool:
        """Whether pipeline completed successfully."""
        return self.failed_steps == 0
