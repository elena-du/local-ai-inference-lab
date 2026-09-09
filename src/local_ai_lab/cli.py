from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from local_ai_lab.adapters.registry import ADAPTERS, create_adapter
from local_ai_lab.benchmarking.report import read_records, write_markdown_report
from local_ai_lab.benchmarking.runner import run_benchmark, run_experiment
from local_ai_lab.config import ModelConfig, ROOT, Workload
from local_ai_lab.education.compare import explain_differences
from local_ai_lab.education.render import render_path, render_probe, render_responsibilities
from local_ai_lab.hardware import discover_hardware


app = typer.Typer(no_args_is_help=True, help="Learn, inspect, run, and compare Windows AI inference paths.")
console = Console()


@app.command()
def hardware(
    output: Annotated[Path | None, typer.Option(help="Optional JSON output path.")] = None,
) -> None:
    """Discover the actual OS and hardware; device presence is not proof of inference use."""
    snapshot = discover_hardware()
    data = snapshot.to_dict()
    console.print_json(data=data)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        console.print(f"Saved [link={output.resolve()}]{output}[/link]")


@app.command()
def runtimes() -> None:
    """Probe all runtime adapters without importing unavailable optional dependencies."""
    for adapter_id in ADAPTERS:
        render_probe(console, adapter_id, create_adapter(adapter_id).probe())


@app.command()
def doctor() -> None:
    """Diagnose hardware and optional runtime readiness."""
    snapshot = discover_hardware()
    console.print(f"[bold]OS:[/bold] {snapshot.os.get('caption', snapshot.os.get('system'))} build {snapshot.os.get('buildnumber', snapshot.os.get('version'))}")
    console.print(f"[bold]Architecture:[/bold] {snapshot.architecture}")
    console.print(f"[bold]CPU:[/bold] {snapshot.cpu['name']}")
    for device in snapshot.devices:
        console.print(f"[bold]{device.kind} present:[/bold] {device.name} ({device.status})")
    console.print("[yellow]Presence does not prove an inference workload used that device.[/yellow]\n")
    runtimes()


@app.command()
def models() -> None:
    """List model configurations; model binaries are never included."""
    table = Table("Configuration", "ID", "Format", "Quantization", "Path")
    for path in sorted((ROOT / "configs" / "models").glob("*.yaml")):
        model = ModelConfig.load(str(path))
        table.add_row(path.stem, model.id, model.format, model.quantization or "-", model.path or "-")
    console.print(table)


@app.command()
def explain(runtime: Annotated[str, typer.Argument(help="Runtime path identifier.")]) -> None:
    """Explain layering and ownership for one inference path."""
    adapter = create_adapter(runtime)
    description = adapter.describe()
    render_path(console, description)
    console.print()
    render_responsibilities(console, description)
    probe = adapter.probe()
    console.print("\n[bold]EVIDENCE / READINESS[/bold]")
    render_probe(console, runtime, probe)


@app.command("run")
def run_command(
    runtime: Annotated[str, typer.Option(help="Adapter identifier.")],
    workload: Annotated[str, typer.Option(help="Workload name or YAML path.")],
    model: Annotated[str | None, typer.Option(help="Model name or YAML path.")] = None,
    output: Annotated[Path, typer.Option(help="Raw JSONL output.")] = ROOT / "results" / "run.jsonl",
) -> None:
    """Run one measured experiment."""
    try:
        model_config = ModelConfig.load(model) if model else None
        adapter = create_adapter(runtime, model_config)
        render_path(console, adapter.describe())
        render_probe(console, runtime, adapter.probe())
        records = run_experiment(runtime, Workload.load(workload), model_config, 1, 0, output)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Cannot run experiment:[/bold red] {exc}")
        raise typer.Exit(2) from None
    record = records[0]
    console.print(f"\n[bold green]Completed[/bold green] in {record['timing']['end_to_end_seconds']:.3f}s")
    console.print(record["text"])
    console.print("\n[bold]RUNTIME EVIDENCE[/bold]")
    for evidence in record["evidence"]:
        console.print(f"- [{evidence['kind']}] {evidence['claim']}: {evidence['value']} ({evidence['source']})")
    console.print(f"Raw result: [link={output.resolve()}]{output}[/link]")


@app.command()
def benchmark(
    config: Annotated[str, typer.Option(help="Benchmark name or YAML path.")] = "default",
) -> None:
    """Run a configured benchmark and generate a Markdown report."""
    try:
        output, records = run_benchmark(config)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Benchmark stopped:[/bold red] {exc}")
        raise typer.Exit(2) from None
    report = write_markdown_report(records, output.with_suffix(".md"))
    console.print(f"[bold green]Benchmark complete.[/bold green] Raw: {output} Report: {report}")


@app.command()
def compare(
    runtimes_to_compare: Annotated[list[str], typer.Argument(help="Two or more adapter identifiers.")],
    input_path: Annotated[Path, typer.Option("--input", help="Raw JSONL results.")] = ROOT / "results" / "benchmark.jsonl",
) -> None:
    """Compare latest records and explain only observed/configured/runtime-reported differences."""
    if len(runtimes_to_compare) < 2:
        raise typer.BadParameter("Provide at least two runtimes.")
    if not input_path.exists():
        raise typer.BadParameter(f"Results file does not exist: {input_path}")
    records = read_records(input_path)
    selected = []
    for runtime in runtimes_to_compare:
        matches = [record for record in records if record["adapter"] == runtime]
        if not matches:
            raise typer.BadParameter(f"No result for {runtime} in {input_path}")
        selected.append(matches[-1])
    for left, right in zip(selected, selected[1:]):
        console.rule(f"{left['adapter']} vs {right['adapter']}")
        for item in explain_differences(left, right):
            console.print(f"[bold]{item['category']}[/bold]: {item['observation']}")
            console.print(f"  {item['possible_explanation']}")


if __name__ == "__main__":
    app()
