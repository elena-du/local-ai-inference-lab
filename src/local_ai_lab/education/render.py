from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from local_ai_lab.adapters.base import PathDescription, ProbeResult


def render_path(console: Console, description: PathDescription) -> None:
    console.rule(f"[bold cyan]PATH: {description.name}")
    console.print("\n  [dim]| v[/dim]\n".join(description.stack))
    for heading, values in (
        ("YOU CONTROL", description.you_control),
        ("RUNTIME CONTROLS", description.runtime_controls),
        ("WINDOWS CONTROLS", description.windows_controls),
    ):
        console.print(f"\n[bold]{heading}[/bold]")
        for value in values:
            console.print(f"- {value}")
    console.print(f"\n[bold]MODEL FORMAT:[/bold] {description.model_format}")
    console.print(f"[bold]ACTUAL EXECUTION DEVICE:[/bold] {description.actual_device}")
    console.print("[bold]EVIDENCE:[/bold] shown by readiness probes and runtime results; configured values are labeled separately")
    if description.notes:
        console.print("\n[bold yellow]CAVEATS[/bold yellow]")
        for note in description.notes:
            console.print(f"- {note}")


def render_probe(console: Console, name: str, probe: ProbeResult) -> None:
    marker = "[green]READY[/green]" if probe.available else "[yellow]UNAVAILABLE[/yellow]"
    console.print(f"{marker} [bold]{escape(name)}[/bold]: {escape(probe.status)}")
    for detail in probe.details:
        console.print(f"  {escape(detail)}")
    for evidence in probe.evidence:
        console.print(f"  [{evidence.kind}] {escape(evidence.claim)}: {escape(evidence.value)} ({escape(evidence.source)})")


def render_responsibilities(console: Console, description: PathDescription) -> None:
    table = Table("Responsibility", "Owner / answer")
    for key, value in description.responsibilities.items():
        table.add_row(key, value)
    console.print(table)
