"""Display formatting for ``dapper mixture check``."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape as _e
from rich.panel import Panel
from rich.table import Table

from dapper.mixture.check import MixtureCheck

console = Console(force_terminal=True, highlight=False)


def format_check(result: MixtureCheck) -> str:
    with console.capture() as capture:
        header = Panel.fit(
            "\n".join([
                "[bold cyan]Dapper mixture check[/bold cyan]",
                "",
                f"Budget: [bold]{result.budget_tokens:,}[/bold] tokens",
                f"Corpus: [bold]{result.total_available:,}[/bold] tokens available",
            ]),
            border_style="cyan",
        )
        console.print(header)

        for bin_check in result.bins:
            table = Table(
                title=f"bin [bold]{bin_check.bin_name}[/bold]",
                show_header=True,
                header_style="bold",
                border_style="dim blue",
            )
            table.add_column("Domain", style="dim", min_width=36)
            table.add_column("Share", justify="right")
            table.add_column("Need", justify="right")
            table.add_column("Have", justify="right")
            table.add_column("Status", justify="left")

            flag_style = "green" if bin_check.satisfiable else "red"
            flag_text = "ok" if bin_check.satisfiable else "SHORT"
            table.add_row(
                f"[bold]bin {_e(str(bin_check.bin_name))}[/bold]",
                f"[bold]{bin_check.share:.1%}[/bold]",
                f"[bold]{bin_check.needed:,}[/bold]",
                f"[bold]{bin_check.available:,}[/bold]",
                f"[bold {flag_style}]{flag_text}[/bold {flag_style}]",
            )

            for cell in bin_check.cells:
                label = (
                    f"{cell.domain}/{cell.subdomain}" if cell.subdomain else cell.domain
                )
                if cell.satisfiable:
                    status = "[green]ok[/green]"
                else:
                    status = f"[red]SHORT {cell.shortfall:,}[/red]"
                table.add_row(
                    f"    {_e(label)}",
                    f"{cell.share:>7.2%}",
                    f"{cell.needed:>15,}",
                    f"{cell.available:>15,}",
                    status,
                )

            if not bin_check.cells:
                table.add_row(
                    "[dim](no domains declared)[/dim]", "", "", "", ""
                )

            console.print(table)
            console.print()

        if result.satisfiable:
            verdict = Panel.fit(
                "[bold green]Mixture is satisfiable against the current corpus.[/bold green]",
                border_style="green",
            )
        else:
            verdict = Panel.fit(
                "[bold red]Mixture is NOT satisfiable. Archive more sources for the short cells, or lower their shares.[/bold red]",
                border_style="red",
            )
        console.print(verdict)

    return capture.get()
