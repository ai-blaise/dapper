"""Display formatting for ``dapper mixture check``."""

from __future__ import annotations

from rich.markup import escape as _e
from rich.table import Table

from dapper.mixture.check import MixtureCheck
from utils.display import (
    BAD,
    BORDER,
    GOOD,
    HEADING,
    MUTED,
    console,
    header_panel,
    kv_table,
    panel,
)


def format_check(result: MixtureCheck) -> str:
    with console.capture() as capture:
        console.print(header_panel("Dapper mixture check"))
        console.print(
            kv_table(
                [
                    ("Budget", f"{result.budget_tokens:,} tokens"),
                    ("Corpus", f"{result.total_available:,} tokens available"),
                ],
                value_style=GOOD,
            )
        )
        console.print()

        for bin_check in result.bins:
            table = Table(
                title=f"bin [{HEADING}]{bin_check.bin_name}[/{HEADING}]",
                show_header=True,
                header_style=HEADING,
                border_style=BORDER,
            )
            table.add_column("Domain", style=MUTED, min_width=36)
            table.add_column("Share", justify="right")
            table.add_column("Need", justify="right")
            table.add_column("Have", justify="right")
            table.add_column("Status", justify="left")

            flag_style = GOOD if bin_check.satisfiable else BAD
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
                    status = f"[{GOOD}]ok[/{GOOD}]"
                else:
                    status = f"[{BAD}]SHORT {cell.shortfall:,}[/{BAD}]"
                table.add_row(
                    f"    {_e(label)}",
                    f"{cell.share:>7.2%}",
                    f"{cell.needed:>15,}",
                    f"{cell.available:>15,}",
                    status,
                )

            if not bin_check.cells:
                table.add_row(
                    f"[{MUTED}](no domains declared)[/{MUTED}]", "", "", "", ""
                )

            console.print(table)
            console.print()

        if result.satisfiable:
            verdict = panel(
                f"[bold {GOOD}]Mixture is satisfiable against the current corpus.[/bold {GOOD}]",
                border_style=GOOD,
            )
        else:
            verdict = panel(
                f"[bold {BAD}]Mixture is NOT satisfiable. Archive more sources for "
                f"the short cells, or lower their shares.[/bold {BAD}]",
                border_style=BAD,
            )
        console.print(verdict)

    return capture.get()
