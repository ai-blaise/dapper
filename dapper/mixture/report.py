"""Display formatting for ``dapper mixture check``."""

from __future__ import annotations

from dapper.mixture.check import MixtureCheck


def format_check(result: MixtureCheck) -> str:
    lines = [
        "Dapper mixture check",
        "",
        f"Budget: {result.budget_tokens:,} tokens",
        f"Corpus: {result.total_available:,} tokens available",
        "",
    ]
    for bin_check in result.bins:
        flag = "ok" if bin_check.satisfiable else "SHORT"
        lines.append(
            f"bin {bin_check.bin_name} — target {bin_check.share:.1%}  "
            f"need {bin_check.needed:,}  have {bin_check.available:,}  [{flag}]"
        )
        for cell in bin_check.cells:
            label = (
                f"{cell.domain}/{cell.subdomain}" if cell.subdomain else cell.domain
            )
            status = (
                "ok"
                if cell.satisfiable
                else f"SHORT {cell.shortfall:,}"
            )
            lines.append(
                f"    {label:<34} {cell.share:>7.2%}  "
                f"need {cell.needed:>15,}  have {cell.available:>15,}  {status}"
            )
        if not bin_check.cells:
            lines.append("    (no domains declared)")
        lines.append("")

    if result.satisfiable:
        lines.append("Mixture is satisfiable against the current corpus.")
    else:
        lines.append(
            "Mixture is NOT satisfiable. Archive more sources for the short "
            "cells, or lower their shares."
        )
    return "\n".join(lines)
