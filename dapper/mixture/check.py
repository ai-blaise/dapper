"""Resolve a mixture against measured capacities.

The manifest says what exists; the mixture says what we want. This turns the
pair into the only question that matters before a build: **is the plan
satisfiable, and where does it fall short?**

Nothing here writes. A check is cheap because the manifest is kilobytes -- it
never touches the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dapper.mixture.config import Mixture


@dataclass(frozen=True)
class CellCheck:
    """One (bin, domain, subdomain) target against what exists."""

    bin_name: int
    domain: str
    subdomain: str
    share: float
    needed: int
    available: int

    @property
    def satisfiable(self) -> bool:
        return self.available >= self.needed

    @property
    def shortfall(self) -> int:
        return max(0, self.needed - self.available)


@dataclass(frozen=True)
class BinCheck:
    bin_name: int
    share: float
    needed: int
    available: int
    cells: tuple[CellCheck, ...]

    @property
    def satisfiable(self) -> bool:
        return self.available >= self.needed and all(c.satisfiable for c in self.cells)


@dataclass(frozen=True)
class MixtureCheck:
    budget_tokens: int
    total_available: int
    bins: tuple[BinCheck, ...]

    @property
    def satisfiable(self) -> bool:
        return all(b.satisfiable for b in self.bins)


def check_mixture(
    mixture: Mixture, manifest: dict[str, Any], *, budget_tokens: int | None = None
) -> MixtureCheck:
    """Resolve ``mixture`` against ``manifest``.

    ``budget_tokens`` is the size of the training run being planned. Defaulting
    it to the corpus total answers "can I use everything I have in these
    proportions?", which is the useful question when no budget is fixed yet.
    """
    bins_data = manifest.get("bins") or {}
    total_available = int(manifest.get("total_tokens") or 0)
    budget = int(budget_tokens or total_available)

    results = []
    for bin_target in mixture.bins:
        available_bin = _bin_tokens(bins_data, bin_target.name)
        needed_bin = round(budget * bin_target.share)
        cells = []
        for domain in bin_target.domains:
            if domain.subdomains:
                for sub, sub_share in domain.subdomains.items():
                    cells.append(
                        _cell(
                            bins_data,
                            bin_target.name,
                            domain.name,
                            sub,
                            domain.share * sub_share,
                            needed_bin,
                        )
                    )
            else:
                cells.append(
                    _cell(
                        bins_data,
                        bin_target.name,
                        domain.name,
                        None,
                        domain.share,
                        needed_bin,
                    )
                )
        results.append(
            BinCheck(
                bin_name=bin_target.name,
                share=bin_target.share,
                needed=needed_bin,
                available=available_bin,
                cells=tuple(cells),
            )
        )
    return MixtureCheck(
        budget_tokens=budget,
        total_available=total_available,
        bins=tuple(results),
    )


def _cell(
    bins_data: dict[str, Any],
    bin_name: int,
    domain: str,
    subdomain: str | None,
    share_of_bin: float,
    needed_bin: int,
) -> CellCheck:
    return CellCheck(
        bin_name=bin_name,
        domain=domain,
        subdomain=subdomain or "",
        share=share_of_bin,
        needed=round(needed_bin * share_of_bin),
        available=_cell_tokens(bins_data, bin_name, domain, subdomain),
    )


def _bin_tokens(bins_data: dict[str, Any], bin_name: int) -> int:
    node = bins_data.get(str(bin_name)) or {}
    return sum(
        int(sub.get("n_tokens", 0))
        for domain in node.values()
        for sub in domain.values()
    )


def _cell_tokens(
    bins_data: dict[str, Any], bin_name: int, domain: str, subdomain: str | None
) -> int:
    node = (bins_data.get(str(bin_name)) or {}).get(domain) or {}
    if subdomain is None:
        # No subdomain requested: the domain's whole capacity counts, however
        # it happens to be subdivided.
        return sum(int(sub.get("n_tokens", 0)) for sub in node.values())
    return int((node.get(subdomain) or {}).get("n_tokens", 0))
