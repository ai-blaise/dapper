"""Parsing for ``mixture.yaml``.

Lives in the repo beside ``dapper.yaml``: a mixture is a reviewable decision
about what to train on, not an artifact of the corpus. It *populates* the
bucket rather than living in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MIXTURE_FILENAMES = ("mixture.yaml", "mixture.yml")


class MixtureError(ValueError):
    """Raised when a mixture file is missing or malformed."""


@dataclass(frozen=True)
class DomainTarget:
    """A domain's share of a bin, optionally split across subdomains."""

    name: str
    share: float
    # Subdomain shares are *of the domain*, not of the bin -- so `code: 0.16`
    # with `repo_connected: 0.55` means 8.8% of the bin.
    subdomains: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BinTarget:
    """A bin's share of the corpus, and how that share splits by domain."""

    name: int
    share: float
    domains: tuple[DomainTarget, ...]


@dataclass(frozen=True)
class Mixture:
    bins: tuple[BinTarget, ...]


def find_mixture_path(start_dir: str | Path = ".") -> Path | None:
    root = Path(start_dir)
    for filename in DEFAULT_MIXTURE_FILENAMES:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def load_mixture(path: str | Path | None = None) -> Mixture:
    """Load and validate a mixture file."""
    mixture_path = Path(path) if path is not None else find_mixture_path()
    if mixture_path is None:
        raise MixtureError(
            "No mixture file found. Create mixture.yaml or pass --mixture PATH."
        )
    if not mixture_path.exists():
        raise MixtureError(f"Mixture file not found: {mixture_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a dependency
        raise MixtureError("Reading mixture.yaml requires PyYAML.") from exc

    data = yaml.safe_load(mixture_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise MixtureError(f"Mixture root must be a mapping: {mixture_path}")
    return parse_mixture(data)


def parse_mixture(data: dict[str, Any]) -> Mixture:
    raw_bins = data.get("bins")
    if not isinstance(raw_bins, dict) or not raw_bins:
        raise MixtureError("mixture.bins must be a non-empty mapping.")

    bins = []
    for raw_name, raw_bin in raw_bins.items():
        if not isinstance(raw_bin, dict):
            raise MixtureError(f"Bin {raw_name!r} must be a mapping.")
        try:
            name = int(raw_name)
        except (TypeError, ValueError) as exc:
            raise MixtureError(
                f"Bin keys must be integers matching dedup.len_bins, got "
                f"{raw_name!r}."
            ) from exc
        bins.append(
            BinTarget(
                name=name,
                share=_share(raw_bin.get("share"), f"bins.{name}.share"),
                domains=_domains(raw_bin.get("domains") or {}, name),
            )
        )

    _check_sums("bin shares", {b.name: b.share for b in bins})
    return Mixture(bins=tuple(sorted(bins, key=lambda b: b.name)))


def _domains(raw: Any, bin_name: int) -> tuple[DomainTarget, ...]:
    if not isinstance(raw, dict):
        raise MixtureError(f"bins.{bin_name}.domains must be a mapping.")
    targets = []
    for name, value in raw.items():
        if isinstance(value, dict):
            share = _share(value.get("share"), f"bins.{bin_name}.{name}.share")
            subs = value.get("subdomains") or {}
            if not isinstance(subs, dict):
                raise MixtureError(
                    f"bins.{bin_name}.{name}.subdomains must be a mapping."
                )
            parsed = {
                str(k): _share(v, f"bins.{bin_name}.{name}.{k}")
                for k, v in subs.items()
            }
            if parsed:
                _check_sums(f"bins.{bin_name}.{name} subdomains", parsed)
            targets.append(DomainTarget(str(name), share, parsed))
        else:
            targets.append(
                DomainTarget(
                    str(name), _share(value, f"bins.{bin_name}.{name}"), {}
                )
            )
    # An empty `domains` is a bin whose composition is not yet decided; that is
    # under-specified, not malformed, and `mixture check` reports it as such.
    if targets:
        _check_sums(f"bins.{bin_name} domains", {t.name: t.share for t in targets})
    return tuple(targets)


def _share(value: Any, where: str) -> float:
    try:
        share = float(value)
    except (TypeError, ValueError) as exc:
        raise MixtureError(f"{where} must be a number, got {value!r}.") from exc
    if not 0.0 <= share <= 1.0:
        raise MixtureError(f"{where} must be between 0 and 1, got {share}.")
    return share


def _check_sums(label: str, shares: dict[Any, float]) -> None:
    """Shares at one level must sum to 1.

    Caught at parse time rather than at check time: a mixture that does not sum
    to 1 is not "unsatisfiable", it is meaningless -- the percentages no longer
    describe a partition of anything.
    """
    total = sum(shares.values())
    if abs(total - 1.0) > 1e-6:
        raise MixtureError(
            f"{label} must sum to 1.0, got {total:.6f} "
            f"({', '.join(f'{k}={v}' for k, v in shares.items())})."
        )
