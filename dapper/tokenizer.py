"""Canonical tokenizer configuration and frozen tokenizer identity.

The tokenizer is a corpus-wide contract.  Deduplication may count tokens and
the tokenization commands materialize them, but neither stage owns the choice
of vocabulary or special-token semantics.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_TOKENIZER = "zai-org/GLM-5.2"


class TokenizerConfigError(ValueError):
    """Raised when the project tokenizer contract is ambiguous or invalid."""


@dataclass(frozen=True)
class BoundaryConfig:
    token: str = "eos"
    after_each_document: bool = True
    include_in_loss: bool = True


@dataclass(frozen=True)
class PaddingConfig:
    token: str = "pad"
    label_value: int = -100
    reuse_eos: bool = False


@dataclass(frozen=True)
class TokenizerConfig:
    name: str = DEFAULT_TOKENIZER
    add_special_tokens: bool = False
    boundary: BoundaryConfig = BoundaryConfig()
    padding: PaddingConfig = PaddingConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizerIdentity:
    """Tokenizer settings and special IDs frozen into a run manifest."""

    name: str
    content_hash: str
    add_special_tokens: bool
    eos_token: str
    eos_id: int
    pad_token: str
    pad_id: int
    pad_reuses_eos: bool
    boundary_after_each_document: bool
    boundary_include_in_loss: bool
    padding_label_value: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_tokenizer_config(config: dict[str, Any]) -> TokenizerConfig:
    """Parse the project-level tokenizer block with legacy compatibility.

    ``dedup.tokenizer`` remains a read-only migration fallback.  A project that
    defines both names must keep them identical so stages cannot silently use
    different vocabularies.
    """

    raw = config.get("tokenizer")
    if raw is not None and not isinstance(raw, dict):
        raise TokenizerConfigError("tokenizer must be a mapping with a name field.")
    raw = raw or {}
    dedup = config.get("dedup")
    dedup = dedup if isinstance(dedup, dict) else {}
    general_name = raw.get("name")
    legacy_name = dedup.get("tokenizer")

    if (
        general_name is not None
        and legacy_name is not None
        and str(general_name) != str(legacy_name)
    ):
        raise TokenizerConfigError(
            "tokenizer.name and deprecated dedup.tokenizer disagree: "
            f"{general_name!r} != {legacy_name!r}. Remove dedup.tokenizer."
        )
    if general_name is None and legacy_name is not None:
        warnings.warn(
            "dedup.tokenizer is deprecated; move it to tokenizer.name.",
            DeprecationWarning,
            stacklevel=2,
        )
        general_name = legacy_name

    boundary_raw = raw.get("boundary")
    boundary_raw = boundary_raw if isinstance(boundary_raw, dict) else {}
    padding_raw = raw.get("padding")
    padding_raw = padding_raw if isinstance(padding_raw, dict) else {}
    add_special_tokens = bool(raw.get("add_special_tokens", False))
    if add_special_tokens:
        raise TokenizerConfigError(
            "tokenizer.add_special_tokens must be false: documents are encoded "
            "independently and Dapper inserts the frozen EOS boundary itself."
        )

    boundary = BoundaryConfig(
        token=str(boundary_raw.get("token", "eos")),
        after_each_document=bool(boundary_raw.get("after_each_document", True)),
        include_in_loss=bool(boundary_raw.get("include_in_loss", True)),
    )
    if not boundary.after_each_document:
        raise TokenizerConfigError(
            "tokenizer.boundary.after_each_document must be true for packed runs."
        )
    padding = PaddingConfig(
        token=str(padding_raw.get("token", "pad")),
        label_value=int(padding_raw.get("label_value", -100)),
        reuse_eos=bool(
            padding_raw.get("reuse_eos", padding_raw.get("reuse_eos_as_pad", False))
        ),
    )
    return TokenizerConfig(
        name=str(general_name or DEFAULT_TOKENIZER),
        add_special_tokens=add_special_tokens,
        boundary=boundary,
        padding=padding,
    )


def resolve_tokenizer(config: TokenizerConfig, tokenizer: Any | None = None) -> tuple[Any, TokenizerIdentity]:
    """Load and validate a tokenizer, returning it with its frozen identity."""

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency validation
            raise TokenizerConfigError(
                "transformers is required to resolve the configured tokenizer."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=True)

    eos_token, eos_id = _resolve_special(tokenizer, config.boundary.token, "EOS")
    try:
        pad_token, pad_id = _resolve_special(tokenizer, config.padding.token, "PAD")
    except TokenizerConfigError:
        if not config.padding.reuse_eos:
            raise TokenizerConfigError(
                "The tokenizer has no usable PAD token. Set "
                "tokenizer.padding.reuse_eos: true to explicitly reuse EOS as "
                "the physical PAD ID while retaining padding masks and labels."
            ) from None
        pad_token, pad_id = eos_token, eos_id

    pad_reuses_eos = pad_id == eos_id
    if pad_reuses_eos and not config.padding.reuse_eos:
        raise TokenizerConfigError(
            "EOS and PAD resolve to the same ID. Explicitly set "
            "tokenizer.padding.reuse_eos: true to acknowledge that policy."
        )

    identity = TokenizerIdentity(
        name=config.name,
        content_hash=_tokenizer_content_hash(tokenizer, config.name),
        add_special_tokens=False,
        eos_token=eos_token,
        eos_id=eos_id,
        pad_token=pad_token,
        pad_id=pad_id,
        pad_reuses_eos=pad_reuses_eos,
        boundary_after_each_document=config.boundary.after_each_document,
        boundary_include_in_loss=config.boundary.include_in_loss,
        padding_label_value=config.padding.label_value,
    )
    return tokenizer, identity


def _resolve_special(tokenizer: Any, configured: str, label: str) -> tuple[str, int]:
    """Resolve ``eos``/``pad`` aliases or an explicit token to one vocab ID."""

    alias = configured.lower()
    if alias in {"eos", "pad", "bos", "unk"}:
        token = getattr(tokenizer, f"{alias}_token", None)
        token_id = getattr(tokenizer, f"{alias}_token_id", None)
    else:
        token = configured
        token_id = tokenizer.convert_tokens_to_ids(configured)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or (unk_id is not None and token_id == unk_id and configured != getattr(tokenizer, "unk_token", None)):
            token_id = None

    if token is None or token_id is None:
        raise TokenizerConfigError(
            f"Configured {label} token {configured!r} does not resolve in the tokenizer vocabulary."
        )
    try:
        ids = tokenizer.encode(str(token), add_special_tokens=False)
    except TypeError:
        ids = tokenizer(str(token), add_special_tokens=False)["input_ids"]
    if hasattr(ids, "ids"):
        ids = ids.ids
    if len(ids) != 1 or int(ids[0]) != int(token_id):
        raise TokenizerConfigError(
            f"Configured {label} token {token!r} must encode to exactly its one vocabulary ID; got {list(ids)!r}."
        )
    return str(token), int(token_id)


def _tokenizer_content_hash(tokenizer: Any, fallback_name: str) -> str:
    """Hash resolved tokenizer content instead of only its mutable repository name."""

    payload: str | bytes | None = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        payload = backend.to_str()
    elif hasattr(tokenizer, "to_str"):
        payload = tokenizer.to_str()
    elif hasattr(tokenizer, "get_vocab"):
        payload = json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":"))
    if payload is None:
        payload = fallback_name
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tokenizer_name(config: dict[str, Any]) -> str:
    """Small compatibility helper for consumers that need only the repository."""

    return parse_tokenizer_config(config).name
