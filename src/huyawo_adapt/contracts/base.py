from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[
    str,
    StringConstraints(min_length=1, strip_whitespace=True),
]

Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class StrictFrozenModel(BaseModel):
    """Shared strict and immutable model behavior."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ArtifactDigest(StrictFrozenModel):
    """Identity of one immutable artifact by logical name and SHA-256 digest."""

    name: NonEmptyString
    sha256: Sha256Digest
    size_bytes: int | None = Field(default=None, ge=0)


class ContractModel(StrictFrozenModel):
    """Base model for canonical Huyawo Adapt contracts."""

    schema_version: Literal["1.0"] = "1.0"

    def canonical_data(self) -> dict[str, Any]:
        """Return the normalized JSON-compatible representation."""
        return self.model_dump(mode="json", round_trip=True)

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON."""
        return json.dumps(
            self.canonical_data(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def canonical_bytes(self) -> bytes:
        """Return UTF-8 encoded canonical JSON bytes."""
        return self.canonical_json().encode("utf-8")

    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the canonical bytes."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_canonical_json(cls, payload: str | bytes | bytearray) -> Self:
        """Validate and restore a contract from canonical JSON input."""
        return cls.model_validate_json(payload)
