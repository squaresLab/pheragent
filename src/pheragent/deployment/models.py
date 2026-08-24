from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import InventoryCategory, SourceKind

IDENTIFIER_PATTERN = r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$"


class ContractModel(BaseModel):
    """Base for persisted deployment contracts that reject unknown fields."""

    model_config = ConfigDict(extra="forbid")


class SourceSpec(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: SourceKind
    location: str = Field(min_length=1)
    purpose: Literal["repository", "documentation"] = "repository"
    revision: str | None = None
    root_path: str = "."
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)


class SourcesConfig(ContractModel):
    system: str = Field(min_length=1)
    sources: list[SourceSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_source_ids(self) -> Self:
        _require_unique((source.id for source in self.sources), "source id")
        return self


class SourceManifestEntry(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: SourceKind
    location: str = Field(min_length=1)
    purpose: Literal["repository", "documentation"] = "repository"
    requested_revision: str | None = None
    resolved_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40,64}$")
    root_path: str = "."
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def require_revision_for_git(self) -> Self:
        if self.kind == SourceKind.GIT and self.resolved_revision is None:
            raise ValueError("Git manifest entries require a resolved revision")
        if self.kind != SourceKind.GIT and self.resolved_revision is not None:
            raise ValueError("only Git manifest entries may have a resolved revision")
        return self


class SourceManifest(ContractModel):
    manifest_version: str = Field(pattern=r"^0\.1$")
    system: str = Field(min_length=1)
    sources: list[SourceManifestEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_source_ids(self) -> Self:
        _require_unique((source.id for source in self.sources), "source id")
        return self


class InventoryEntry(ContractModel):
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1)
    category: InventoryCategory
    size_bytes: int = Field(ge=0)
    selected: bool = False
    inspected: bool = False
    skip_reason: str | None = None
    relevance_score: float = Field(default=0.0, ge=0.0)
    parser: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection_state(self) -> Self:
        if not self.selected and not self.skip_reason:
            raise ValueError("unselected inventory entries require skip_reason")
        if self.selected and self.skip_reason:
            raise ValueError("selected inventory entries must not have skip_reason")
        if self.inspected and not self.selected:
            raise ValueError("inspected inventory entries must be selected")
        if self.parser and not self.inspected:
            raise ValueError("inventory parser requires inspected=true")
        return self


class RepositoryInventory(ContractModel):
    inventory_version: str = Field(default="0.1", pattern=r"^0\.1$")
    detected_technologies: list[str] = Field(default_factory=list)
    entries: list[InventoryEntry] = Field(default_factory=list)


def _require_unique(values: Iterable[str], label: str) -> None:
    materialized = list(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"duplicate {label}")
