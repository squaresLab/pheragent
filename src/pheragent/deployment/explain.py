from __future__ import annotations

from .models import DeploymentArtifact, DeploymentBlock


def find_block(artifact: DeploymentArtifact, block_id: str) -> DeploymentBlock:
    block = next((candidate for candidate in artifact.blocks if candidate.id == block_id), None)
    if block is None:
        raise ValueError(f"unknown deployment block: {block_id}")
    return block


def render_block_explanation(block: DeploymentBlock) -> str:
    lines = [
        f"{block.name} ({block.id})",
        f"type: {block.type}",
        f"purpose: {block.purpose}",
        f"grouping: {block.grouping_rationale}",
        f"provenance: {block.provenance.origin}/{block.provenance.confidence}",
    ]
    _append_named_items(lines, "components", ((item.id, item.name) for item in block.components))
    _append_values(lines, "requires", (item.capability for item in block.requires))
    _append_values(lines, "provides", (item.capability for item in block.provides))
    _append_named_items(lines, "operations", ((item.id, item.phase) for item in block.operations))
    _append_named_items(
        lines,
        "validations",
        ((item.id, item.description) for item in block.validations),
    )
    return "\n".join(lines) + "\n"


def _append_values(lines: list[str], heading: str, values: object) -> None:
    items = list(values)
    if not items:
        return
    lines.append(f"{heading}:")
    lines.extend(f"- {item}" for item in items)


def _append_named_items(lines: list[str], heading: str, values: object) -> None:
    items = list(values)
    if not items:
        return
    lines.append(f"{heading}:")
    lines.extend(f"- {name}: {description}" for name, description in items)
