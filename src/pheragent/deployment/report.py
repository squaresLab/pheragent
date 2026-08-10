from __future__ import annotations

from .models import (
    ArtifactValidationResult,
    DependencyGraph,
    DeploymentArtifact,
    RepositoryInventory,
)


def render_inspection_report(
    *,
    artifact: DeploymentArtifact,
    inventory: RepositoryInventory,
    graph: DependencyGraph,
    validation: ArtifactValidationResult,
    extraction_warnings: list[str],
    unassigned_commands: list[str],
) -> str:
    inspected = [entry for entry in inventory.entries if entry.inspected]
    skipped = [entry for entry in inventory.entries if not entry.selected]
    lines = [
        f"# Deployment Inspection Report: {artifact.metadata.system_name}",
        "",
        "## Source summary",
        "",
    ]
    for source in artifact.sources:
        revision = source.revision or source.content_hash
        lines.append(f"- `{source.id}` ({source.kind}): `{revision}`")
    lines.extend(
        [
            "",
            "## Repository inspection",
            "",
            f"- Relevant files inspected: {len(inspected)}",
            f"- Files skipped: {len(skipped)}",
            "- Detected deployment technologies: "
            + (", ".join(inventory.detected_technologies) or "none"),
            "",
            "## Proposed block graph",
            "",
        ]
    )
    if graph.edges:
        for edge in graph.edges:
            lines.append(
                f"- `{edge.provider_block}` → `{edge.consumer_block}` via `{edge.capability}`"
            )
    else:
        lines.append("No inter-block dependency edges were resolved.")
    if graph.cycle:
        lines.append(f"- Hard cycle: `{' -> '.join(graph.cycle)}`")

    lines.extend(["", "## Blocks and components", ""])
    for block in artifact.blocks:
        lines.extend(
            [
                f"### {block.name} (`{block.id}`)",
                "",
                block.purpose,
                "",
                "Components: "
                + (", ".join(f"`{item.name}`" for item in block.components) or "none"),
                "",
                "Requires: "
                + (", ".join(f"`{item.capability}`" for item in block.requires) or "none"),
                "",
                "Provides: "
                + (", ".join(f"`{item.capability}`" for item in block.provides) or "none"),
                "",
                "Explicit validations: "
                + (", ".join(item.description for item in block.validations) or "none"),
                "",
            ]
        )

    low_confidence = _low_confidence_claims(artifact)
    human_gates = [block for block in artifact.blocks if block.type.value == "human_gate"]
    lines.extend(
        [
            "## Unmatched capabilities",
            "",
            *([f"- `{item}`" for item in graph.unmatched_capabilities] or ["None."]),
            "",
            "## Low-confidence claims",
            "",
            *([f"- {item}" for item in low_confidence] or ["None."]),
            "",
            "## Human gates",
            "",
            *([f"- `{block.id}`: {block.purpose}" for block in human_gates] or ["None."]),
            "",
            "## Unresolved questions",
            "",
            *(
                [f"- **{item.question}** {item.reason}" for item in artifact.unresolved_questions]
                or ["None."]
            ),
            "",
            "## Unassigned discovered commands",
            "",
            *([f"- `{item}`" for item in unassigned_commands] or ["None."]),
            "",
            "## Extraction warnings",
            "",
            *([f"- {item}" for item in extraction_warnings] or ["None."]),
            "",
            "## Artifact validation findings",
            "",
            *(
                [
                    f"- **{item.severity} `{item.code}`**"
                    f"{f' (`{item.location}`)' if item.location else ''}: {item.message}"
                    for item in validation.findings
                ]
                or ["None. The artifact passed all current checks."]
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _low_confidence_claims(artifact: DeploymentArtifact) -> list[str]:
    claims: list[str] = []
    for block in artifact.blocks:
        if block.provenance.confidence.value == "low":
            claims.append(f"Block `{block.id}`")
        for component in block.components:
            if component.provenance.confidence.value == "low":
                claims.append(f"Component `{component.id}` in `{block.id}`")
    return claims
