from __future__ import annotations

from .models import DependencyGraph, DependencyGraphEdge, DeploymentBlock


def build_dependency_graph(blocks: tuple[DeploymentBlock, ...]) -> DependencyGraph:
    block_ids = {block.id for block in blocks}
    providers: dict[str, list[str]] = {}
    for block in blocks:
        for capability in block.provides:
            providers.setdefault(capability.capability, []).append(block.id)

    edges: dict[tuple[str, str, str], DependencyGraphEdge] = {}
    unmatched: set[str] = set()
    for block in blocks:
        for requirement in block.requires:
            provider = requirement.provider_block
            if provider is None:
                candidates = sorted(set(providers.get(requirement.capability, [])) - {block.id})
                if len(candidates) == 1:
                    provider = candidates[0]
            if provider is None:
                if requirement.mandatory:
                    unmatched.add(f"{block.id}:{requirement.capability}")
                continue
            if provider not in block_ids:
                unmatched.add(f"{block.id}:{requirement.capability}")
                continue
            edge = DependencyGraphEdge(
                provider_block=provider,
                consumer_block=block.id,
                capability=requirement.capability,
                mandatory=requirement.mandatory,
            )
            edges[(provider, block.id, requirement.capability)] = edge

    ordered_edges = sorted(
        edges.values(),
        key=lambda edge: (edge.provider_block, edge.consumer_block, edge.capability),
    )
    cycle = find_hard_cycle(tuple(sorted(block_ids)), tuple(ordered_edges))
    return DependencyGraph(
        nodes=sorted(block_ids),
        edges=ordered_edges,
        unmatched_capabilities=sorted(unmatched),
        cycle=cycle or [],
    )


def find_hard_cycle(
    nodes: tuple[str, ...], edges: tuple[DependencyGraphEdge, ...]
) -> list[str] | None:
    dependencies = {node: set() for node in nodes}
    for edge in edges:
        if edge.mandatory:
            dependencies[edge.consumer_block].add(edge.provider_block)
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(node: str) -> list[str] | None:
        if node in active_set:
            start = active.index(node)
            return [*active[start:], node]
        if node in visited:
            return None
        active.append(node)
        active_set.add(node)
        for dependency in sorted(dependencies[node]):
            cycle = visit(dependency)
            if cycle:
                return cycle
        active.pop()
        active_set.remove(node)
        visited.add(node)
        return None

    for node in nodes:
        cycle = visit(node)
        if cycle:
            return cycle
    return None
