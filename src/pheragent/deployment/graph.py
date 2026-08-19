from __future__ import annotations

from collections.abc import Iterable, Mapping, Set


def add_dependency_if_acyclic(
    dependencies: dict[str, set[str]],
    *,
    dependent: str,
    prerequisite: str,
) -> bool:
    """Add one prerequisite unless it is invalid or would introduce a cycle."""
    if dependent == prerequisite or dependent not in dependencies:
        return False
    pending = [prerequisite]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == dependent:
            return False
        if current in visited:
            continue
        visited.add(current)
        pending.extend(dependencies.get(current, ()))
    dependencies[dependent].add(prerequisite)
    return True


def topological_order(
    node_ids: Iterable[str],
    dependencies: Mapping[str, Set[str]],
    *,
    cycle_label: str,
) -> tuple[str, ...]:
    """Return a stable dependency order or reject a cyclic graph."""
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    emitted: set[str] = set()
    ordered: list[str] = []
    while len(ordered) < len(positions):
        ready = sorted(
            (
                node_id
                for node_id in positions
                if node_id not in emitted and dependencies.get(node_id, set()) <= emitted
            ),
            key=positions.__getitem__,
        )
        if not ready:
            raise ValueError(f"{cycle_label} contains a cycle")
        ordered.extend(ready)
        emitted.update(ready)
    return tuple(ordered)


def topological_levels(
    node_ids: Iterable[str],
    dependencies: Mapping[str, Set[str]],
    *,
    cycle_label: str,
) -> list[list[str]]:
    """Return stable parallel levels while preserving dependency constraints."""
    remaining = tuple(node_ids)
    emitted: set[str] = set()
    levels: list[list[str]] = []
    while len(emitted) < len(remaining):
        ready = sorted(
            node_id
            for node_id in remaining
            if node_id not in emitted and dependencies.get(node_id, set()) <= emitted
        )
        if not ready:
            raise ValueError(f"{cycle_label} contains a cycle")
        levels.append(ready)
        emitted.update(ready)
    return levels
