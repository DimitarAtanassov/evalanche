"""Pairwise swap resolution and Bradley-Terry connectivity."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from evalharness.judge.models import (
    BradleyTerryAccepted,
    BradleyTerryRefused,
    PairwiseItem,
    PairwiseOrdering,
)


def resolve_original_preference(
    *,
    swap_position: int,
    preference: str,
) -> str:
    """Map a position-relative preference onto the original A/B identity."""
    if preference == "tie":
        return "tie"
    if swap_position == 0:
        return preference
    # swap_position 1 shows original B first (position A) and original A second.
    if preference == "A":
        return "B"
    return "A"


def build_pairwise_item(
    *,
    case_id: str,
    a_generation_id: str,
    b_generation_id: str,
    a_model_label: str,
    b_model_label: str,
    orderings: list[PairwiseOrdering],
) -> PairwiseItem:
    """Resolve swap consistency and the final preference for one pair."""
    if len(orderings) != 2:
        raise ValueError("SWAP_INCOMPLETE")
    positions = {ordering.swap_position for ordering in orderings}
    if positions != {0, 1}:
        raise ValueError("SWAP_INCOMPLETE")
    resolved = [
        resolve_original_preference(
            swap_position=ordering.swap_position,
            preference=ordering.preference,
        )
        for ordering in sorted(orderings, key=lambda item: item.swap_position)
    ]
    consistent = resolved[0] == resolved[1] and resolved[0] != "tie"
    if not consistent:
        final: str = "tie"
    else:
        final = resolved[0]
    return PairwiseItem(
        case_id=case_id,
        a_generation_id=a_generation_id,
        b_generation_id=b_generation_id,
        a_model_label=a_model_label,
        b_model_label=b_model_label,
        orderings=sorted(orderings, key=lambda item: item.swap_position),
        consistent=consistent,
        final_preference=final,  # type: ignore[arg-type]
    )


def _connected_components(nodes: set[str], edges: Iterable[tuple[str, str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(nodes)
    components: list[set[str]] = []
    while remaining:
        start = next(iter(remaining))
        queue: deque[str] = deque([start])
        component: set[str] = set()
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            queue.extend(adjacency[node] - component)
        components.append(component)
        remaining -= component
    return components


def bradley_terry_summary(
    items: list[PairwiseItem],
) -> BradleyTerryAccepted | BradleyTerryRefused:
    """Accept Bradley-Terry only under the contract connectivity rule."""
    nodes = {item.a_model_label for item in items} | {item.b_model_label for item in items}
    edges: list[tuple[str, str]] = []
    wins: dict[str, float] = defaultdict(float)
    matches: dict[str, float] = defaultdict(float)
    for item in items:
        if item.final_preference == "tie":
            continue
        winner = item.a_model_label if item.final_preference == "A" else item.b_model_label
        loser = item.b_model_label if item.final_preference == "A" else item.a_model_label
        edges.append((item.a_model_label, item.b_model_label))
        wins[winner] += 1.0
        matches[winner] += 1.0
        matches[loser] += 1.0

    unique_edges = {frozenset(edge) for edge in edges}
    degrees = {node: 0 for node in nodes}
    for edge in unique_edges:
        left, right = tuple(edge)
        degrees[left] += 1
        degrees[right] += 1
    components = _connected_components(nodes, ((a, b) for a, b in edges))
    component_sizes = sorted((len(component) for component in components), reverse=True)
    isolated = sorted(node for node, degree in degrees.items() if degree < 1)
    connected = (
        len(nodes) >= 2 and len(components) == 1 and all(degree >= 1 for degree in degrees.values())
    )
    if not connected:
        return BradleyTerryRefused(
            n_models=len(nodes),
            n_edges=len(unique_edges),
            component_sizes=component_sizes,
            isolated_models=isolated,
        )

    # Simple win-rate strengths; connectivity is the contract gate, not MLE.
    scores = {
        node: (wins[node] / matches[node]) if matches[node] else 0.0 for node in sorted(nodes)
    }
    return BradleyTerryAccepted(
        n_models=len(nodes),
        n_edges=len(unique_edges),
        scores=scores,
        component_sizes=component_sizes,
        isolated_models=isolated,
    )


def swap_consistency_rate(items: list[PairwiseItem]) -> float | None:
    if not items:
        return None
    return sum(1 for item in items if item.consistent) / len(items)


def position_bias_rate(items: list[PairwiseItem]) -> float | None:
    """Share of orderings that preferred the first-shown candidate."""
    preferences = [
        ordering.preference
        for item in items
        for ordering in item.orderings
        if ordering.preference != "tie"
    ]
    if not preferences:
        return None
    return sum(1 for preference in preferences if preference == "A") / len(preferences)
