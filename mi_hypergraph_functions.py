"""Weighted hypergraph mutual-information functions used by this project."""

import math

from collections import Counter
from collections.abc import Mapping
from itertools import combinations
from mpmath import loggamma


def logchoose(n, k):
    """Log binomial coefficient."""
    return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)


def logmultiset(n, k):
    """Log multiset coefficient."""
    return logchoose(n + k - 1, k)


def coarse_grain(G, partition):#coarse object is the hypergraph under the node partition so here it stays the same because B=N
    """Return the weighted hypergraph under the node partition.Changed according to definition of weighted/multiscale hypergraph."""
    Gc = Counter()
    for edge, weight in G.items():
        coarse_edge = tuple(sorted(partition[node] for node in edge))
        Gc[coarse_edge] += weight
    return Gc


def get_layers(G, partition):#duplicates/multiplicities are now not discarded-partition is requiered
    """Split a weighted hypergraph into Counter layers."""
    layers = {len(edge): Counter() for edge in G}
    for edge, weight in coarse_grain(G, partition).items():
        layers[len(edge)][edge] = weight
    return layers


def H(N, G, partition):
    """Weighted cross entropy from Eqs.  S7 and S9."""
    B = len(set(partition))
    layers = get_layers(G, partition)
    return sum(
        logmultiset(math.comb(B + order - 1, order), sum(layer.values()))
        for order, layer in layers.items()
    )


def get_projections(G, layers):
    """Project weighted hyperedges and sum their multiplicities."""
    projections = {order: Counter() for order in layers}
    for edge, weight in G.items():
        for order in layers:
            if order <= len(edge):
                for projected_edge in combinations(edge, order):
                    projections[order][projected_edge] += weight
    return projections


def get_sizes_proj(G, indices):
    """Calculate weighted projection sizes for the requested orders."""
    if len(G) == 0:
        return Counter()

    max_order = max(len(edge) for edge in G)
    # The NMI functions enforce B=N, so every disease remains a distinct node.
    # With additive multiedge weights, each copy of an edge contributes all
    # C(|edge|, order) subedges and the total size can be counted directly.
    return Counter({
        order: sum(
            weight * math.comb(len(edge), order)
            for edge, weight in G.items()
        )
        for order in indices
        if order <= max_order
    })


def get_overlap_size(layerk, layerl):
    """Calculate the S11/S12 weighted multiset overlap."""
    order = len(next(iter(layerl)))
    k = len(next(iter(layerk)))

    direct_cost = len(layerk) * math.comb(k, order)
    counting_cost = len(layerk) * len(layerl)

    if direct_cost <= counting_cost:
        projection = get_projections(layerk, [order])[order]
        return sum((projection & layerl).values())

    source_edges = [
        (frozenset(edge), weight)
        for edge, weight in layerk.items()
    ]
    overlap = 0
    for target_edge, target_weight in layerl.items():
        projected_weight = sum(
            source_weight
            for source_edge, source_weight in source_edges
            if source_edge.issuperset(target_edge)
        )
        overlap += min(projected_weight, target_weight)
    return overlap


def CE_matrices(N, G1, G2, partition):
    """Calculate weighted conditional-description costs from Eqs. S8 and S10."""
    layers1 = get_layers(G1, partition)
    layers2 = get_layers(G2, partition)
    B = len(set(partition))

    M2given1 = {}
    for k in layers1:
        below_k = [order for order in layers2 if order <= k]
        if len(below_k) == 0:
            continue

        sizes1to2 = get_sizes_proj(layers1[k], below_k)
        M2given1[k] = {}
        for order, projected_size in sizes1to2.items():
            overlap = get_overlap_size(layers1[k], layers2[order])
            layer_size = sum(layers2[order].values())
            M2given1[k][order] = (
                logchoose(projected_size, overlap)
                + logmultiset(
                    math.comb(B + order - 1, order), layer_size - overlap
                )
            )

    for order in layers2:
        if order not in layers1:
            M2given1[order] = {
                order: logmultiset(
                    math.comb(B + order - 1, order),
                    sum(layers2[order].values()),
                )
            }

    M1given2 = {}
    for k in layers2:
        below_k = [order for order in layers1 if order <= k]
        if len(below_k) == 0:
            continue

        sizes2to1 = get_sizes_proj(layers2[k], below_k)
        M1given2[k] = {}
        for order, projected_size in sizes2to1.items():
            overlap = get_overlap_size(layers2[k], layers1[order])
            layer_size = sum(layers1[order].values())
            M1given2[k][order] = (
                logchoose(projected_size, overlap)
                + logmultiset(
                    math.comb(B + order - 1, order), layer_size - overlap
                )
            )

    for order in layers1:
        if order not in layers2:
            M1given2[order] = {
                order: logmultiset(
                    math.comb(B + order - 1, order),
                    sum(layers1[order].values()),
                )
            }

    return M1given2, M2given1


def NMIalign(G1, G2, partition):
    """Compute weighted NMIalign from Eqs. S7-S8, normalized as in Eq. (9)."""

    if not isinstance(G1, Mapping) or not isinstance(G2, Mapping):
        raise TypeError("G1 and G2 must be weighted Counter or mapping inputs.")
    if partition is None or len(set(partition)) != len(partition):
        raise ValueError("Weighted input requires a partition with one group per node.")

    if len(G1) == 0:
        return 1.0 if len(G2) == 0 else 0.0
    if len(G2) == 0:
        return 0.0

    N = len(partition)
    B = len(set(partition))
    H1 = H(N, G1, partition)
    H2 = H(N, G2, partition)
    M1given2, M2given1 = CE_matrices(N, G1, G2, partition)
    layers1 = get_layers(G1, partition)
    layers2 = get_layers(G2, partition)

    CE1given2 = 0
    for order, layer in layers1.items():
        layer_entropy = logmultiset(
            math.comb(B + order - 1, order), sum(layer.values())
        )
        CE1given2 += min(layer_entropy, M1given2[order][order])

    CE2given1 = 0
    for order, layer in layers2.items():
        layer_entropy = logmultiset(
            math.comb(B + order - 1, order), sum(layer.values())
        )
        CE2given1 += min(layer_entropy, M2given1[order][order])

    nmi12 = (H1 - CE1given2) / (H1 + 1e-100)
    nmi21 = (H2 - CE2given1) / (H2 + 1e-100)
    return max(nmi12, nmi21)

# Natural logs are acceptable because the scale cancels after normalization
def NMIcross(G1, G2, partition):
    """Compute weighted NMIcross from Eqs. S9-S13, normalized as in Eq. (9)."""
    if not isinstance(G1, Mapping) or not isinstance(G2, Mapping):
        raise TypeError("G1 and G2 must be weighted Counter or mapping inputs.")
    if partition is None or len(set(partition)) != len(partition):
        raise ValueError("Weighted input requires a partition with one group per node.")

    if len(G1) == 0:
        return 1.0 if len(G2) == 0 else 0.0
    if len(G2) == 0:
        return 0.0

    N = len(partition)
    B = len(set(partition))
    H1 = H(N, G1, partition)
    H2 = H(N, G2, partition)
    M1given2, M2given1 = CE_matrices(N, G1, G2, partition)
    layers1 = get_layers(G1, partition)
    layers2 = get_layers(G2, partition)

    CE1given2 = 0
    for order, layer in layers1.items():
        candidates = [
            M1given2[k][order]
            for k in layers2
            if k >= order and order in M1given2[k]
        ]
        layer_entropy = logmultiset(
            math.comb(B + order - 1, order), sum(layer.values())
        )
        CE1given2 += min(layer_entropy, min(candidates)) if candidates else layer_entropy

    CE2given1 = 0
    for order, layer in layers2.items():
        candidates = [
            M2given1[k][order]
            for k in layers1
            if k >= order and order in M2given1[k]
        ]
        layer_entropy = logmultiset(
            math.comb(B + order - 1, order), sum(layer.values())
        )
        CE2given1 += min(layer_entropy, min(candidates)) if candidates else layer_entropy

    nmi12 = (H1 - CE1given2) / (H1 + 1e-100)
    nmi21 = (H2 - CE2given1) / (H2 + 1e-100)
    return max(nmi12, nmi21)
