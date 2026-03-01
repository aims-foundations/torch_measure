# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Network centrality metrics for network psychometric models.

All functions accept a symmetric adjacency (edge-weight) matrix and return
a score per node (item). These are the standard centrality measures used
in the network psychometrics literature (Opsahl et al., 2010;
Epskamp & Fried, 2018).

Functions
---------
strength_centrality
    Sum of absolute edge weights — most common measure in psychometrics.
expected_influence
    Signed sum of edge weights — sensitive to edge polarity.
closeness_centrality
    Reciprocal of mean shortest-path distance in the weighted graph.
betweenness_centrality
    Fraction of shortest paths passing through each node.

References
----------
.. [1] Opsahl, T., Agneessens, F., & Skvoretz, J. (2010). Node centrality in
       weighted networks: Generalizing degree and shortest paths. *Social
       Networks*, 32(3), 245–251.
.. [2] Epskamp, S., & Fried, E. I. (2018). A tutorial on regularized partial
       correlation networks. *Psychological Methods*, 23(4), 617–634.
"""

from __future__ import annotations

import torch


def strength_centrality(adjacency: torch.Tensor) -> torch.Tensor:
    """Node strength: sum of absolute edge weights.

    The most widely used centrality measure in network psychometrics.
    A high-strength node has strong connections (in absolute value) with
    many other nodes.

    Parameters
    ----------
    adjacency : torch.Tensor
        Symmetric edge-weight matrix (n_items, n_items), zero diagonal.

    Returns
    -------
    torch.Tensor
        Strength per node, shape (n_items,).
    """
    return adjacency.abs().sum(dim=1)


def expected_influence(adjacency: torch.Tensor) -> torch.Tensor:
    """Expected influence: signed sum of edge weights.

    Unlike strength, this is sensitive to the *polarity* of edges and can
    be negative for nodes connected primarily by negative edges. Proposed
    by Robinaugh et al. (2016) for signed networks (e.g., symptom networks).

    Parameters
    ----------
    adjacency : torch.Tensor
        Symmetric edge-weight matrix (n_items, n_items), zero diagonal.

    Returns
    -------
    torch.Tensor
        Expected influence per node, shape (n_items,).

    References
    ----------
    .. [1] Robinaugh, D. J., et al. (2016). Identifying highly influential
           nodes in the complicated grief network. *Journal of Abnormal
           Psychology*, 125(6), 747–757.
    """
    return adjacency.sum(dim=1)


def _shortest_path_distances(adjacency: torch.Tensor) -> torch.Tensor:
    """All-pairs shortest path distances via Floyd-Warshall.

    Edge weights are treated as similarities; distances are ``1 / |w|``
    for non-zero weights and ``inf`` for absent edges.

    Parameters
    ----------
    adjacency : torch.Tensor
        Symmetric edge-weight matrix (n_items, n_items), zero diagonal.

    Returns
    -------
    torch.Tensor
        Distance matrix (n_items, n_items). Self-distances are 0.
    """
    n = adjacency.shape[0]
    W_abs = adjacency.abs()

    # Convert weights to distances: large weight → short distance
    dist = torch.where(
        W_abs > 0,
        1.0 / W_abs.clamp(min=1e-10),
        torch.full_like(W_abs, float("inf")),
    )
    dist.fill_diagonal_(0.0)

    # Floyd-Warshall O(n³)
    for k in range(n):
        candidate = dist[:, k : k + 1] + dist[k : k + 1, :]
        dist = torch.minimum(dist, candidate)

    return dist


def closeness_centrality(adjacency: torch.Tensor) -> torch.Tensor:
    """Closeness centrality: normalised reciprocal of mean shortest-path distance.

    Defined as ``(reachable − 1) / Σ dist(i, j)`` over all reachable j ≠ i,
    matching the Wasserman-Faust normalisation for possibly disconnected graphs.

    Parameters
    ----------
    adjacency : torch.Tensor
        Symmetric edge-weight matrix (n_items, n_items), zero diagonal.

    Returns
    -------
    torch.Tensor
        Closeness scores per node, shape (n_items,). Zero for isolated nodes.
    """
    dist = _shortest_path_distances(adjacency)
    finite = dist.isfinite()
    # Exclude self (diagonal) from reachable count
    finite.fill_diagonal_(False)

    reachable = finite.float().sum(dim=1)  # number of reachable others
    sum_dist = (dist * finite.float()).sum(dim=1).clamp(min=1e-10)

    closeness = reachable / sum_dist
    # Nodes with no reachable neighbours get zero
    closeness = torch.where(reachable > 0, closeness, torch.zeros_like(closeness))
    return closeness


def betweenness_centrality(adjacency: torch.Tensor) -> torch.Tensor:
    """Node betweenness centrality.

    For each node v, counts the fraction of (s, t) pairs (s < t, s ≠ v, t ≠ v)
    for which v lies on a shortest path. A node on a shortest path satisfies

        dist(s, v) + dist(v, t) ≈ dist(s, t).

    The result is normalised by ``(n−1)(n−2)/2``, the total number of
    source–target pairs.

    Parameters
    ----------
    adjacency : torch.Tensor
        Symmetric edge-weight matrix (n_items, n_items), zero diagonal.

    Returns
    -------
    torch.Tensor
        Betweenness per node in [0, 1], shape (n_items,).
    """
    n = adjacency.shape[0]
    dist = _shortest_path_distances(adjacency)  # (n, n)

    betweenness = torch.zeros(n, device=adjacency.device)

    # Upper-triangular mask for source < target pairs
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=adjacency.device), diagonal=1)
    # Only connected pairs
    connected = dist.isfinite() & triu  # (n, n)

    for v in range(n):
        # Mask: exclude s==v or t==v
        mask_v = connected.clone()
        mask_v[v, :] = False
        mask_v[:, v] = False

        # v is on s→t path: dist[s,v] + dist[v,t] ≈ dist[s,t]
        d_sv = dist[:, v]  # (n,)
        d_vt = dist[v, :]  # (n,)
        on_path = torch.isclose(
            dist,
            d_sv.unsqueeze(1) + d_vt.unsqueeze(0),
            atol=1e-6,
        )
        betweenness[v] = (on_path & mask_v).float().sum()

    norm = (n - 1) * (n - 2) / 2
    if norm > 0:
        betweenness = betweenness / norm

    return betweenness
