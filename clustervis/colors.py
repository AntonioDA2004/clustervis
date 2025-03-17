def compute_weighted_rgb(weights, colors):
    """
    Compute the weighted RGB value based on classifier votes.
    """
    total_weight = sum(weights)
    if total_weight == 0:
        return (0.0, 0.0, 0.0)

    return tuple(sum(w * c[i] for w, c in zip(weights, colors)) / total_weight for i in range(3))