def compute_weighted_rgb(weights, colors):
    """
    Compute the weighted RGB value for each pixel.
    """
    total_weight = sum(weights)

    if total_weight == 0:
        return (0.0, 0.0, 0.0)

    # Compute weighted sum for each RGB component
    rgb = tuple(
        sum(w * c[i] for w, c in zip(weights, colors)) / total_weight
        for i in range(3)
    )

    return rgb