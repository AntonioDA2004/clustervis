import numpy as np
import matplotlib.pyplot as plt
from clustervis.colors import compute_weighted_rgb

def plot_decision_boundary(X, classifier, colors, resolution=50):
    """
    Plot the decision boundary of a classifier with weighted RGB visualization.
    """
    plt.figure(figsize=(6, 6))

    min_x, max_x = np.min(X[:, 0]), np.max(X[:, 0])
    min_y, max_y = np.min(X[:, 1]), np.max(X[:, 1])
    xx, yy = np.meshgrid(np.linspace(min_x - 1, max_x + 1, resolution),
                         np.linspace(min_y - 1, max_y + 1, resolution))

    rgb_grid = np.zeros((xx.shape[0], yy.shape[0], 3))

    colors_normalized = [(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0) for c in colors]

    for i in range(xx.shape[0]):
        for j in range(yy.shape[0]):
            point = np.array([[xx[i, j], yy[i, j]]])
            predictions = [est.predict(point)[0] for est in classifier.estimators_]
            class_weights = [predictions.count(c) for c in range(len(colors_normalized))]
            rgb_grid[i, j] = compute_weighted_rgb(class_weights, colors_normalized)

    plt.imshow(rgb_grid, extent=(min_x - 1, max_x + 1, min_y - 1, max_y + 1), origin='lower')

    point_predictions = classifier.predict(X)
    point_colors = np.array([colors_normalized[p] for p in point_predictions])
    plt.scatter(X[:, 0], X[:, 1], c=point_colors, edgecolor='black', s=20)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Classifier Model Decision Boundaries')
    plt.show()