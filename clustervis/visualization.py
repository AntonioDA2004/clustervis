import numpy as np
import matplotlib.pyplot as plt
from clustervis.colors import compute_weighted_rgb

def ensemble_classifier_plot(X, ensembleClassifier, colors, resolution, plotTitle, show, ax, plotPath=None, fileName=""):
    """
    Plot the decision boundary of an ensemble classifier with weighted RGB visualization.
    """
    
    min_x, max_x = np.min(X[:, 0]), np.max(X[:, 0])
    min_y, max_y = np.min(X[:, 1]), np.max(X[:, 1])
    xx, yy = np.meshgrid(np.linspace(min_x - 1, max_x + 1, resolution),
                         np.linspace(min_y - 1, max_y + 1, resolution))

    rgb_grid = np.zeros((xx.shape[0], yy.shape[0], 3))

    colors_normalized = [(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0) for c in colors]

    for i in range(xx.shape[0]):
        for j in range(yy.shape[0]):
            point = np.array([[xx[i, j], yy[i, j]]])
            predictions = [est.predict(point)[0] for est in ensembleClassifier.estimators_]
            class_weights = [predictions.count(c) for c in range(len(colors_normalized))]
            rgb_grid[i, j] = compute_weighted_rgb(class_weights, colors_normalized)

    ax.imshow(rgb_grid, extent=(min_x - 1, max_x + 1, min_y - 1, max_y + 1), origin='lower')

    point_predictions = ensembleClassifier.predict(X)
    point_colors = np.array([colors_normalized[p] for p in point_predictions])
    ax.scatter(X[:, 0], X[:, 1], c=point_colors, edgecolor='black', s=20)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(plotTitle)

    if plotPath is not None:
        plt.savefig(f"{plotPath}/{fileName}")

    if show is True:
        plt.show()

def base_classifier_plot(X, baseClassifier, colors, resolution, plotTitle, show, ax, percentageSelected, plotPath=None, fileName=""):
    """
    Plot the decision boundary of a base classifier with weighted RGB visualization.
    """

    min_x, max_x = np.min(X[:, 0]), np.max(X[:, 0])
    min_y, max_y = np.min(X[:, 1]), np.max(X[:, 1])
    xx, yy = np.meshgrid(np.linspace(min_x - 1, max_x + 1, resolution),
                         np.linspace(min_y - 1, max_y + 1, resolution))

    rgb_grid = np.zeros((xx.shape[0], yy.shape[0], 3))

    colors_normalized = [(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0) for c in colors]

    for i in range(xx.shape[0]):
        for j in range(yy.shape[0]):
            point = np.array([[xx[i, j], yy[i, j]]])
            prediction = baseClassifier.predict(point)[0]
            rgb_grid[i, j] = colors_normalized[prediction]

    ax.imshow(rgb_grid, extent=(min_x - 1, max_x + 1, min_y - 1, max_y + 1), origin='lower')

    point_predictions = baseClassifier.predict(X)

    if 0 < percentageSelected < 1:
        num_points_to_display = int(percentageSelected * len(X))
        indices_to_display = np.random.choice(len(X), num_points_to_display, replace=False)
        X_subset = X[indices_to_display]
        y_subset = point_predictions[indices_to_display]
        point_colors_subset = [colors_normalized[p] for p in y_subset]
        ax.scatter(X_subset[:, 0], X_subset[:, 1], c=point_colors_subset, edgecolor='black', s=20)
    else:
        point_colors = [colors_normalized[p] for p in point_predictions]
        ax.scatter(X[:, 0], X[:, 1], c=point_colors, edgecolor='black', s=20)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(plotTitle)

    if plotPath is not None:
        plt.savefig(f"{plotPath}/{fileName}")
    if show is True:
        plt.show()