# Clustervis

Clustervis is a Python package for visualizing clustering results. It provides a visual representation of decision boundaries.

![classifier.png](classifier.png)

## Features
- Visualize decision boundaries with color-coded cluster regions.
- Save the plot in a path as an image (optional).

## Installation

To install Clustervis you can use pip:
```sh
pip install clustervis
```

## Usage

```python
from clustervis import plot_decision_boundaries

from sklearn.datasets import make_blobs
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, random_state=76, cluster_std=1.0)

# Step 2: Train a classifier (e.g., a Bagging Classifier)
base_estimator = KNeighborsClassifier(n_neighbors=3)
bagging_classifier = BaggingClassifier(estimator=base_estimator, n_estimators=8, max_samples=0.05, random_state=1)
bagging_classifier.fit(X, y)

# Step 3: Define some colors for each class (e.g., for 4 classes)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

# Step 4: Declare a path to save the plot (optional)
path = "/data/notebook_files" # Example path for Jetbrains Datalore
filename = "classifier.png"

# Step 5: Plot the decision boundary
plot_decision_boundaries(X, bagging_classifier, colors, 100, path, filename)
```

## Running Tests

To run unit tests, use:
```sh
python -m unittest discover tests
```

## License

This project is licensed under the MIT License.

## Author

- **Antonio De Angelis**  
- **Email:** deangelis.antonio122@gmail.com  
- **PyPI:** [https://pypi.org/project/clustervis/](https://pypi.org/project/clustervis/)