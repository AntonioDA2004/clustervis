# Clustervis

Clustervis is a Python package for visualizing clustering results using a Bagging Classifier. It provides a visual representation of decision boundaries.

## Features
- Visualize decision boundaries with color-coded cluster regions.

## Installation

To install Clustervis, clone the repository and install it using pip:
```sh
pip install .
```

## Usage

```python
from clustervis import plot_decision_boundary

from sklearn.datasets import make_blobs
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, random_state=76, cluster_std=1.0)

# Step 2: Train the BaggingClassifier
base_estimator = KNeighborsClassifier(n_neighbors=3)
bagging_classifier = BaggingClassifier(estimator=base_estimator, n_estimators=8, max_samples=0.05, random_state=1)
bagging_classifier.fit(X, y)

# Step 3: Define some colors for each class (e.g., for 4 classes)
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]  # Red, Green, Blue, Yellow

# Step 4: Plot the decision boundary
plot_decision_boundary(X, bagging_classifier, colors, 50)
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
- **GitHub:** https://github.com/antonioda2004/clustervis