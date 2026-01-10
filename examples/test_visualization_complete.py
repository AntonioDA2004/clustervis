from clustervis import ensemble_classifier_plot, base_classifier_plot

import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)

# Define some colors for each class
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

# Define the resolution of the plots
resolution = 100

# Define the percentage selected for KNN classifiers
percentageSelected = 0.05

# Initialize the base classifier and bagging ensemble
clf = KNeighborsClassifier(n_neighbors=3)
bag = BaggingClassifier(clf, n_estimators=10, max_samples=percentageSelected, random_state=1)
bag.fit(X, y)

# We will create a separate axis for this plot to avoid it getting mixed with the KNN plots.
fig_ensemble, ax_ensemble = plt.subplots(figsize=(5, 5))

# First, plot the decision boundaries for the ensemble (Bagging) in a separate plot
ensemble_classifier_plot(X, bag, colors, resolution, 'Bagging Classifier', show=True, ax=ax_ensemble, plotPath='C:/Users/anton/Desktop/Antonio/University/Programming/IntelliJ/Projects/University/GitHub/clustervis', fileName='baggingClassifier.png')

# Define the number of KNN classifiers in the bagging ensemble
n_knns = len(bag.estimators_)

# Create a 2-row, 5-column grid for KNN plots
fig, axes = plt.subplots(2, 5, figsize=(15, 10))

# Flatten the axes array in case of a 2D grid (2 rows, 5 columns)
axes = axes.flatten()

# Now, use clustervis to plot the decision boundaries for the base classifiers (KNN)
for i, base_estimator in enumerate(bag.estimators_):
    base_classifier_plot(X, base_estimator, colors, resolution, f'KNN #{i+1}', False, axes[i], percentageSelected, plotPath='C:/Users/anton/Desktop/Antonio/University/Programming/IntelliJ/Projects/University/GitHub/clustervis', fileName='KNNClassifiers.png')

# Adjust layout for tight spacing between plots
plt.subplots_adjust(wspace=0.4, hspace=0.1)  # Add space between subplots

# Show the plots
plt.show()