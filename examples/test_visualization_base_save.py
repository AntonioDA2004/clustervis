from clustervis import base_classifier_plot

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, random_state=76, cluster_std=1.0)

# Step 2: Train a base classifier (e.g., a KNN Classifier)
base_estimator = KNeighborsClassifier(n_neighbors=3)
base_estimator.fit(X, y)

# Step 3: Define some colors for each class (e.g., for 4 classes)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

# Step 4: Declare the name, the resolution and the visibility of the plot
plotTitle = 'RGB Clustering Decision Boundaries (KNN Classifier)'
resolution = 100
show = True

# Step 5: Declare a path to save the plot
plotPath = "/data/notebook_files" # Example path for JetBrains Datalore
fileName = "classifier.png"

# Step 6: Create a figure and an axes
fig, ax = plt.subplots()

# Step 7: Declare the percentage of points selected
percentageSelected = 1.0

# Step 8: Plot the decision boundary and save it
base_classifier_plot(X, base_estimator, colors, resolution, plotTitle, show, ax, percentageSelected, plotPath, fileName)