import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import openml
import time
import matplotlib.pyplot as plt
import torch
import pickle

from pathlib import Path
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tabpfn.scripts.decision_boundary import DecisionBoundaryDisplay
from tabpfn import TabPFNClassifier

# Ensure the output directory exists
output_dir = Path('ru89vot2/TabPFN_Project')
output_dir.mkdir(parents=True, exist_ok=True)

# Get dataset by ID
dataset = openml.datasets.get_dataset(15)
print(dataset)

X, y, _, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute,
    dataset_format='dataframe'
)
print("This is X is the training data and y is the target")
print(X, y)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the classifier
print("This is the classifier")
classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)

print("Start training")
start = time.time()
classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)
print('Prediction time: ', time.time() - start, 'Accuracy', accuracy_score(y_test, y_eval))

classifier.predict_proba(X_test).shape

out_table = pd.DataFrame(X_test.copy().astype(str))
out_table['prediction'] = [f"{y_e} (p={p_e:.2f})" for y_e, p_e in zip(y_eval, p_eval)]
print(out_table)

# Save the model using pickle
model_path = output_dir / 'tabpfn_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(classifier, f)
print(f"Model saved to {model_path}")

# PLOTTING
print("Define the plots")
fig, ax = plt.subplots(figsize=(10, 10))

# Using the RdBu colormap from matplotlib
cmap = plt.get_cmap('RdBu')

# Custom ListedColormap
print("Setting custom colors")
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

# Plot the training points
print("Plot the training points")
vfunc = np.vectorize(lambda x: np.where(classifier.classes_ == x)[0])
y_train_index = vfunc(y_train)
y_train_index = y_train_index == 0

ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train_index, cmap=cm_bright)

# Display decision boundary from estimator
#print("This is the decision boundary")
#DecisionBoundaryDisplay.from_estimator(
#    classifier, X_train.iloc[:, 0:2], alpha=0.6, ax=ax, eps=2.0, grid_resolution=25, response_method="predict_proba"
#)

# Save the plot
plot_path = output_dir / 'decision_boundary.png'
plt.savefig(plot_path)
print(f"Decision boundary plot saved to {plot_path}")

plt.close(fig)

# Simple plot
print("This is a simple plot")
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Simple Plot")
plt.xlabel("x")
plt.ylabel("sin(x)")

# Save the simple plot
simple_plot_path = output_dir / 'simple_plot.png'
plt.savefig(simple_plot_path)
print(f"Simple plot saved to {simple_plot_path}")

plt.close()

# Evaluation Metrics
print("Evaluating model")
accuracy = accuracy_score(y_test, y_eval)
f1 = f1_score(y_test, y_eval, average='macro')
conf_matrix = confusion_matrix(y_test, y_eval)
report = classification_report(y_test, y_eval)

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

# Save evaluation metrics
metrics_path = output_dir / 'metrics.txt'
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"F1-Score: {f1}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\nClassification Report:\n")
    f.write(report)
print(f"Metrics saved to {metrics_path}")
