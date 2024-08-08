import numpy as np
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
from tabpfn import TabPFNClassifier
from scipy.stats import norm

# Load the trained target model
output_dir = Path('/Users/nwd/Code/Research/')
model_path = output_dir / 'tabpfn_model.joblib'
classifier = joblib.load(model_path)

# Define the SYNTHESIZE procedure (as in the previous implementation)
def synthesize(target_model, class_label, k_max=10, iter_max=1000, conf_min=0.5, rej_max=10, k_min=1):
    def rand_record():
        # Generate a random record
        return np.random.rand(X_train.shape[1])
    
    x = rand_record()
    y_star_c = 0
    j = 0
    k = k_max
    
    for iteration in range(iter_max):
        y = target_model.predict_proba([x])[0]  # Query the target model
        y_c = y[class_label]
        
        if y_c >= y_star_c:
            if y_c > conf_min and class_label == np.argmax(y):
                if random.random() < y_c:
                    return x  # Return synthetic data
            x_star = x
            y_star_c = y_c
            j = 0
        else:
            j += 1
            if j > rej_max:
                k = max(k_min, k // 2)
                j = 0
        
        x = rand_record()
    
    return None  # Failed to synthesize

# Generate synthetic dataset
def generate_synthetic_data(target_model, num_samples, class_label):
    synthetic_data = []
    for _ in range(num_samples):
        data_point = None
        while data_point is None:
            data_point = synthesize(target_model, class_label)
        synthetic_data.append(data_point)
    return np.array(synthetic_data)

# Train shadow models
def train_shadow_models(target_model, num_shadow_models, num_samples_per_model, class_label):
    shadow_models = []
    for _ in range(num_shadow_models):
        synthetic_data = generate_synthetic_data(target_model, num_samples_per_model, class_label)
        X_synth_train, _, y_synth_train, _ = train_test_split(
            synthetic_data, np.full(num_samples_per_model, class_label), test_size=0.2, random_state=42)
        
        shadow_model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
        shadow_model.fit(X_synth_train, y_synth_train)
        shadow_models.append(shadow_model)
    return shadow_models

# Calculate loss (assuming cross-entropy loss for this example)
def calculate_loss(model, data_point, true_label):
    proba = model.predict_proba([data_point])[0]
    loss = -np.log(proba[true_label])
    return loss

# Estimate distributions Q_in and Q_out
def estimate_distributions(target_model, shadow_models, data_point, true_label):
    losses_in = []
    losses_out = []
    
    # Train shadow models on datasets containing the target point (Q_in)
    for shadow_model in shadow_models:
        loss = calculate_loss(shadow_model, data_point, true_label)
        losses_in.append(loss)
    
    # Train shadow models on datasets not containing the target point (Q_out)
    for shadow_model in shadow_models:
        X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(
            shadow_model._tabular_X, shadow_model._tabular_y, test_size=0.2, random_state=42)
        shadow_model.fit(X_shadow_train, y_shadow_train)
        loss = calculate_loss(shadow_model, data_point, true_label)
        losses_out.append(loss)
    
    # Estimate parameters for Gaussian distributions
    mean_in, std_in = np.mean(losses_in), np.std(losses_in)
    mean_out, std_out = np.mean(losses_out), np.std(losses_out)
    
    return mean_in, std_in, mean_out, std_out

# Perform likelihood-ratio test
def likelihood_ratio_test(target_model, shadow_models, data_point, true_label):
    mean_in, std_in, mean_out, std_out = estimate_distributions(target_model, shadow_models, data_point, true_label)
    
    # Calculate the loss on the target model
    loss = calculate_loss(target_model, data_point, true_label)
    
    # Calculate likelihoods under Q_in and Q_out
    likelihood_in = norm.pdf(loss, mean_in, std_in)
    likelihood_out = norm.pdf(loss, mean_out, std_out)
    
    # Perform the likelihood-ratio test
    ratio = likelihood_in / likelihood_out
    return ratio

# Generate shadow models
num_shadow_models = 5
num_samples_per_model = 1000
class_label = 0  # Adjust based on your class labels
shadow_models = train_shadow_models(classifier, num_shadow_models, num_samples_per_model, class_label)

# Test likelihood-ratio test
test_data_point = X_train.iloc[0].values  # Example data point from original training data
true_label = y_train.iloc[0]
ratio = likelihood_ratio_test(classifier, shadow_models, test_data_point, true_label)
print(f"Likelihood-ratio: {ratio}")

# Decision threshold (you might need to tune this threshold)
threshold = 1.0
is_member = ratio > threshold
print(f"Is the data point a member? {'Yes' if is_member else 'No'}")
