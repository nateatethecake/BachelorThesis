
### Part 1: Regular Training with TabPFN Classifier


# Train and save the model
classifier.fit(X_train, y_train)  # Train the TabPFN classifier on the training data
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)  # Predict on the test data

# Save the trained model using pickle
model_path = output_dir / 'diabetes_tabpfn_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(classifier, f)
print(f"Model saved to {model_path}")
```

### Part 2: Implementing the Membership Inference Attack

# Load necessary modules
import numpy as np
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
from tabpfn import TabPFNClassifier
from scipy.stats import norm

# Load the trained target model
output_dir = Path('ru89vot2/TabPFN_Project')  # Ensure this matches the directory in Part 1
model_path = output_dir / 'diabetes_tabpfn_model.pkl'
with open(model_path, 'rb') as f:
    classifier = pickle.load(f)

# Define the SYNTHESIZE procedure to generate synthetic data
def synthesize(target_model, class_label, k_max=10, iter_max=1000, conf_min=0.5, rej_max=10, k_min=1):
    def rand_record():
        return np.random.rand(X_train.shape[1])  # Generate a random feature vector with the same number of features as X_train

    x = rand_record()
    y_star_c = 0
    j = 0
    k = k_max
    
    for iteration in range(iter_max):
        y = target_model.predict_proba([x])[0]  # Query the target model to get probabilities
        y_c = y[class_label]
        
        if y_c >= y_star_c:
            if y_c > conf_min and class_label == np.argmax(y):
                if random.random() < y_c:
                    return x  # Return the synthetic data point if it meets the criteria
            x_star = x
            y_star_c = y_c
            j = 0
        else:
            j += 1
            if j > rej_max:
                k = max(k_min, k // 2)
                j = 0
        
        x = rand_record()
    
    return None  # Return None if synthesis fails after iter_max iterations

# Generate synthetic data for training shadow models
def generate_synthetic_data(target_model, num_samples, class_label):
    synthetic_data = []
    for _ in range(num_samples):
        data_point = None
        while data_point is None:
            data_point = synthesize(target_model, class_label)  # Continuously attempt to synthesize data until successful
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
        
        # Save each shadow model
        shadow_model_path = output_dir / f'shadow_model_{_}.pkl'
        with open(shadow_model_path, 'wb') as f:
            pickle.dump(shadow_model, f)
        print(f"Shadow model {_} saved to {shadow_model_path}")

    return shadow_models

# Calculate loss using cross-entropy
def calculate_loss(model, data_point, true_label):
    proba = model.predict_proba([data_point])[0]
    loss = -np.log(proba[true_label] + 1e-15)  # Add a small constant to avoid log(0)
    return loss

# Estimate the distributions Q_in and Q_out
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

# Generate shadow models for the attack
num_shadow_models = 5  # Number of shadow models to train
num_samples_per_model = 1000  # Number of synthetic samples per shadow model
class_label = 0  # The class label to target for the attack (adjust as needed)

# Train shadow models
shadow_models = train_shadow_models(classifier, num_shadow_models, num_samples_per_model, class_label)

# Test the likelihood-ratio test on a known data point
test_data_point = X_train.iloc[0].values  # Example data point from original training data
true_label = y_train.iloc[0]
ratio = likelihood_ratio_test(classifier, shadow_models, test_data_point, true_label)
print(f"Likelihood-ratio: {ratio}")

# Decision threshold (tune this threshold based on ROC curve)
threshold = 1.0
is_member = ratio > threshold
print(f"Is the data point a member? {'Yes' if is_member else 'No'}")
