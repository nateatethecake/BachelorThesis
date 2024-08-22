from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import openml
import pickle
from pathlib import Path

def train_shadow_models(classifier, num_shadow_models, num_samples_per_model, class_label):
    shadow_models = []
    for _ in range(num_shadow_models):
        # Sample data for the shadow model
        X_shadow, _, y_shadow, _ = train_test_split(X, y, train_size=num_samples_per_model, stratify=y)
        shadow_model = RandomForestClassifier(n_estimators=100, random_state=42)
        shadow_model.fit(X_shadow, y_shadow)
        shadow_models.append(shadow_model)
    return shadow_models

def likelihood_ratio_test(target_model, shadow_models, test_data_point, true_label):
    target_proba = target_model.predict_proba([test_data_point])[0][true_label]
    shadow_probabilities = [
        model.predict_proba([test_data_point])[0][true_label] for model in shadow_models
    ]
    shadow_proba_mean = sum(shadow_probabilities) / len(shadow_probabilities)
    return target_proba / (shadow_proba_mean + 1e-10)  # Add small epsilon to avoid division by zero

def evaluate_membership_inference(shadow_models, target_model, X_test, y_test):
    # Example evaluation, can be customized based on your membership inference evaluation methods
    # Here you would calculate and plot the ROC curve or other metrics.
    pass  # Placeholder

def main():
    # Load and preprocess the dataset
    print("Loading dataset...")
    dataset = openml.datasets.get_dataset(37)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define and train the target model (Random Forest classifier)
    print("Training the target model (Random Forest)...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Save the trained target model
    output_dir = Path('ru89vot2/RandomForest_Project')
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'diabetes_random_forest_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Target model saved to {model_path}")

    # Train shadow models for the membership inference attack
    print("Training shadow models...")
    num_shadow_models = 5
    num_samples_per_model = 1000
    class_label = 0  # Adjust based on your class labels
    shadow_models = train_shadow_models(classifier, num_shadow_models, num_samples_per_model, class_label)

    # Test the likelihood-ratio test on a known data point
    test_data_point = X_train.iloc[0].values  # Example data point from original training data
    true_label = y_train.iloc[0]
    ratio = likelihood_ratio_test(classifier, shadow_models, test_data_point, true_label)
    print(f"Likelihood-ratio: {ratio}")

    # Evaluate the attack and plot ROC curve
    evaluate_membership_inference(shadow_models, classifier, X_test, y_test)

if __name__ == '__main__':
    main()
