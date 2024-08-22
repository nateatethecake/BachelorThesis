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

    # Define and train the target model (TabPFN classifier)
    print("Training the target model (TabPFN)...")
    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
    classifier.fit(X_train, y_train)

    # Save the trained target model
    output_dir = Path('ru89vot2/TabPFN_Project')
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'diabetes_tabpfn_model.pkl'
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
