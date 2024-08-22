
import sys  # For handling command-line arguments
import numpy as np  # For handling arrays and numerical operations
import os  # For operating system interactions (file and directory handling)
import multiprocessing as mp  # For parallel processing

def load_one(base):
    """
    Loads logits for a given experiment, processes them to compute
    scored predictions, and saves the results.
    
    Args:
        base (str): The base directory for an experiment within the log directory.
    """
    root = os.path.join(logdir, base, 'logits')  # Path to the logits directory for this experiment
    if not os.path.exists(root):
        return None  # Return if the logits directory doesn't exist

    # Create a directory to save the scores if it doesn't already exist
    if not os.path.exists(os.path.join(logdir, base, 'scores')):
        os.mkdir(os.path.join(logdir, base, 'scores'))
    
    # Iterate over each file in the logits directory
    for f in os.listdir(root):
        try:
            # Load the logits from the file
            opredictions = np.load(os.path.join(root, f))
        except:
            print("Fail")  # Print fail message if the file can't be loaded
            continue

        # Numerically stable computation of softmax, as described in the paper
        predictions = opredictions - np.max(opredictions, axis=3, keepdims=True)  # Subtract max for numerical stability
        predictions = np.array(np.exp(predictions), dtype=np.float64)  # Compute exponentials of logits
        predictions = predictions / np.sum(predictions, axis=3, keepdims=True)  # Normalize to get probabilities

        COUNT = predictions.shape[0]  # Number of examples in this batch

        # Extract true class probabilities (y_true) based on the provided labels
        y_true = predictions[np.arange(COUNT), :, :, labels[:COUNT]]
        print(y_true.shape)  # Print the shape for debugging

        # Print the mean accuracy for this batch (how often the top prediction matches the true label)
        print('mean acc', np.mean(predictions[:, 0, 0, :].argmax(1) == labels[:COUNT]))
        
        # Set the true class probabilities to 0 in the predictions array (focus on wrong predictions)
        predictions[np.arange(COUNT), :, :, labels[:COUNT]] = 0
        y_wrong = np.sum(predictions, axis=3)  # Sum over the remaining (wrong) predictions

        # Compute logits difference (logit) between true and wrong class predictions
        logit = (np.log(y_true.mean(1) + 1e-45) - np.log(y_wrong.mean(1) + 1e-45))

        # Save the computed logit scores to the scores directory
        np.save(os.path.join(logdir, base, 'scores', f), logit)

def load_stats():
    """
    Load and process logits for all experiments in parallel.
    """
    with mp.Pool(8) as p:  # Use a multiprocessing pool with 8 workers
        # Apply the load_one function to all experiment directories containing 'exp' in their name
        p.map(load_one, [x for x in os.listdir(logdir) if 'exp' in x])

# Main execution
logdir = sys.argv[1]  # Get the log directory from command-line arguments
labels = np.load(os.path.join(logdir, "y_train.npy"))  # Load the true labels from the log directory
load_stats()  # Start processing all experiments
```

### How This Code Relates to the Previous Codes:
'''
1. **Logits Handling and Softmax Computation:**
   - This code deals with processing logits, which are outputs 
   from a neural network before applying the softmax function. 
   The `load_one` function computes the softmax in a numerically stable way by first subtracting the maximum logit value. This is important when working with models that output large values, which can lead to numerical instability in the exponential function.

2. **Prediction and Scoring:**
   - After computing the softmax probabilities, the code compares 
   the true class probabilities (extracted using the provided labels)
   with the probabilities of incorrect classes. 
   This aligns with typical post-processing steps in machine learning,
   where models' predictions are evaluated against true labels 
   to compute metrics like accuracy and confidence scores.

3. **Parallel Processing:**
   - The code uses `multiprocessing` to process multiple experiments 
   in parallel, similar to how training might be parallelized across 
   different configurations or datasets in the previous code examples. 
   This allows for faster processing of large datasets or 
   multiple experiment results.

4. **Integration with Experiment Workflow:**
   - This script seems to be part of a larger experiment pipeline, 
   likely running after models have been trained and logits have 
   been saved, which correlates with the model training and evaluation workflow from previous codes. The `logdir` directory structure and the use of labels from `y_train.npy` suggest this script fits into the overall experiment management and result processing tasks.

Overall, this code appears to be a post-processing script used 
to evaluate and compute scores from the logits generated by trained 
models in various experiments, leveraging numerical stability 
techniques and parallel processing for efficiency.

'''