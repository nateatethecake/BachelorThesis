# Import necessary libraries
import json  # For reading JSON configuration files
import os  # For file and directory management
import re  # For regular expression operations

import numpy as np  # For numerical operations on arrays
import objax  # For building and training neural networks using Objax
import tensorflow as tf  # For data augmentation and possibly some operations
from absl import app  # For defining the application entry point
from absl import flags  # For handling command-line flags

# Import custom modules for memory management and network creation
from train import MemModule
from train import network

# Initialize the FLAGS object to store command-line flags
FLAGS = flags.FLAGS

def main(argv):
    """
    Main function to perform inference on a saved model.
    It generates output logits using a set of data augmentations.
    """
    del argv  # Unused argument is removed
    tf.config.experimental.set_visible_devices([], "GPU")  # Disable GPU visibility for TensorFlow (force CPU execution)

    def load(arch):
        """
        Load the MemModule with specified architecture.
        """
        return MemModule(
            network(arch),  # Load the network with the given architecture
            nclass=100 if FLAGS.dataset == 'cifar100' else 10,  # Set number of classes based on dataset
            mnist=FLAGS.dataset == 'mnist',  # Check if the dataset is MNIST
            arch=arch,  # Set architecture
            lr=.1,  # Learning rate (not used here)
            batch=0,  # Batch size (not used here)
            epochs=0,  # Number of epochs (not used here)
            weight_decay=0  # Weight decay (not used here)
        )

    def cache_load(arch):
        """
        Cache the loaded model to avoid redundant loads.
        """
        thing = []  # Create an empty list to store the model
        def fn():
            if len(thing) == 0:  # If the model is not loaded yet
                thing.append(load(arch))  # Load and cache the model
            return thing[0]  # Return the cached model
        return fn

    # Load training data (first 'dataset_size' samples) from the specified directory
    xs_all = np.load(os.path.join(FLAGS.logdir, "x_train.npy"))[:FLAGS.dataset_size]
    ys_all = np.load(os.path.join(FLAGS.logdir, "y_train.npy"))[:FLAGS.dataset_size]

    def get_loss(model, xbatch, ybatch, shift, reflect=True, stride=1):
        """
        Compute the loss for a batch of data with specified augmentations.
        """
        outs = []  # List to store outputs

        # Iterate over original and reflected batches (if reflect is True)
        for aug in [xbatch, xbatch[:,:,::-1,:]][:reflect+1]:
            # Pad the images using reflection padding
            aug_pad = tf.pad(aug, [[0] * 2, [shift] * 2, [shift] * 2, [0] * 2], mode='REFLECT').numpy()
            # Iterate over different shifts in x and y directions
            for dx in range(0, 2*shift+1, stride):
                for dy in range(0, 2*shift+1, stride):
                    # Extract a sub-image after shifting and transpose for correct shape
                    this_x = aug_pad[:, dx:dx+32, dy:dy+32, :].transpose((0, 3, 1, 2))
                    # Compute logits using the model
                    logits = model.model(this_x, training=True)
                    # Append logits to the output list
                    outs.append(logits)

        # Print the shape of the outputs array for debugging
        print(np.array(outs).shape)
        # Return the outputs in the required shape
        return np.array(outs).transpose((1, 0, 2))

    N = 5000  # Set batch size for processing

    def features(model, xbatch, ybatch):
        """
        Wrapper around get_loss to get model features.
        """
        return get_loss(model, xbatch, ybatch, shift=0, reflect=True, stride=1)

    # Iterate over each path in the log directory sorted alphabetically
    for path in sorted(os.listdir(os.path.join(FLAGS.logdir))):
        if re.search(FLAGS.regex, path) is None:
            # Skip files that do not match the regex pattern
            print("Skipping from regex")
            continue

        # Load hyperparameters from the JSON file in the path
        hparams = json.load(open(os.path.join(FLAGS.logdir, path, "hparams.json")))
        arch = hparams['arch']  # Extract architecture from hyperparameters
        model = cache_load(arch)()  # Load the model using the cached loader

        logdir = os.path.join(FLAGS.logdir, path)  # Set the log directory path

        # Restore the latest checkpoint for the model
        checkpoint = objax.io.Checkpoint(logdir, keep_ckpts=10, makedir=True)
        max_epoch, last_ckpt = checkpoint.restore(model.vars())
        if max_epoch == 0:
            continue  # Skip if no checkpoint is found

        # Create a directory to store logits if it doesn't exist
        if not os.path.exists(os.path.join(FLAGS.logdir, path, "logits")):
            os.mkdir(os.path.join(FLAGS.logdir, path, "logits"))
        if FLAGS.from_epoch is not None:
            first = FLAGS.from_epoch  # Start from the specified epoch
        else:
            first = max_epoch-1  # Otherwise, start from the second last epoch

        # Iterate over each epoch from 'first' to 'max_epoch'
        for epoch in range(first, max_epoch+1):
            if not os.path.exists(os.path.join(FLAGS.logdir, path, "ckpt", "%010d.npz"%epoch)):
                # Skip if checkpoint for this epoch is not saved
                continue

            if os.path.exists(os.path.join(FLAGS.logdir, path, "logits", "%010d.npy"%epoch)):
                # Skip if logits have already been generated for this epoch
                print("Skipping already generated file", epoch)
                continue

            try:
                # Attempt to restore the model for the current epoch
                start_epoch, last_ckpt = checkpoint.restore(model.vars(), epoch)
            except:
                print("Fail to load", epoch)  # Print error if failed to load checkpoint
                continue

            stats = []  # List to accumulate statistics

            # Iterate over the dataset in chunks of size N
            for i in range(0, len(xs_all), N):
                # Extend the stats list with features computed for each chunk
                stats.extend(features(model, xs_all[i:i+N], ys_all[i:i+N]))
            
            # Save the computed logits to a file
            np.save(os.path.join(FLAGS.logdir, path, "logits", "%010d"%epoch),
                    np.array(stats)[:,None,:,:])

if __name__ == '__main__':
    # Define command-line flags
    flags.DEFINE_string('dataset', 'cifar10', 'Dataset.')
    flags.DEFINE_string('logdir', 'experiments/', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_string('regex', '.*experiment.*', 'keep files when matching')
    flags.DEFINE_integer('dataset_size', 50000, 'size of dataset.')
    flags.DEFINE_integer('from_epoch', None, 'which epoch to load from.')
    # Run the main function using the defined command-line flags
    app.run(main)
