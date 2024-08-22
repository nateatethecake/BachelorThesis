# Import necessary modules
import functools  # For higher-order functions, partial functions
import os  # For operating system interactions (file handling)
import shutil  # For file and directory management
from typing import Callable  # For type annotations
import json  # For handling JSON files

import jax  # For high-performance machine learning on GPU/TPU
import jax.numpy as jn  # JAX's version of numpy for GPU/TPU
import numpy as np  # For handling arrays and numerical data
import tensorflow as tf  # For data augmentation and handling
import tensorflow_datasets as tfds  # For loading datasets from TensorFlow Datasets
from absl import app, flags  # For application management and flag handling

import objax  # For building neural networks using JAX
from objax.jaxboard import SummaryWriter, Summary  # For logging training progress
from objax.util import EasyDict  # For easy access to dictionary attributes
from objax.zoo import convnet, wide_resnet  # Predefined models in Objax's zoo

from dataset import DataSet  # Import custom dataset class from the previous code

# Define command-line flags for configuration
FLAGS = flags.FLAGS

def augment(x, shift: int, mirror=True):
    """
    Augmentation function used during training to augment images by flipping, shifting, and cropping.
    
    Args:
        x (dict): A dictionary containing an image and its corresponding label.
        shift (int): Number of pixels to shift for padding.
        mirror (bool): Whether to randomly flip the image horizontally.
    
    Returns:
        dict: A dictionary with the augmented image and the original label.
    """
    y = x['image']
    if mirror:  # Apply random horizontal flip if mirror is True
        y = tf.image.random_flip_left_right(y)
    # Apply reflection padding
    y = tf.pad(y, [[shift] * 2, [shift] * 2, [0] * 2], mode='REFLECT')
    # Randomly crop the image back to the original shape
    y = tf.image.random_crop(y, tf.shape(x['image']))
    return dict(image=y, label=x['label'])

# Define a class for the training loop using Objax
class TrainLoop(objax.Module):
    """
    Class to define the training loop for machine learning models using Objax.
    Inherits from objax.Module.
    """
    predict: Callable  # Callable for making predictions
    train_op: Callable  # Callable for performing a training step

    def __init__(self, nclass: int, **kwargs):
        """
        Initialize the training loop.
        
        Args:
            nclass (int): Number of classes in the dataset.
            **kwargs: Additional parameters passed to the EasyDict.
        """
        self.nclass = nclass  # Number of classes
        self.params = EasyDict(kwargs)  # Store other parameters in an EasyDict

    def train_step(self, summary: Summary, data: dict, progress: np.ndarray):
        """
        Perform a single training step and log the progress.
        
        Args:
            summary (Summary): Object to record training statistics.
            data (dict): Batch of data containing images and labels.
            progress (np.ndarray): Progress indicator for learning rate scheduling.
        """
        # Execute the training operation and return key-value pairs of results
        kv = self.train_op(progress, data['image'].numpy(), data['label'].numpy())
        for k, v in kv.items():
            if jn.isnan(v):  # Check for NaN values in the results
                raise ValueError('NaN, try reducing learning rate', k)
            if summary is not None:  # Log the values if a summary object is provided
                summary.scalar(k, float(v))

    def train(self, num_train_epochs: int, train_size: int, train: DataSet, test: DataSet, logdir: str, save_steps=100, patience=None):
        """
        Perform the full training loop over multiple epochs.
        
        Args:
            num_train_epochs (int): Number of epochs to train.
            train_size (int): Size of the training dataset.
            train (DataSet): Training dataset.
            test (DataSet): Testing dataset for evaluation.
            logdir (str): Directory to save checkpoints and logs.
            save_steps (int): Frequency of saving checkpoints.
            patience (int): Number of epochs with no improvement after which training stops early.
        """
        # Create or load from a checkpoint
        checkpoint = objax.io.Checkpoint(logdir, keep_ckpts=20, makedir=True)
        start_epoch, last_ckpt = checkpoint.restore(self.vars())
        train_iter = iter(train)  # Create an iterator for the training dataset
        progress = np.zeros(jax.local_device_count(), 'f')  # Initialize progress tracker

        best_acc = 0  # Track the best accuracy
        best_acc_epoch = -1  # Track the epoch at which the best accuracy was achieved

        # Open a summary writer for TensorBoard logging
        with SummaryWriter(os.path.join(logdir, 'tb')) as tensorboard:
            for epoch in range(start_epoch, num_train_epochs):
                # Training phase
                summary = Summary()  # Initialize a new summary for this epoch
                loop = range(0, train_size, self.params.batch)
                for step in loop:
                    progress[:] = (step + (epoch * train_size)) / (num_train_epochs * train_size)
                    self.train_step(summary, next(train_iter), progress)

                # Evaluation phase
                accuracy, total = 0, 0
                if epoch % FLAGS.eval_steps == 0 and test is not None:
                    for data in test:
                        total += data['image'].shape[0]
                        preds = np.argmax(self.predict(data['image'].numpy()), axis=1)
                        accuracy += (preds == data['label'].numpy()).sum()
                    accuracy /= total
                    summary.scalar('eval/accuracy', 100 * accuracy)
                    tensorboard.write(summary, step=(epoch + 1) * train_size)
                    print('Epoch %04d  Loss %.2f  Accuracy %.2f' % (epoch + 1, summary['losses/xe'](),
                                                                    summary['eval/accuracy']()))

                    if summary['eval/accuracy']() > best_acc:
                        best_acc = summary['eval/accuracy']()
                        best_acc_epoch = epoch
                    elif patience is not None and epoch > best_acc_epoch + patience:
                        print("early stopping!")
                        checkpoint.save(self.vars(), epoch + 1)
                        return
                else:
                    print('Epoch %04d  Loss %.2f  Accuracy --' % (epoch + 1, summary['losses/xe']()))

                if epoch % save_steps == save_steps - 1:
                    checkpoint.save(self.vars(), epoch + 1)


# Define a memory module class inheriting from TrainLoop
class MemModule(TrainLoop):
    def __init__(self, model: Callable, nclass: int, mnist=False, **kwargs):
        """
        Initialize the memory module, which includes the model, optimizer, and EMA.
        
        Args:
            model (Callable): The model architecture to use.
            nclass (int): Number of classes.
            mnist (bool): Whether the dataset is MNIST.
            **kwargs: Additional parameters for training.
        """
        super().__init__(nclass, **kwargs)
        self.model = model(1 if mnist else 3, nclass)  # Initialize the model with 1 channel for MNIST, otherwise 3
        self.opt = objax.optimizer.Momentum(self.model.vars())  # Use momentum optimizer
        # Use Exponential Moving Average (EMA) for the model's weights
        self.model_ema = objax.optimizer.ExponentialMovingAverageModule(self.model, momentum=0.999, debias=True)

        # Define the loss function with weight decay
        @objax.Function.with_vars(self.model.vars())
        def loss(x, label):
            logit = self.model(x, training=True)
            loss_wd = 0.5 * sum((v.value ** 2).sum() for k, v in self.model.vars().items() if k.endswith('.w'))
            loss_xe = objax.functional.loss.cross_entropy_logits(logit, label).mean()
            return loss_xe + loss_wd * self.params.weight_decay, {'losses/xe': loss_xe, 'losses/wd': loss_wd}

        gv = objax.GradValues(loss, self.model.vars())  # Compute gradients and values from the loss
        self.gv = gv  # Store the gradient and value function

        # Define the training operation
        @objax.Function.with_vars(self.vars())
        def train_op(progress, x, y):
            g, v = gv(x, y)  # Compute gradients
            lr = self.params.lr * jn.cos(progress * (7 * jn.pi) / (2 * 8))  # Cosine annealing learning rate
            lr = lr * jn.clip(progress * 100, 0, 1)
            self.opt(lr, g)  # Apply gradients with momentum
            self.model_ema.update_ema()  # Update EMA for model weights
            return {'monitors/lr': lr, **v[1]}  # Return learning rate and loss values

        # Define the prediction function using Just-In-Time (JIT) compilation
        self.predict = objax.Jit(objax.nn.Sequential([objax.ForceArgs(self.model_ema, training=False)]))

        self.train_op = obj

ax.Jit(train_op)  # JIT compile the training operation


def network(arch: str):
    """
    Factory function to create network architectures based on a string identifier.
    
    Args:
        arch (str): The architecture name.
    
    Returns:
        Callable: The model constructor based on the chosen architecture.
    """
    if arch == 'cnn32-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=32, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn32-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=32, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'cnn64-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn64-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'wrn28-1':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=1)
    elif arch == 'wrn28-2':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=2)
    elif arch == 'wrn28-10':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=10)
    raise ValueError('Architecture not recognized', arch)  # Raise an error if the architecture is not recognized

def get_data(seed):
    """
    Function to prepare the training and testing datasets.
    
    Args:
        seed (int): Seed for random number generation.
    
    Returns:
        Tuple containing training dataset, testing dataset, training inputs and labels, 
        a mask for which examples to keep, and the number of classes.
    """
    DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')  # Set the dataset directory

    # Check if cached data exists, otherwise load from TensorFlow Datasets
    if os.path.exists(os.path.join(FLAGS.logdir, "x_train.npy")):
        inputs = np.load(os.path.join(FLAGS.logdir, "x_train.npy"))
        labels = np.load(os.path.join(FLAGS.logdir, "y_train.npy"))
    else:
        print("First time, creating dataset")
        data = tfds.as_numpy(tfds.load(name=FLAGS.dataset, batch_size=-1, data_dir=DATA_DIR))
        inputs = data['train']['image']
        labels = data['train']['label']

        # Normalize inputs to [-1, 1]
        inputs = (inputs/127.5) - 1
        # Save the inputs and labels to the log directory for future use
        np.save(os.path.join(FLAGS.logdir, "x_train.npy"), inputs)
        np.save(os.path.join(FLAGS.logdir, "y_train.npy"), labels)

    nclass = np.max(labels) + 1  # Determine the number of classes

    np.random.seed(seed)  # Set the random seed for reproducibility
    if FLAGS.num_experiments is not None:  # Handle multiple experiments
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(FLAGS.num_experiments, FLAGS.dataset_size))
        order = keep.argsort(0)
        keep = order < int(FLAGS.pkeep * FLAGS.num_experiments)
        keep = np.array(keep[FLAGS.expid], dtype=bool)
    else:
        keep = np.random.uniform(0, 1, size=FLAGS.dataset_size) <= FLAGS.pkeep

    if FLAGS.only_subset is not None:  # If only a subset should be used
        keep[FLAGS.only_subset:] = 0

    # Filter the inputs and labels based on the mask
    xs = inputs[keep]
    ys = labels[keep]

    # Determine the augmentation strategy
    if FLAGS.augment == 'weak':
        aug = lambda x: augment(x, 4)
    elif FLAGS.augment == 'mirror':
        aug = lambda x: augment(x, 0)
    elif FLAGS.augment == 'none':
        aug = lambda x: augment(x, 0, mirror=False)
    else:
        raise

    # Create training and testing datasets using the DataSet class
    train = DataSet.from_arrays(xs, ys, augment_fn=aug)
    test = DataSet.from_tfds(tfds.load(name=FLAGS.dataset, split='test', data_dir=DATA_DIR), xs.shape[1:])
    train = train.cache().shuffle(8192).repeat().parse().augment().batch(FLAGS.batch)
    train = train.nchw().one_hot(nclass).prefetch(16)
    test = test.cache().parse().batch(FLAGS.batch).nchw().prefetch(16)

    return train, test, xs, ys, keep, nclass

def main(argv):
    del argv  # Unused argument
    tf.config.experimental.set_visible_devices([], "GPU")  # Force the use of CPU

    seed = FLAGS.seed  # Use provided seed, or generate a new one
    if seed is None:
        import time
        seed = np.random.randint(0, 1000000000)
        seed ^= int(time.time())

    # Create a dictionary of arguments to pass to the MemModule
    args = EasyDict(arch=FLAGS.arch,
                    lr=FLAGS.lr,
                    batch=FLAGS.batch,
                    weight_decay=FLAGS.weight_decay,
                    augment=FLAGS.augment,
                    seed=seed)


    # Define the log directory based on various conditions
    if FLAGS.tunename:
        logdir = '_'.join(sorted('%s=%s' % k for k in args.items()))
    elif FLAGS.expid is not None:
        logdir = "experiment-%d_%d" % (FLAGS.expid, FLAGS.num_experiments)
    else:
        logdir = "experiment-" + str(seed)
    logdir = os.path.join(FLAGS.logdir, logdir)

    # Check if the experiment has already been completed
    if os.path.exists(os.path.join(logdir, "ckpt", "%010d.npz" % FLAGS.epochs)):
        print(f"run {FLAGS.expid} already completed.")
        return
    else:
        # If the log directory exists but the experiment wasn't completed, remove it
        if os.path.exists(logdir):
            print(f"deleting run {FLAGS.expid} that did not complete.")
            shutil.rmtree(logdir)

    print(f"starting run {FLAGS.expid}.")
    if not os.path.exists(logdir):
        os.makedirs(logdir)  # Create the log directory if it doesn't exist

    # Prepare data
    train, test, xs, ys, keep, nclass = get_data(seed)

    # Initialize the MemModule (network + training loop)
    tm = MemModule(network(FLAGS.arch), nclass=nclass,
                   mnist=FLAGS.dataset == 'mnist',
                   epochs=FLAGS.epochs,
                   expid=FLAGS.expid,
                   num_experiments=FLAGS.num_experiments,
                   pkeep=FLAGS.pkeep,
                   save_steps=FLAGS.save_steps,
                   only_subset=FLAGS.only_subset,
                   **args
    )

    r = {}
    r.update(tm.params)

    # Save hyperparameters and the mask for kept examples
    open(os.path.join(logdir, 'hparams.json'), "w").write(json.dumps(tm.params))
    np.save(os.path.join(logdir, 'keep.npy'), keep)

    # Start the training process
    tm.train(FLAGS.epochs, len(xs), train, test, logdir,
             save_steps=FLAGS.save_steps, patience=FLAGS.patience)

if __name__ == '__main__':
    # Define command-line flags for configuring the training process
    flags.DEFINE_string('arch', 'cnn32-3-mean', 'Model architecture.')
    flags.DEFINE_float('lr', 0.1, 'Learning rate.')
    flags.DEFINE_string('dataset', 'cifar10', 'Dataset.')
    flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay ratio.')
    flags.DEFINE_integer('batch', 256, 'Batch size')
    flags.DEFINE_integer('epochs', 501, 'Training duration in number of epochs.')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_integer('seed', None, 'Training seed.')
    flags.DEFINE_float('pkeep', .5, 'Probability to keep examples.')
    flags.DEFINE_integer('expid', None, 'Experiment ID')
    flags.DEFINE_integer('num_experiments', None, 'Number of experiments')
    flags.DEFINE_string('augment', 'weak', 'Strong or weak augmentation')
    flags.DEFINE_integer('only_subset', None, 'Only train on a subset of images.')
    flags.DEFINE_integer('dataset_size', 50000, 'number of examples to keep.')
    flags.DEFINE_integer('eval_steps', 1, 'how often to get eval accuracy.')
    flags.DEFINE_integer('abort_after_epoch', None, 'stop training early at an epoch')
    flags.DEFINE_integer('save_steps', 10, 'how often to save model.')
    flags.DEFINE_integer('patience', None, 'Early stopping after this many epochs without progress')
    flags.DEFINE_bool('tunename', False, 'Use tune

 name?')
    app.run(main)  # Start the application
```

### How This Code Relates to the Previous Codes:
'''
1. **Data Handling (`DataSet` Class):** This code uses the `DataSet` 
class from the previous code to manage and preprocess data. 
The `get_data` function loads the dataset, applies augmentations, 
and creates `DataSet` objects for training and testing.

2. **Training and Augmentation:** The `augment` function is 
consistent with the previous use of augmentation strategies. 
It applies transformations to the images to make the model more robust during training.

3. **Training Loop (`TrainLoop` and `MemModule`):** 
This code extends the training loop with specific functionalities. 
The `MemModule` class inherits from `TrainLoop`, and it is customized for models defined by the `network` function. The training loop handles logging, checkpointing, and early stopping, making it a practical implementation for iterative model training.

4. **Architecture Selection (`network` function):** 
The `network` function provides a way to select and instantiate 
different model architectures, similar to how architectures were defined and used in the previous code.

5. **Parameter Management:** The use of command-line flags for 
managing hyperparameters and configurations allows for flexibility 
and experimentation, which is consistent with the design philosophy 
in the previous code.