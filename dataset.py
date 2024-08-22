
# Import necessary types for type hinting and defining function signatures
from typing import Callable, Optional, Tuple, List

# Importing necessary libraries
import numpy as np  # For handling arrays and numerical data
import tensorflow as tf  # For TensorFlow operations, including data processing

def record_parse(serialized_example: str, image_shape: Tuple[int, int, int]):
    """
    Parses a serialized TFRecord example into a dictionary with an image and label.
    
    Args:
        serialized_example (str): A single serialized example from a TFRecord.
        image_shape (Tuple[int, int, int]): The expected shape of the image (height, width, channels).

    Returns:
        dict: A dictionary containing the parsed image and label.
    """
    # Define the features to extract from the serialized example
    features = tf.io.parse_single_example(serialized_example,
                                          features={
                                              'image': tf.io.FixedLenFeature([], tf.string),  # Raw image data (encoded as a string)
                                              'label': tf.io.FixedLenFeature([], tf.int64)    # Label associated with the image
                                          })
    # Decode the image from the serialized string and set its shape
    image = tf.image.decode_image(features['image']).set_shape(image_shape)
    # Normalize the image to the range [-1, 1]
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    # Return a dictionary containing the image and the corresponding label
    return dict(image=image, label=features['label'])

# Define a class for managing and extending tf.data.Dataset objects
class DataSet:
    """Wrapper for tf.data.Dataset to permit extensions."""

    def __init__(self, data: tf.data.Dataset,
                 image_shape: Tuple[int, int, int],
                 augment_fn: Optional[Callable] = None,
                 parse_fn: Optional[Callable] = record_parse):
        """
        Initialize the DataSet object.
        
        Args:
            data (tf.data.Dataset): The dataset object.
            image_shape (Tuple[int, int, int]): The shape of the images in the dataset.
            augment_fn (Optional[Callable]): Function for data augmentation (if any).
            parse_fn (Optional[Callable]): Function for parsing the dataset examples.
        """
        self.data = data  # Store the dataset object
        self.parse_fn = parse_fn  # Store the parsing function
        self.augment_fn = augment_fn  # Store the augmentation function
        self.image_shape = image_shape  # Store the image shape

    @classmethod
    def from_arrays(cls, images: np.ndarray, labels: np.ndarray, augment_fn: Optional[Callable] = None):
        """
        Create a DataSet instance from NumPy arrays of images and labels.
        
        Args:
            images (np.ndarray): NumPy array of images.
            labels (np.ndarray): NumPy array of labels.
            augment_fn (Optional[Callable]): Function for data augmentation (if any).
            
        Returns:
            DataSet: An instance of the DataSet class.
        """
        # Convert arrays into a tf.data.Dataset and initialize a DataSet instance
        return cls(tf.data.Dataset.from_tensor_slices(dict(image=images, label=labels)), 
                   images.shape[1:],  # Image shape inferred from the array
                   augment_fn=augment_fn, parse_fn=None)  # No parsing function needed for arrays

    @classmethod
    def from_files(cls, filenames: List[str],
                   image_shape: Tuple[int, int, int],
                   augment_fn: Optional[Callable],
                   parse_fn: Optional[Callable] = record_parse):
        """
        Create a DataSet instance from a list of TFRecord files.
        
        Args:
            filenames (List[str]): List of filenames or patterns matching TFRecord files.
            image_shape (Tuple[int, int, int]): Shape of the images in the TFRecords.
            augment_fn (Optional[Callable]): Function for data augmentation (if any).
            parse_fn (Optional[Callable]): Function for parsing the dataset examples.
            
        Returns:
            DataSet: An instance of the DataSet class.
        """
        filenames_in = filenames  # Store the original list of filenames
        # Resolve file patterns into a sorted list of files
        filenames = sorted(sum([tf.io.gfile.glob(x) for x in filenames], []))
        if not filenames:  # Raise an error if no files were found
            raise ValueError('Empty dataset, files not found:', filenames_in)
        # Create a DataSet instance from the TFRecord files
        return cls(tf.data.TFRecordDataset(filenames), image_shape, augment_fn=augment_fn, parse_fn=parse_fn)

    @classmethod
    def from_tfds(cls, dataset: tf.data.Dataset, image_shape: Tuple[int, int, int],
                  augment_fn: Optional[Callable] = None):
        """
        Create a DataSet instance from a TensorFlow Dataset object (e.g., from TensorFlow Datasets).
        
        Args:
            dataset (tf.data.Dataset): TensorFlow Dataset object.
            image_shape (Tuple[int, int, int]): Shape of the images in the dataset.
            augment_fn (Optional[Callable]): Function for data augmentation (if any).
            
        Returns:
            DataSet: An instance of the DataSet class.
        """
        # Map the dataset to normalize the images and initialize a DataSet instance
        return cls(dataset.map(lambda x: dict(image=tf.cast(x['image'], tf.float32) / 127.5 - 1, label=x['label'])),
                   image_shape, augment_fn=augment_fn, parse_fn=None)

    def __iter__(self):
        """
        Allow the DataSet object to be iterated over.
        
        Returns:
            Iterator for the data.
        """
        return iter(self.data)

    def __getattr__(self, item):
        """
        Custom attribute access for DataSet class.
        Allows calling TensorFlow Dataset methods directly on the DataSet instance.
        
        Args:
            item: Name of the attribute or method to access.
            
        Returns:
            Callable that updates the dataset in place if it returns a Dataset object.
        """
        if item in self.__dict__:
            return self.__dict__[item]  # Return the item if it is already an attribute

        def call_and_update(*args, **kwargs):
            # Call the corresponding method on the tf.data.Dataset
            v = getattr(self.__dict__['data'], item)(*args, **kwargs)
            if isinstance(v, tf.data.Dataset):  # If the method returns a Dataset object
                # Return a new DataSet instance with the updated dataset
                return self.__class__(v, self.image_shape, augment_fn=self.augment_fn, parse_fn=self.parse_fn)
            return v  # Otherwise, return the result of the method call

        return call_and_update

    def augment(self, para_augment: int = 4):
        """
        Apply the augmentation function to the dataset if it exists.
        
        Args:
            para_augment (int): Number of parallel calls to use for augmentation.
            
        Returns:
            DataSet: The augmented DataSet instance.
        """
        if self.augment_fn:  # If an augmentation function is defined
            return self.map(self.augment_fn, para_augment)  # Apply it using parallel mapping
        return self  # Return the original DataSet if no augmentation function is defined

    def nchw(self):
        """
        Convert the image format from NHWC (default) to NCHW (commonly used in deep learning frameworks).
        
        Returns:
            DataSet: The DataSet instance with transposed images.
        """
        return self.map(lambda x: dict(image=tf.transpose(x['image'], [0, 3, 1, 2]), label=x['label']))

    def one_hot(self, nclass: int):
        """
        Convert labels to one-hot encoding.
        
        Args:
            nclass (int): Number of classes for one-hot encoding.
            
        Returns:
            DataSet: The DataSet instance with one-hot encoded labels.
        """
        return self.map(lambda x: dict(image=x['image'], label=tf.one_hot(x['label'], nclass)))

    def parse(self, para_parse: int = 2):
        """
        Apply the parsing function to the dataset.
        
        Args:
            para_parse (int): Number of parallel calls to use for parsing.
            
        Returns:
            DataSet: The parsed DataSet instance.
        """
        if not self.parse_fn:  # If no parsing function is defined, return the original DataSet
            return self
        if self.image_shape:  # If image_shape is defined
            # Apply the parsing function with the specified image shape
            return self.map(lambda x: self.parse_fn(x, self.image_shape), para_parse)
        # Apply the parsing function without image shape
        return self.map(self.parse_fn, para_parse)
```

### Summary of Code Functionality:
'''
- **`record_parse` function:** Parses a serialized TFRecord example into a dictionary containing a normalized 
image and its corresponding label.
- **`DataSet` class:** A wrapper around `tf.data.Dataset` that 
extends its functionality. It includes methods for parsing, augmenting, transposing, and one-hot encoding data, and provides a flexible way to load datasets from various sources like arrays, files, or TensorFlow Datasets.
- **Key Methods in `DataSet`:**
  - `from_arrays`: Initializes a `DataSet` from NumPy arrays.
  - `from_files`: Initializes a `DataSet` from TFRecord files.
  - `from_tfds`: Initializes a `DataSet` from a 
  TensorFlow Dataset object

.
  - `augment`: Applies augmentation to the dataset if 
  an augmentation function is provided.
  - `nchw`: Transforms the dataset images from NHWC 
  format to NCHW format.
  - `one_hot`: Converts labels to one-hot encoding.
  - `parse`: Applies the parsing function to the dataset.

This class encapsulates a lot of functionality, making it easy to preprocess and manipulate datasets for 
machine learning tasks in TensorFlow.