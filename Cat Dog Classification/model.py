# Import necessary libraries
import numpy as np
import tensorflow as tf
from config import Config

from tensorflow import keras
from keras import layers
from keras.models import Sequential


def load_data(data_dir, img_height, img_width, batch_size = 32):
    """
    Prepares the training and validation datasets from the provided directory.

    Args:
        data_dir (str): Path to the dataset directory.
        img_height (int): Height of the images.
        img_width (int): Width of the images.
        batch_size (int): Number of images per batch. Defaults to 32.
    
    Returns:
        tuple: Prepared training and validation datasets
    """
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = 0.2,
        subset = "training",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = 0.2,
        subset="validation",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )
    
    # Configure datasets for performance
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE) # type: ignore
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE) # type: ignore
    
    return train_ds, val_ds


def build_model(img_height, img_width, num_classes):
    
    """
    Builds a sequential CNN model for image classification.

    Args:
        img_height (int, optional): Height of the images. Defaults to 256.
        img_width (int, optional): Width of the images. Defaults to 256.
        num_classes (int): Number of output classes.
        
    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    
    model = Sequential([
        layers.Rescaling(1./255, input_shape = (img_height, img_width, 3)),
        layers.Conv2D(16, 3, activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(num_classes)           
    ])
    
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy']
    )
    
    return model
    
    
def train_model(model, train_ds, val_ds, epochs):
    """
    Trains the model on the provided dataset.

    Args:
        model (tf.keras.Model): Compiled CNN model.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    
    return model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = epochs
    )
    
def save_model(model, save_path):
    """
    Saves the trained model to the specified path.
    
    Args:
        model (tf.keras.Model): Trained model.
        save_path (str): Path to save the model.
    """
    
    model.save(save_path)
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    try:
        # Load configuration parameters
        data_dir = Config.RAW_DATA_DIR
        save_dir = Config.SAVE_DIR
        batch_size = Config.BATCH_SIZE
        img_height = Config.IMG_HEIGHT
        img_width = Config.IMG_WIDTH
        epochs = Config.EPOCHS 
    
        # Prepare datasets
        train_ds, val_ds = load_data(data_dir, img_height, img_width, batch_size)
        
        # Build model
        num_classes = 2
        model = build_model(img_height, img_width, num_classes)
        
        # Train model
        history = train_model(model, train_ds, val_ds, epochs)
        
        # Save the trained model
        save_model(model, save_dir)
        
    except Exception as e:
        print(f"Error: {e}")

