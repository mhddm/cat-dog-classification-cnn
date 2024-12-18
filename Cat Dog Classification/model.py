# Import necessary libraries
import numpy as np
import tensorflow as tf
from config import Config
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling


def load_data(data_dir, img_height = 256, img_width = 256, batch_size = 32):
    """
    Prepares the training and validation datasets from the provided directory.

    Args:
        data_dir (str): Path to the dataset directory.
        img_height (int, optional): Height of the images. Defaults to 256.
        img_width (int, optional): Width of the images. Defaults to 256.
        batch_size (int, optional): Number of images per batch. Defaults to 32.
    
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
    AUTOTUNE =  tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE).cache()
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE).cache()
    
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
    
    model = tf.keras.Sequential([
        Rescaling(1./255, input_shape = (img_height, img_width, 3)),
        Conv2D(32, 3, activation = 'relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation = 'relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation = 'relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation = 'relu'),
        Dense(num_classes)                  
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
        # Load configuration
        data_dir = Config.RAW_DATA_DIR
        save_dir = Config.SAVE_DIR
        
        # Configuration parameters
        batch_size = 32
        img_height = 180
        img_width = 180 
    
        # Prepare datasets
        train_ds, val_ds = load_data(data_dir, img_height, img_width, batch_size)
        
        # Build model
        num_classes = 2
        model = build_model(img_height, img_width, num_classes)
        
        # Train model
        history = train_model(model, train_ds, val_ds, epochs = 3)
        
        # Save the trained model
        save_model(model, save_dir)
        
    except Exception as e:
        print(f"Error: {e}")

