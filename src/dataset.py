# src/dataset.py
import tensorflow as tf
from src.config import TRAIN_PATH, TEST_PATH, IMAGE_SIZE, BATCH_SIZE, SEED

def get_data_loaders():
    """
    Loads the Intel Image Classification dataset from directory paths 
    defined in config.py and converts them into optimized tf.data.Dataset objects.
    """
    print("Initializing Data Pipelines...")
    
    # 1. Load the training data split (creating a validation split from it programmatically)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"  # Multi-class classification optimization
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    
    # 2. Load the dedicated test data split
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    
    # 3. Optimize the performance of the datasets using the tf.data API
    # AUTOTUNE dynamically adjusts buffer sizes based on your hardware's CPU/GPU footprint
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds