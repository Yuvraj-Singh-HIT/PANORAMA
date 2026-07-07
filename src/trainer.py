# src/trainer.py
import tensorflow as tf
from src.config import CHECKPOINT_DIR

def compile_and_get_callbacks(model, learning_rate=0.001):
    """
    Compiles the Keras model with structural loss/optimizers 
    and sets up automated monitoring callbacks.
    """
    # 1. Compile the model
    # Using CategoricalCrossentropy because our data pipeline uses label_mode="categorical"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # 2. Setup Checkpointing Callback
    # This automatically tracks validation loss and saves only the best model iteration
    checkpoint_path = CHECKPOINT_DIR / "best_baseline_model.keras"
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    print(f"Model compiled. Checkpoints will be saved to: {checkpoint_path}")
    return [checkpoint_callback]