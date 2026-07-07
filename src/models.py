import tensorflow as tf
from tensorflow.keras import layers, models
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def build_baseline_cnn(input_shape=(150, 150, 3), num_classes=6):
    """
    Builds a baseline Convolutional Neural Network using tf.keras.
    """
    model = models.Sequential([
        # Standardize pixel values to [0, 1]
        layers.Rescaling(1./255, input_shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flattening and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
# src/models.py (Append this to the bottom of the file)

def build_transfer_learning_model():
    """
    Loads a pre-trained MobileNetV2 base model, freezes its layers,
    and appends a trainable Classification Head optimized for the Intel dataset.
    """
    print("Loading pre-trained MobileNetV2 Base...")
    # 1. Instantiate the pre-trained base model
    # include_top=False drops the original ImageNet 1000-class output layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(150, 150, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 2. FREEZE the base layers so we don't destroy their pre-trained features
    base_model.trainable = False
    
    # 3. Chain the layers together using the Keras Functional API
    inputs = tf.keras.Input(shape=(150, 150, 3))
    
    # MobileNetV2 expects inputs in a specific scale (-1 to 1). 
    # This built-in layer handles that automatically!
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Pass inputs through the frozen base
    x = base_model(x, training=False)
    
    # Global Average Pooling collapses spatial dimensions (Height x Width) into a 1D vector
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add a Dropout layer to structurally mitigate overfitting
    x = layers.Dropout(0.2)(x)
    
    # Final classification output layer (6 classes)
    outputs = layers.Dense(6, activation='softmax')(x)
    
    # Build the final combined model
    model = tf.keras.Model(inputs, outputs, name="Transfer_MobileNetV2")
    
    return model