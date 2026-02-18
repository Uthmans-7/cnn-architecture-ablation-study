import tensorflow as tf
from tensorflow.keras import layers, models

def build_production_model(input_shape=(64, 64, 3), num_classes=6):
    """
    The FINAL SELECTED MODEL.
    Architecture: Medium Depth (3 Blocks) with 3x3 Kernels.
    Performance: ~82% Validation Accuracy on Intel Dataset.
    """
    model = models.Sequential(name="Intel_Classifier_V1")
    
    model.add(layers.Input(shape=input_shape))
    
    # ---------------------------------------------------------
    # Feature Extraction (The "Eyes")
    # ---------------------------------------------------------
    # Block 1: Capture fine textures (leaves, ice cracks)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 2: Capture shapes (windows, tree trunks)
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 3: Capture complex objects (entire buildings, mountains)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # ---------------------------------------------------------
    # Classification (The "Brain")
    # ---------------------------------------------------------
    model.add(layers.Flatten())
    
    # Dense layer for decision making
    model.add(layers.Dense(128, activation='relu'))
    
    # Dropout to prevent memorization (Regularization)
    model.add(layers.Dropout(0.5))
    
    # Output layer (6 probabilities)
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model