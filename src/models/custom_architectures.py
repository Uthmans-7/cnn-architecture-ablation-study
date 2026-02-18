import tensorflow as tf
from tensorflow.keras import layers, models

def build_experimental_cnn(model_type, input_shape=(64, 64, 3), num_classes=6, kernel_size=(3, 3)):
    """
    Factory function for ablation studies (Shallow vs Deep, Kernel Sizes).
    Use this for EXPERIMENTS only.
    """
    model = models.Sequential()
    
    # =========================================================
    # 1. SHALLOW MODEL (The Baseline)
    # =========================================================
    if model_type == 'shallow':
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))

    # =========================================================
    # 2. MEDIUM MODEL (The Variable One)
    # Allows changing kernel_size for the "Eye Exam" experiment
    # =========================================================
    elif model_type == 'medium':
        model.add(layers.Input(shape=input_shape))
        
        # Block 1
        model.add(layers.Conv2D(32, kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Block 2
        model.add(layers.Conv2D(64, kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Block 3
        model.add(layers.Conv2D(128, kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Classifier
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))

    # =========================================================
    # 3. DEEP MODEL (The Complex One)
    # =========================================================
    elif model_type == 'deep':
        model.add(layers.Input(shape=input_shape))
        
        # Block 1
        model.add(layers.Conv2D(32, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Block 2
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Block 3
        model.add(layers.Conv2D(128, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Block 4
        model.add(layers.Conv2D(256, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Block 5
        model.add(layers.Conv2D(512, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.6))
        model.add(layers.Dense(num_classes, activation='softmax'))

    return model