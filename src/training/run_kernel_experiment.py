import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
#  UPDATED IMPORT
from src.models.custom_architectures import build_experimental_cnn

#  CONFIGURATION
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "data/raw"

def run_kernel_study():
    print("------------------------------------------------")
    print(" STARTING KERNEL SIZE STUDY (3x3 vs 5x5 vs 7x7)")
    print("------------------------------------------------")

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_dir = os.path.join(DATA_DIR, 'seg_train', 'seg_train')
    test_dir = os.path.join(DATA_DIR, 'seg_test', 'seg_test')
    if not os.path.exists(train_dir):
        train_dir = os.path.join(DATA_DIR, 'seg_train')
        test_dir = os.path.join(DATA_DIR, 'seg_test')

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
    )

    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    history_storage = {}

    for k_size in kernel_sizes:
        name = f"kernel_{k_size[0]}x{k_size[1]}"
        print(f"\n\n TRAINING WITH KERNEL SIZE: {k_size}")
        
        #  UPDATED FUNCTION CALL
        # Note: We hardcode 'medium' here because that's the base model for this study
        model = build_experimental_cnn('medium', input_shape=IMG_SIZE + (3,), num_classes=6, kernel_size=k_size)
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=3)

        history = model.fit(train_gen, epochs=EPOCHS, validation_data=test_gen, callbacks=[early_stop])
        
        history_storage[name] = history.history
        os.makedirs("ml_models", exist_ok=True)
        model.save(f"ml_models/intel_{name}.keras")

    os.makedirs("results", exist_ok=True)
    with open('results/kernel_history.pkl', 'wb') as f:
        pickle.dump(history_storage, f)
        
    print("\n KERNEL EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    run_kernel_study()