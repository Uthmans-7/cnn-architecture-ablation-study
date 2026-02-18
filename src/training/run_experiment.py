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

def run_ablation_study():
    print("------------------------------------------------")
    print(" STARTING ARCHITECTURE ABLATION STUDY")
    print("------------------------------------------------")

    # 1. SETUP DATA
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = os.path.join(DATA_DIR, 'seg_train', 'seg_train')
    test_dir = os.path.join(DATA_DIR, 'seg_test', 'seg_test')
    
    if not os.path.exists(train_dir):
        train_dir = os.path.join(DATA_DIR, 'seg_train')
        test_dir = os.path.join(DATA_DIR, 'seg_test')

    print(f" Loading Data from: {train_dir}")

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse'
    )

    # 2. EXPERIMENT LOOP
    model_names = ['shallow', 'medium', 'deep']
    history_storage = {}

    for name in model_names:
        print(f"\n\n TRAINING MODEL: {name.upper()}")
        
        #  UPDATED FUNCTION CALL
        model = build_experimental_cnn(name, input_shape=IMG_SIZE + (3,), num_classes=6)
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=3)

        history = model.fit(train_gen, epochs=EPOCHS, validation_data=test_gen, callbacks=[early_stop])
        
        history_storage[name] = history.history
        
        # Ensure ml_models directory exists
        os.makedirs("ml_models", exist_ok=True)
        model.save(f"ml_models/intel_{name}.keras")

    # 3. SAVE RESULTS
    os.makedirs("results", exist_ok=True)
    with open('results/experiment_history.pkl', 'wb') as f:
        pickle.dump(history_storage, f)
        
    print("\n EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    run_ablation_study()