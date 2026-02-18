import os
import zipfile
from pathlib import Path

# Setup Kaggle config (Point to current directory for kaggle.json)
os.environ['KAGGLE_CONFIG_DIR'] = "."

def download_intel_data():
    # 1. Setup Paths
    # We use Pathlib to handle Windows paths ("\") automatically
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)

    print("------------------------------------------------")
    print("⬇ STARTING DOWNLOAD: Intel Image Classification")
    print("------------------------------------------------")
    
    # 2. Download via Kaggle API
    # dataset: puneet6060/intel-image-classification
    os.system('kaggle datasets download -d puneet6060/intel-image-classification')

    # 3. Check if download worked
    zip_file = Path("intel-image-classification.zip")
    if not zip_file.exists():
        print(" ERROR: Download failed. Check your kaggle.json file.")
        return

    print(" Unzipping data (this might take a moment)...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(raw_data_path)
    
    # 4. Clean up
    os.remove(zip_file)
    print(f"✅ SUCCESS: Data extracted to {raw_data_path.absolute()}")
    print("   Check for folders: 'seg_train' and 'seg_test'")

if __name__ == "__main__":
    download_intel_data()