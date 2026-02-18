

```markdown
#  Intel Image Classification: A CNN Architecture Ablation Study

A systematic deep learning research project investigating the impact of **Network Depth** and **Receptive Field Size (Kernel)** on the classification of natural scenes. This project moves beyond "just training a model" to empirically validate architectural decisions using the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

---

##  Project Goals & Hypothesis
1.  **Depth Analysis (The "Layer" Question):** Does adding more layers always improve accuracy? We hypothesized that a "Medium" depth (3 blocks) would outperform both "Shallow" (underfitting) and "Deep" (unstable/overfitting) models on this specific dataset size (~14k images).
2.  **Receptive Field Study (The "Eye" Question):** How does the size of the convolution kernel affect feature extraction? We compared **3x3** (texture-focused) vs. **7x7** (shape-focused) kernels to see which is better for classifying texture-heavy classes like Forests and Glaciers.
3.  **Mathematical Validation:** Manually implemented a 2D Convolution layer in pure NumPy to demonstrate a foundational understanding of feature extraction before using TensorFlow/Keras.

---

##  Key Findings

### Experiment 1: Architecture Depth
| Model Type | Layers | Training Accuracy | Validation Accuracy | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Shallow** | 1 Conv Block | ~90% | ~74% | **Underfitting:** High bias, failed to generalize. |
| **Deep** | 5 Conv Blocks | ~85% | ~55-78% (Unstable) | **Unstable:** Struggled to converge without massive data. |
| **Medium** | 3 Conv Blocks | ~88% | **~82% (Winner)** | **Optimal:** Best balance of learning capacity and stability. |

### Experiment 2: Kernel Size (Receptive Field)
| Kernel Size | Focus | Speed (sec/epoch) | Accuracy | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **3x3** | Fine Texture | **~46s**  | **~82.3%**  | Captured critical textures (leaves, ice). |
| **5x5** | Balanced | ~78s | ~81.4% | Good, but slower without accuracy gain. |
| **7x7** | Broad Shape | ~140s  | ~77.8% | **Failed:** Blurred out fine details and was 3x slower. |

---

##  Project Structure

```text
intel_cnn_study/
├── data/                      # (Ignored via .gitignore) Raw Intel dataset
├── src/                 
│   ├── manual_math/           #  NumPy implementation of Conv2D
│   │   └── conv_layer.py      # The mathematical proof of convolution
│   ├── models/          
│   │   ├── custom_architectures.py  # Factory for experimental models (Shallow/Medium/Deep)
│   │   └── final_model.py           #  The CLEAN production-ready model (Medium + 3x3)
│   ├── training/              # Experiment runners
│   │   ├── run_experiment.py        # Depth Study Runner
│   │   ├── run_kernel_experiment.py # Kernel Size Study Runner
│   │   ├── plot_results.py          # Visualization tools
│   │   └── plot_kernel_results.py   # Visualization tools
│   └── download_data.py       # Automated Kaggle downloader
├── ml_models/                 # Saved .keras files from experiments
├── results/                   # Generated plots (.png) and history logs (.pkl)
└── requirements.txt           # Project dependencies

```

---

##  Setup & Usage

### 1. Environment Initialization

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

```

### 2. Data Acquisition

Automatically download and unzip the 350MB dataset from Kaggle:

```bash
python -m src.download_data

```

### 3. Verify the Math (Optional)

Run the manual NumPy convolution script to see the pixel-level math of edge detection:

```bash
python -m src.manual_math.conv_layer

```

---

##  Reproducing the Experiments

### Phase 1: The Depth Study

Train 3 models (Shallow, Medium, Deep) and save their training history.

```bash
python -m src.training.run_experiment

```

**Visualize Results:**

```bash
python -m src.training.plot_results
# Check results/architecture_comparison.png

```

### Phase 2: The Kernel Study

Train the "Medium" model with 3 different kernel sizes (3x3, 5x5, 7x7).

```bash
python -m src.training.run_kernel_experiment

```

**Visualize Results:**

```bash
python -m src.training.plot_kernel_results
# Check results/kernel_comparison.png

```

---

##  Production Usage

For deployment, use the `final_model.py` which contains the optimized architecture (Medium Depth + 3x3 Kernels).

```python
from src.models.final_model import build_production_model

# Build the optimized model
model = build_production_model(input_shape=(64, 64, 3), num_classes=6)
model.summary()

```

---

##  Future Work

* **Transfer Learning:** Compare these custom CNNs against a pre-trained **ResNet50** to benchmark performance.
* **Data Augmentation:** Implement rotation and flipping to improve the "Deep" model's stability.
* **Grad-CAM:** Visualize exactly which pixels (leaves vs. sky) the 3x3 kernel is focusing on.

```

```