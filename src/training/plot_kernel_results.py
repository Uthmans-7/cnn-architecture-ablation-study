import pickle
import matplotlib.pyplot as plt
import os

def plot_kernel_study():
    print("------------------------------------------------")
    print(" GENERATING KERNEL STUDY PLOTS")
    print("------------------------------------------------")

    file_path = 'results/kernel_history.pkl'
    if not os.path.exists(file_path):
        print(" Error: No history file found.")
        return

    with open(file_path, 'rb') as f:
        history_storage = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors: 3x3 (Green/Good), 5x5 (Blue/Neutral), 7x7 (Red/Bad)
    colors = {'kernel_3x3': 'green', 'kernel_5x5': 'blue', 'kernel_7x7': 'red'}
    
    ax1.set_title('Validation Accuracy (Kernel Size)')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Validation Loss (Kernel Size)')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.grid(True, alpha=0.3)

    for name, history in history_storage.items():
        epochs = range(1, len(history['val_accuracy']) + 1)
        ax1.plot(epochs, history['val_accuracy'], label=name, color=colors.get(name, 'black'), linewidth=2)
        ax2.plot(epochs, history['val_loss'], label=name, color=colors.get(name, 'black'), linewidth=2)

    ax1.legend()
    ax2.legend()
    
    plt.savefig("results/kernel_comparison.png")
    print(" Plot saved to results/kernel_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_kernel_study()