import pickle
import matplotlib.pyplot as plt
import os

def plot_ablation_study():
    print("------------------------------------------------")
    print(" GENERATING ABLATION STUDY PLOTS")
    print("------------------------------------------------")

    # 1. Load the history file
    file_path = 'results/experiment_history.pkl'
    if not os.path.exists(file_path):
        print(" Error: No history file found. Run experiment first.")
        return

    with open(file_path, 'rb') as f:
        history_storage = pickle.load(f)

    # 2. Setup the Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for our contenders
    colors = {'shallow': 'red', 'medium': 'blue', 'deep': 'green'}
    styles = {'shallow': '--', 'medium': '-', 'deep': '-.'}

    # 3. Plot Accuracy
    ax1.set_title('Validation Accuracy Comparison')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)

    # 4. Plot Loss
    ax2.set_title('Validation Loss Comparison')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)

    # Loop through models and plot
    for name, history in history_storage.items():
        epochs = range(1, len(history['val_accuracy']) + 1)
        
        # Accuracy Line
        ax1.plot(epochs, history['val_accuracy'], 
                 label=f'{name.capitalize()} Model', 
                 color=colors[name], 
                 linestyle=styles[name], linewidth=2)
        
        # Loss Line
        ax2.plot(epochs, history['val_loss'], 
                 label=f'{name.capitalize()} Model', 
                 color=colors[name], 
                 linestyle=styles[name], linewidth=2)

    ax1.legend()
    ax2.legend()

    # Save the plot
    os.makedirs("results", exist_ok=True)
    save_path = "results/architecture_comparison.png"
    plt.savefig(save_path)
    print(f" Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_ablation_study()