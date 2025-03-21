import matplotlib.pyplot as plt

def plot_training_history(error_history, accuracy_history, loss_type="Cross-Entropy"):
    plt.figure(figsize=(12, 5))
    
    # Plot error
    plt.subplot(1, 2, 1)
    plt.plot(error_history, 'r-', linewidth=2)
    plt.title(f'{loss_type} Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, 'b-', linewidth=2)
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()