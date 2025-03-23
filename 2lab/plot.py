import matplotlib.pyplot as plt

def plot_training_validation(train_error, train_acc, val_error, val_acc):
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_error, 'r-', linewidth=2, label='Training')
    plt.plot(val_error, 'm--', linewidth=2, label='Validation')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, 'b-', linewidth=2, label='Training')
    plt.plot(val_acc, 'c--', linewidth=2, label='Validation')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()