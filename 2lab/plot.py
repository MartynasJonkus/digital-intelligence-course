import matplotlib.pyplot as plt
from tabulate import tabulate

def plot_training_validation(train_error, train_acc, val_error, val_acc):
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_error, 'r-', linewidth=2, label='Training')
    plt.plot(val_error, 'm--', linewidth=2, label='Validation')
    plt.title('Training vs validation error')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, 'b-', linewidth=2, label='Training')
    plt.plot(val_acc, 'c--', linewidth=2, label='Validation')
    plt.title('Training vs validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_best_models(s_val_err, s_val_acc, b_val_err, b_val_acc):
    # Validation Data Comparison Table
    print("\nValidation data performance:")
    print(tabulate([
        ["Stochastic", s_val_err[-1], s_val_acc[-1]],
        ["Batch", b_val_err[-1], b_val_acc[-1]]
    ], headers=["Method", "Loss", "Accuracy"], floatfmt=".4f"))

    # Comparative Visualization
    plt.figure(figsize=(12, 5))
    
    # Error Comparison
    plt.subplot(1, 2, 1)
    plt.plot(s_val_err, 'm-', label='Stochastic')
    plt.plot(b_val_err, 'c-', label='Batch')
    plt.title('Validation error comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.ylim(0, 2.2)
    plt.grid(True, alpha=0.3)
    
    # Accuracy Comparison
    plt.subplot(1, 2, 2)
    plt.plot(s_val_acc, 'g-', label='Stochastic')
    plt.plot(b_val_acc, 'b-', label='Batch')
    plt.title('Validation accuracy comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()