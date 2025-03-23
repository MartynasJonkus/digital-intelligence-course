import numpy as np
import matplotlib.pyplot as plt
from prepare_data import prepare_data
from train import stochastic_gradient, batch_gradient
from tqdm import tqdm

def learning_rate_experiment(data_train, data_val, learning_rates, is_stochastic=True):
    results = {
        'lr': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'histories': []
    }
    
    for lr in tqdm(learning_rates, desc="Testing LRs"):
        if is_stochastic:
            weights, epoch, time_to_train, error_history, accuracy_history, val_error_history, val_accuracy_history = stochastic_gradient(
                training_data=data_train,
                learning_rate=lr,
                epoch_count=1000,
                target_error_value=0.1,
                validation_data=data_val
            )
        else:
            weights, epoch, time_to_train, error_history, accuracy_history, val_error_history, val_accuracy_history = batch_gradient(
                training_data=data_train,
                learning_rate=lr,
                epoch_count=1000,
                target_error_value=0.1,
                validation_data=data_val
            )
        
        results['lr'].append(lr)
        results['train_loss'].append(error_history[-1])
        results['val_loss'].append(val_error_history[-1])
        results['train_acc'].append(accuracy_history[-1])
        results['val_acc'].append(val_accuracy_history[-1])
        results['histories'].append((error_history, val_error_history, accuracy_history, val_accuracy_history))
    
    return results

def plot_lr_impact(results):
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.semilogx(results['lr'], results['train_loss'], 'r-o', label='Training error')
    plt.semilogx(results['lr'], results['val_loss'], 'm--s', label='Validation error')
    
    lrs_to_annotate = [0.2, 0.5]
    for lr in lrs_to_annotate:
        if lr in results['lr']:
            idx = results['lr'].index(lr)
            plt.annotate(f'{lr}', 
                        (results['lr'][idx], results['train_loss'][idx]),
                        textcoords="offset points",
                        xytext=(5,5),
                        ha='left',
                        fontsize=9,
                        arrowprops=dict(arrowstyle="->", lw=0.5))


    plt.title('Final error vs learning rate')
    plt.xlabel('Learning rate (log scale)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    for i, lr in enumerate(results['lr']):
        plt.plot(results['histories'][i][0], 
                label=f'LR={lr:.4f}', 
                alpha=0.7,
                linewidth=1.5)
    plt.title('Training error over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 2.2)
    plt.subplot(1, 3, 3)
    plt.semilogx(results['lr'], results['train_acc'], 'b-o', label='Training accuracy')
    plt.semilogx(results['lr'], results['val_acc'], 'c--s', label='Validation accuracy')

    for lr in lrs_to_annotate:
        if lr in results['lr']:
            idx = results['lr'].index(lr)
            plt.annotate(f'{lr}', 
                        (results['lr'][idx], results['train_acc'][idx]),
                        textcoords="offset points",
                        xytext=(5,5),
                        ha='left',
                        fontsize=9,
                        arrowprops=dict(arrowstyle="->", lw=0.5))

    plt.title('Final accuracy vs learning rate')
    plt.xlabel('Learning rate (log scale)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


data_rows = prepare_data('breast-cancer-wisconsin.data')

training_data = data_rows[:round(len(data_rows) * 0.8)]
validation_data = data_rows[round(len(data_rows) * 0.8):round(len(data_rows) * 0.9)]
testing_data = data_rows[round(len(data_rows) * 0.9):]

# Example usage
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0]

# For Stochastic GD
stochastic_results = learning_rate_experiment(
    training_data, 
    validation_data,
    learning_rates,
    is_stochastic=True
)
plot_lr_impact(stochastic_results)

# For Batch GD
batch_results = learning_rate_experiment(
    training_data,
    validation_data,
    learning_rates,
    is_stochastic=False
)
plot_lr_impact(batch_results)