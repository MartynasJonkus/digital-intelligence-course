import numpy as np
from prepare_data import prepare_data
from train import stochastic_gradient, batch_gradient
from evaluate import evaluate_model
from plot import plot_training_history

data_rows = prepare_data('breast-cancer-wisconsin.data')

training_data = data_rows[:round(len(data_rows) * 0.8)]
validation_data = data_rows[round(len(data_rows) * 0.8):round(len(data_rows) * 0.9)]
testing_data = data_rows[round(len(data_rows) * 0.9):]

(
    stochastic_weights,
    stochastic_epoch_count,
    stochastic_time_to_train,
    train_stochastic_err,
    train_stochastic_acc,
    val_stochastic_err,
    val_stochastic_acc
) = stochastic_gradient(
    training_data=training_data,
    learning_rate=0.01,
    epoch_count=1000,
    target_error_value=0.09,
    validation_data=validation_data
)
test_stochastic_err, test_stochastic_acc = evaluate_model(testing_data, stochastic_weights)

(
    batch_weights,
    batch_epoch_count,
    batch_time_to_train,
    train_batch_err,
    train_batch_acc,
    val_batch_err,
    val_batch_acc
) = batch_gradient(
    training_data=training_data,
    learning_rate=0.2,
    epoch_count=1000,
    target_error_value=0.1,
    validation_data=validation_data
)
test_batch_err, test_batch_acc = evaluate_model(testing_data, batch_weights)

print("\nSTOCHASTIC GRADIENT DESCENT RESULTS")
print(f"Epochs to train: {stochastic_epoch_count}, time to train: {stochastic_time_to_train:.4f} seconds")
print(f"Final weights: {np.round(stochastic_weights[1:], 4)}, bias: {stochastic_weights[0]:.4f}")
print(f"Final training error value: {train_stochastic_err[-1]:.4f}, final accuracy: {train_stochastic_acc[-1]:.4f}")
print(f"Final validation error value: {val_stochastic_err[-1]:.4f}, final accuracy: {val_stochastic_acc[-1]:.4f}")
print(f"Final testing error value: {test_stochastic_err:.4f}, final accuracy: {test_stochastic_acc:.4f}")
plot_training_history(train_stochastic_err, train_stochastic_acc)

print("\nBATCH GRADIENT DESCENT RESULTS")
print(f"Epochs to train: {batch_epoch_count}, time to train: {batch_time_to_train:.4f} seconds")
print(f"Final weights: {np.round(batch_weights[1:], 4)}, bias: {batch_weights[0]:.4f}")
print(f"Final error value: {train_batch_err[-1]:.4f}, final accuracy: {train_batch_acc[-1]:.4f}")
print(f"Final validation error value: {val_batch_err[-1]:.4f}, final accuracy: {val_batch_acc[-1]:.4f}")
print(f"Final testing error value: {test_batch_err:.4f}, final accuracy: {test_batch_acc:.4f}")
plot_training_history(train_batch_err, train_batch_acc)