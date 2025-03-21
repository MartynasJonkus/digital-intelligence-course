import numpy as np
from prepare_data import prepare_data
from train import stochastic_gradient, batch_gradient
from plots import plot_training_history


data_rows = prepare_data('breast-cancer-wisconsin.data')
weights = np.random.normal(-1, 1, len(data_rows[0]))

trained_weights, error_history, accuracy_history = stochastic_gradient(
    loss_type='cross_entropy',
    data_rows=data_rows,
    weights=weights,
    learning_rate=0.001,
    epoch_count=1000,
    target_error_value=0.05
)

plot_training_history(error_history, accuracy_history, "Cross-Entropy")