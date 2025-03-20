import random
import numpy as np
from prepare_data import prepare_data

def sigmoid_activation(a):
    return 1 / (1 + np.exp(-a))

def stochastic_gradient_train(data_rows, weights, learning_rate, epoch_count, target_loss_value):
    random.shuffle(data_rows)
    total_loss = target_loss_value + 1
    epoch = 0
    while (total_loss > target_loss_value) and (epoch < epoch_count):
        total_loss = 0
        total_correct_predictions = 0
        for row in data_rows:
            # Separate features and class label
            features = row[:-1]
            features = [1] + features  # Add bias
            class_label = row[-1]

            # Calculate activation value and output
            a = sum(x * w for x, w in zip(features, weights))
            y = sigmoid_activation(a)

            # Check if prediction is correct
            predicted_class = 1 if y >= 0.5 else 0
            if predicted_class == class_label:
                total_correct_predictions += 1

            # Update weights
            for i, x in enumerate(features):
                weights[i] -= learning_rate * (y - class_label) * y * (1 - y) * x

            total_loss += (class_label - y) ** 2

        print(f"Epoch {epoch}: Total loss = {total_loss}")
        print(f"Accuracy: {total_correct_predictions / len(data_rows)}")
        print(f"Weights: {weights[1:]}, bias: {weights[0]}")
        epoch += 1
    return weights

def batch_gradient_train(data_rows, weights, learning_rate, epoch_count, target_loss_value):
    random.shuffle(data_rows)
    total_loss = target_loss_value + 1
    epoch = 0
    while (total_loss > target_loss_value) and (epoch < epoch_count):
        total_loss = 0
        total_correct_predictions = 0
        gradient_sum = [0] * len(weights)
        for row in data_rows:
            # Separate features and class label
            features = row[:-1]
            features = [1] + features  # Add bias
            class_label = row[-1]

            # Calculate activation value and output
            a = sum(x * w for x, w in zip(features, weights))
            y = sigmoid_activation(a)

            # Check if prediction is correct
            predicted_class = 1 if y >= 0.5 else 0
            if predicted_class == class_label:
                total_correct_predictions += 1

            # Update weights
            for i, x in enumerate(features):
                gradient_sum[i] += (y - class_label) * y * (1 - y) * x

            total_loss += (class_label - y) ** 2
        
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradient_sum[i] / len(data_rows)

        print(f"Epoch {epoch}: Total loss = {total_loss}")
        print(f"Accuracy: {total_correct_predictions / len(data_rows)}")
        print(f"Weights: {weights[1:]}, bias: {weights[0]}")
        epoch += 1
    return weights

def validate():
    # ...
    return

data_rows = prepare_data('breast-cancer-wisconsin.data')
