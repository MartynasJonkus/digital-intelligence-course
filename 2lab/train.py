import random
import numpy as np

def sigmoid_activation(a):
    return 1 / (1 + np.exp(-a))

def stochastic_gradient(loss_type, data_rows, weights, learning_rate, epoch_count, target_error_value):
    error_history = []
    accuracy_history = []
    total_error = target_error_value + 1
    epoch = 0
    
    while (total_error > target_error_value) and (epoch < epoch_count):
        random.shuffle(data_rows)
        total_error = 0
        total_correct = 0
        
        for row in data_rows:
            # Prepare features and label
            features = [1] + row[:-1]  # Add bias term
            class_label = row[-1]
            
            # Forward pass
            a = sum(x * w for x, w in zip(features, weights))
            y = sigmoid_activation(a)

            total_correct += 1 if (y >= 0.5) == class_label else 0
            
            if loss_type == 'cross_entropy':
                for i, x in enumerate(features):
                    weights[i] -= learning_rate * (y - class_label) * x

                total_error += -np.log(y + 1e-8) if class_label == 1 else -np.log(1 - y + 1e-8)

            elif loss_type == 'mean_squared':
                for i, x in enumerate(features):
                    weights[i] -= learning_rate * (y - class_label) * y * (1 - y) * x

                total_error += 0.5 * (class_label - y) ** 2
            
        # Store stats
        total_error /= len(data_rows)
        accuracy = total_correct / len(data_rows)
        error_history.append(total_error)
        accuracy_history.append(accuracy)

        epoch += 1
        print(f"Epoch {epoch}: Loss={total_error:.4f}, Acc={accuracy:.4f}")
    
    return weights, error_history, accuracy_history

def batch_gradient(loss_type, data_rows, weights, learning_rate, epoch_count, target_error_value):
    error_history = []
    accuracy_history = []
    total_error = target_error_value + 1
    epoch = 0

    while (total_error > target_error_value) and (epoch < epoch_count):
        random.shuffle(data_rows)
        total_error = 0.0
        total_correct = 0
        gradients = [0.0] * len(weights)

        for row in data_rows:
            # Separate features and class label
            features = [1] + row[:-1] # Add bias
            class_label = row[-1]

            # Calculate activation value and output
            a = sum(x * w for x, w in zip(features, weights))
            y = sigmoid_activation(a)

            total_correct += 1 if (y >= 0.5) == class_label else 0

            if loss_type == 'cross_entropy':
                for i, x in enumerate(features):
                    gradients[i] += (y - class_label) * x

                total_error += -np.log(y + 1e-8) if class_label == 1 else -np.log(1 - y + 1e-8)

            elif loss_type == 'mean_squared':
                for i, x in enumerate(features):
                    gradients[i] += (y - class_label) * y * (1 - y) * x

                total_error += 0.5 * (class_label - y) ** 2    
        
        # Update weights
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradients[i] / len(data_rows)

        # Store stats
        total_error /= len(data_rows)
        accuracy = total_correct / len(data_rows)
        error_history.append(total_error)
        accuracy_history.append(accuracy)

        epoch += 1
        print(f"Epoch {epoch}: Loss={total_error:.4f}, Acc={accuracy:.4f}")

    return weights, error_history, accuracy_history