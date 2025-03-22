import random
import time
import numpy as np
from evaluate import evaluate_model

def sigmoid_activation(a):
    return 1 / (1 + np.exp(-a))

def stochastic_gradient(training_data, learning_rate, epoch_count, target_error_value, validation_data = None):
    start_time = time.time()

    error_history = []
    accuracy_history = []
    val_error_history = []
    val_accuracy_history = []
    total_error = target_error_value + 1
    epoch = 0
    weights = np.random.normal(0, 0.1, len(training_data[0]))
    
    while (total_error > target_error_value) and (epoch < epoch_count):
        random.shuffle(training_data)
        total_error = 0.0
        total_correct = 0
        
        for row in training_data:
            features = [1] + row[:-1]
            class_label = row[-1]
            
            a = sum(x * w for x, w in zip(features, weights))
            y = sigmoid_activation(a)
            y = np.clip(y, 1e-8, 1 - 1e-8)

            total_correct += 1 if (y >= 0.5) == class_label else 0
            
            for i, x in enumerate(features):
                weights[i] -= learning_rate * (y - class_label) * x

            total_error += -np.log(y) if class_label == 1 else -np.log(1 - y)

        total_error /= len(training_data)
        accuracy = total_correct / len(training_data)
        error_history.append(total_error)
        accuracy_history.append(accuracy)

        epoch += 1
        print(f"Epoch {epoch}: Loss={total_error:.4f}, Acc={accuracy:.4f}")

        if validation_data is not None:
            val_error, val_acc = evaluate_model(validation_data, weights)
            val_error_history.append(val_error)
            val_accuracy_history.append(val_acc)
            print(f"Val Loss: {val_error:.4f}, Val Acc: {val_acc:.4f}")

    time_to_train = time.time() - start_time
    
    return weights, epoch, time_to_train, error_history, accuracy_history, val_error_history, val_accuracy_history


def batch_gradient(training_data, learning_rate, epoch_count, target_error_value, validation_data = None):
    start_time = time.time()

    error_history = []
    accuracy_history = []
    val_error_history = []
    val_accuracy_history = []
    total_error = target_error_value + 1
    epoch = 0
    weights = np.random.normal(0, 0.1, len(training_data[0]))

    while (total_error > target_error_value) and (epoch < epoch_count):
        total_error = 0.0
        total_correct = 0
        gradients = [0.0] * len(weights)

        for row in training_data:
            features = [1] + row[:-1]
            class_label = row[-1]

            a = sum(x * w for x, w in zip(features, weights))
            y = sigmoid_activation(a)
            y = np.clip(y, 1e-8, 1 - 1e-8)

            total_correct += 1 if (y >= 0.5) == class_label else 0

            for i, x in enumerate(features):
                gradients[i] += (y - class_label) * x

            total_error += -np.log(y) if class_label == 1 else -np.log(1 - y)
        
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradients[i] / len(training_data)

        total_error /= len(training_data)
        accuracy = total_correct / len(training_data)
        error_history.append(total_error)
        accuracy_history.append(accuracy)

        epoch += 1
        print(f"Epoch {epoch}: Loss={total_error:.4f}, Acc={accuracy:.4f}")

        if validation_data is not None:
            val_error, val_acc = evaluate_model(validation_data, weights)
            val_error_history.append(val_error)
            val_accuracy_history.append(val_acc)
            print(f"Val Loss: {val_error:.4f}, Val Acc: {val_acc:.4f}")

    time_to_train = time.time() - start_time

    return weights, epoch, time_to_train, error_history, accuracy_history, val_error_history, val_accuracy_history