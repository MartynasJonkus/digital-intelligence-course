import numpy as np

def sigmoid_activation(a):
    return 1 / (1 + np.exp(-a))

def evaluate_model(data_rows, weights):
    total_error = 0.0
    total_correct = 0

    for row in data_rows:
        features = [1] + row[:-1]
        class_label = row[-1]
        
        a = sum(x * w for x, w in zip(features, weights))
        y = sigmoid_activation(a)
        y = np.clip(y, 1e-8, 1 - 1e-8)

        total_correct += 1 if (y >= 0.5) == class_label else 0
        
        total_error += -np.log(y) if class_label == 1 else -np.log(1 - y)
        
    avg_error = total_error / len(data_rows)
    accuracy = total_correct / len(data_rows)
    return avg_error, accuracy