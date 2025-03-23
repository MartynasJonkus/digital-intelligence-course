import numpy as np
import pandas as pd

def sigmoid_activation(a):
    return 1 / (1 + np.exp(-a))

def evaluate_model(data_rows, weights, output_file=None):  # Added output_file parameter
    total_error = 0.0
    total_correct = 0
    results = []

    # Determine column names based on first row (if data exists)
    num_features = len(data_rows[0]) - 1 if data_rows else 0
    feature_columns = [f'Feature_{i+1}' for i in range(num_features)]
    columns = feature_columns + ['True_Label', 'Predicted_Label']

    for row in data_rows:
        # Original features (without bias term)
        features_data = row[:-1]
        class_label = row[-1]
        
        # Add bias term for calculation
        features_with_bias = [1] + features_data
        
        # Calculate activation and prediction
        a = sum(x * w for x, w in zip(features_with_bias, weights))
        y = sigmoid_activation(a)
        y = np.clip(y, 1e-8, 1 - 1e-8)
        predicted_label = 1 if y >= 0.5 else 0

        # Update performance metrics
        total_correct += 1 if predicted_label == class_label else 0
        total_error += -np.log(y) if class_label == 1 else -np.log(1 - y)

        # Collect results for Excel export
        results.append(features_data + [class_label, predicted_label])

    # Calculate final metrics
    avg_error = total_error / len(data_rows) if data_rows else 0
    accuracy = total_correct / len(data_rows) if data_rows else 0

    # Export to Excel if requested
    if output_file is not None:
        df = pd.DataFrame(results, columns=columns)
        df.to_excel(output_file, index=False)

    return avg_error, accuracy