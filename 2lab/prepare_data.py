import random

def prepare_data(input_file):
    # Read and process data
    processed_rows = []
    
    with open(input_file, 'r') as f:
        for line in f:
            # Skip empty lines
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            # Split line into components
            parts = stripped_line.split(',')
            
            # Skip lines that don't have exactly 11 values
            if len(parts) != 11:
                continue
            
            # Skip lines with missing values (containing '?')
            if '?' in parts:
                continue
            
            # Remove ID (first element)
            features = parts[1:-1]  # Keep only features
            class_label = parts[-1]  # Get class label
            
            # Convert class label (2 -> 0, 4 -> 1)
            try:
                class_label = 0 if int(class_label) == 2 else 1
            except ValueError:
                continue  # Skip invalid class labels
            
            # Convert all features to integers and add class label
            try:
                processed_row = [int(x) for x in features] + [class_label]
                processed_rows.append(processed_row)
            except ValueError:
                continue  # Skip rows with non-integer features
    
    # Shuffle the processed data
    random.shuffle(processed_rows)
    
    return processed_rows