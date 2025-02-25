from matplotlib import pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate two clusters of points
# By default Class 0 points around (2.5, 2.5), Class 1 points around (7.5, 7.5)
def generate_clusters(mean0=(2.5, 2.5), mean1=(7.5, 7.5), std=2.5, size=10):
    # Generate points from normal distribution, clipping to stay within bounds
    class0_points = np.random.normal(mean0, std, (size, 2)).clip(0.1, 5)
    class1_points = np.random.normal(mean1, std, (size, 2)).clip(5, 9.9)

    # Print coordinates in a format that's appropriate for the lab report
    for x, y in class0_points:
        print(f"({str(x).replace('.', ',')}; {str(y).replace('.', ',')})")
    for x, y in class1_points:
        print(f"({str(x).replace('.', ',')}; {str(y).replace('.', ',')})")

    return class0_points, class1_points

# Threshold activation function: outputs 1 if a >= 0, else 0
def threshold_activation(a):
    return 1 if a >= 0 else 0

# Sigmoid activation function: maps 'a' to a value between 0 and 1
def sigmoid_activation(a):
    return 1 / (1 + np.exp(-a))

# Artificial neuron function to classify points based on weights, bias, and activation function
def artificial_neuron(points, weights, bias, activation='threshold'):
    outputs = []
    
    for x1, x2 in points:
        # Calculate activation value 'a' based on weights and bias
        a = x1 * weights[0] + x2 * weights[1] + bias
        
        # Use the selected activation function
        if activation == 'threshold':
            output = threshold_activation(a)
        elif activation == 'sigmoid':
            # Convert sigmoid output to binary class 0 or 1
            output = 1 if sigmoid_activation(a) >= 0.5 else 0
        else:
            raise ValueError("Activation must be 'threshold' or 'sigmoid'")
        
        outputs.append(output)
    
    return outputs

# Check if all points are correctly classified by a set of weights and bias
def check_classification(points, true_class, weights, bias, activation='threshold'):
    outputs = artificial_neuron(points, weights, bias, activation)
    return all(output == true_class for output, true_class in zip(outputs, true_class))

# Plot class points
# ax: the plot axis
# class0, class1: coordinates of points
def plot_points(ax, class0, class1):
    ax.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0')
    ax.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.grid(True, zorder=0)
    ax.legend()
    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(0, color='black', linewidth=2)
    ax.set_title('Class points')

# Plot decision boundaries and weight vectors
def plot_boundaries(ax, class0, class1, weight_bias_sets):
    plt_start = -1
    plt_end = 11
    
    ax.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0')
    ax.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
    
    x_vals = np.linspace(plt_start, plt_end, 100)
    
    colors = ['green', 'purple', 'orange']
    
    for i, (weights, bias) in enumerate(weight_bias_sets):
        w1, w2 = weights
        
        # Calculate y-values for the decision boundary
        if w2 != 0:
            y_vals = - (w1 / w2) * x_vals - (bias / w2)
            ax.plot(x_vals, y_vals, color=colors[i], label=f'Boundary {i + 1}')
            # Calculate a point on the decision boundary to anchor vectors
            x0 = 5
            y0 = - (w1 / w2) * x0 - (bias / w2)
        else: # vertical boundary case
            x_intercept = -bias / w1 if w1 != 0 else 0
            ax.axvline(x=x_intercept, color=colors[i], label=f'Boundary {i + 1}')

            x0, y0 = x_intercept, 5
        
        # Plot weight vector starting from the boundary
        ax.quiver(x0, y0, w1, w2, angles='xy', scale_units='xy', scale=1, color=colors[i], label=f'Vector {i + 1}')
    
    ax.set_xlim(plt_start, plt_end)
    ax.set_ylim(plt_start, plt_end)
    ax.set_aspect('equal')
    ax.grid(True, zorder=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(0, color='black', linewidth=2)
    ax.set_title('Decision boundaries and weight vectors')
    

# Generate class points
class0_points, class1_points = generate_clusters(mean0=(2.5, 2.5), mean1=(7.5, 7.5), std=2.5, size=10)


# Define ground truth classes
truth_class0 = [0] * len(class0_points)
truth_class1 = [1] * len(class1_points)
all_points = np.vstack((class0_points, class1_points))
all_classes = truth_class0 + truth_class1


# Find 3 valid sets of weights and bias for both activations
threshold_values = []
sigmoid_values = []

while len(threshold_values) < 3 and len(sigmoid_values) < 3:
    weights = np.random.uniform(-10, 10, 2)
    bias = np.random.uniform(-10, 10)
    
    if check_classification(all_points, all_classes, weights, bias, activation='threshold'):
            threshold_values.append((weights, bias))

    if check_classification(all_points, all_classes, weights, bias, activation='sigmoid'):
            sigmoid_values.append((weights, bias))

print("Valid sets of weights and bias for threshold activation:")
for i, (weights, bias) in enumerate(threshold_values):
    print(f"Set {i+1}: Weights = {weights}, Bias = {bias}")

print("\nValid sets of weights and bias for sigmoid activation:")
for i, (weights, bias) in enumerate(sigmoid_values):
    print(f"Set {i+1}: Weights = {weights}, Bias = {bias}")


# Create and display side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

plot_points(axs[0], class0_points, class1_points) 
plot_boundaries(axs[1], class0_points, class1_points, threshold_values)

plt.tight_layout()
plt.show()
