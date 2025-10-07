import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image

def visualize_normalized_q_values_tf_1d(model, state, robot_action, teacher_action, saved_path, resolution=200):
    """
    Visualizes the normalized Q-values for a given state over a specified 1D action space
    for a Q-function implemented as a TensorFlow model, with higher Q-values represented by darker colors.
    
    Parameters:
    - model: A TensorFlow model that computes Q-values given a state and an action.
    - state: The state for which to visualize the Q-values.
    - resolution: The number of points to sample along the action space.
    - saved_path: Path where the generated image will be saved.
    """
    # Define the 1D action space bounds
    x_min, x_max = -1, 1
    
    # Generate a grid of actions in 1D space
    action_space_x = tf.linspace(x_min, x_max, resolution)
    action_space_flat = tf.reshape(action_space_x, [-1, 1])  # Reshape to make it 2D for the model
    
    # Repeat the state for each action
    state_repeated = tf.tile(state, (action_space_flat.shape[0], 1))
    
    print("action_space_flat: ", action_space_flat.shape, " state_repeated: ", state_repeated.shape)
    # Clone the model and predict Q-values
    evaluation_model = tf.keras.models.clone_model(model)
    evaluation_model.set_weights(model.get_weights())
    Q_values = evaluation_model([state_repeated, action_space_flat])
    Q_values = tf.stop_gradient(Q_values)
    
    # Normalize the Q-values using softmax
    Q_values_normalized = tf.nn.softmax(Q_values / 0.2, axis=0)

    # Convert Q-values to numpy if necessary
    if isinstance(Q_values, tf.Tensor):
        Q_values = Q_values.numpy()
        Q_values_normalized = Q_values_normalized.numpy()

    # # Normalize Q-values to the range [0, 1]
    Q_min = Q_values_normalized.min()
    Q_max = Q_values_normalized.max()
    # Q_values_normalized = (Q_values_normalized - Q_min) / (Q_max - Q_min)
    
    # Plotting in 1D
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.grid(True)
    
    # Plot Q-values along the action dimension
    plt.plot(action_space_x,   Q_values_normalized, label='Q-values', color='blue')

    # Plotting the robot action and teacher action
    # plt.axvline(robot_action[0], color='black', linestyle='--', label='Robot Action', linewidth=2)
    # plt.axvline(teacher_action[0], color='red', linestyle='--', label='Teacher Action', linewidth=2)
    
    # Set the limits of the plot
    plt.xlim(x_min, x_max)
    plt.ylim(0, 1.2 * Q_max)
    
    plt.xlabel('Action Dimension')
    plt.ylabel('Normalized Q-value')
    
    plt.legend()
    plt.title('Normalized Q-value Function Visualization in 1D')
    
    # Save the plot
    plt.savefig(saved_path)
    plt.close()  # Close the figure to free memory
    return None
