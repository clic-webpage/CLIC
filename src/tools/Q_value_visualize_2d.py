import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image

def visualize_normalized_q_values_tf_2d(model, state, robot_action, teacher_action, saved_path, resolution=200, saved_data_path=None):
    """
    Visualizes the normalized Q-values for a given state over a specified action space
    for a Q-function implemented as a TensorFlow model in 2D, with higher Q-values
    represented by darker colors.
    
    Parameters:
    - model: A TensorFlow model that computes Q-values given a state and an action.
    - state: The state for which to visualize the Q-values.
    - action_space_bounds: A tuple of tuples defining the min and max values of the action space in each dimension.
    - resolution: The number of points to sample along each dimension of the action space.
    """
    # Unpack the action space bounds
    (x_min, x_max), (y_min, y_max) = (-1, 1), (-1, 1)
    
    # Generate a grid of actions in 2D space
    action_space_x = tf.linspace(x_min, x_max, resolution)
    action_space_y = tf.linspace(y_min, y_max, resolution)
    action_space_x, action_space_y = tf.meshgrid(action_space_x, action_space_y)
    action_space_flat = tf.stack([tf.reshape(action_space_x, [-1]), tf.reshape(action_space_y, [-1])], axis=1)
    
    state_repeated = tf.tile(state, (action_space_flat.shape[0], 1))
    
    # Predict Q-values using the model
    # print("shape of action_space_flat: ", action_space_flat.shape)
    # print("shape of state_repeated ", state_repeated.shape)
    evaluation_model = tf.keras.models.clone_model(model)

    # Copy the weights from the original model to the evaluation model
    evaluation_model.set_weights(model.get_weights())
    Q_values = evaluation_model([state_repeated, action_space_flat])
    Q_values = tf.stop_gradient(Q_values)
    # print("shape of Q_value: ", Q_values.shape)
    # Q_value_numpy = Q_values.numpy()
    # Q_values = tf.reshape(Q_values, action_space_x.shape)
    
    # Q_values_normalized = tf.nn.softmax(Q_values).numpy()
    Q_values_normalized =  tf.nn.softmax(Q_values/0.2, axis=0)

    # # Convert Q-values from tf.Tensor to numpy if necessary
    if isinstance(Q_values, tf.Tensor):
        Q_values = Q_values.numpy()
        Q_values_normalized = Q_values_normalized.numpy()
    
    # # Normalize Q-values to range [0, 1]
    Q_min = Q_values.min()
    Q_max = Q_values.max()
    # print("Q_min: ", Q_min, " Q_max: ",Q_max)

    Q_min = Q_values_normalized.min()
    Q_max = Q_values_normalized.max()
    # print("Q_values_normalized_min: ", Q_min, " Q_max: ",Q_max)
    Q_values_normalized = (Q_values_normalized - Q_min) / (Q_max - Q_min)
    
    # Plotting in 2D
    # plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.grid(True)
    # plt.plot(Q_values)
    # im = plt.imshow(Q_values_normalized, cmap='Greys', origin='lower', extent=(x_min, x_max, y_min, y_max))
    # plt.colorbar(im, fraction=0.046, pad=0.04)
    # print("action_space_flat: ", action_space_flat.shape)
    plt.scatter(action_space_flat[:, 0], action_space_flat[:, 1],s=10, c= -1 * Q_values_normalized, cmap='gist_earth') # gnuplot2
    # plt.scatter(action_space_flat[:, 0], action_space_flat[:, 1],s=10, c= +1 * Q_values_normalized, cmap='turbo', alpha= 1)
    # plt.imshow(Q_values_normalized, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis', aspect='auto')
    
    # plt.colorbar(label='Normalized Q-value')
    
    # draw teacher_action here a_0 = a_0, a_1 = -1 * a_1
    # modified_teacher_action = [teacher_action[0], -1 * teacher_action[1]]
    modified_teacher_action = [teacher_action[0],  teacher_action[1]]

    # # Plotting the teacher action with a distinct marker, e.g., a red star
    # plt.scatter(robot_action[0], robot_action[1], color='black', marker='o', s=100, label='robot Action')

    # plt.scatter(modified_teacher_action[0], modified_teacher_action[1], color='red', marker='*', s=100, label='Teacher Action')

    # circle = plt.Circle((0, 0), 0.4, color='r', fill=False, linewidth=2, linestyle='--',label='Circle (radius=0.4)')
    # plt.gca().add_patch(circle)

    # Setting the limits
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.xlabel('Action Dimension 1')
    plt.ylabel('Action Dimension 2')
    
    # plt.axhline(0, color='black', linewidth=0.5)
    # plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Normalized Q-value Function Visualization for a Given State')
    plt.savefig(saved_path)

    # Save the data
    if saved_data_path is not None:
        np.savez(
            saved_data_path,
            x=action_space_flat[:, 0].numpy(),
            y=action_space_flat[:, 1].numpy(),
            q_values=Q_values_normalized,
            a_negative=robot_action,
            a_positive=teacher_action,
            sampled_action=None
        )


    plt.close() # Close the figure
    return None
    # plt.show()


def visualize_normalized_q_values_tf_2d_with_DeisreA(model, state, a_negative, a_positive, saved_path, resolution=200):
    """
    Visualizes the normalized Q-values for a given state over a specified action space
    for a Q-function implemented as a TensorFlow model in 2D, with higher Q-values
    represented by darker colors.
    
    Parameters:
    - model: A TensorFlow model that computes Q-values given a state and an action.
    - state: The state for which to visualize the Q-values.
    - action_space_bounds: A tuple of tuples defining the min and max values of the action space in each dimension.
    - resolution: The number of points to sample along each dimension of the action space.
    """
    # Unpack the action space bounds
    (x_min, x_max), (y_min, y_max) = (-1, 1), (-1, 1)
    
    # Generate a grid of actions in 2D space
    action_space_x = tf.linspace(x_min, x_max, resolution)
    action_space_y = tf.linspace(y_min, y_max, resolution)
    action_space_x, action_space_y = tf.meshgrid(action_space_x, action_space_y)
    action_space_flat = tf.stack([tf.reshape(action_space_x, [-1]), tf.reshape(action_space_y, [-1])], axis=1)
    
    state_repeated = tf.tile(state, (action_space_flat.shape[0], 1))
    
    # Predict Q-values using the model
    # print("shape of action_space_flat: ", action_space_flat.shape)
    # print("shape of state_repeated ", state_repeated.shape)
    evaluation_model = tf.keras.models.clone_model(model)

    # Copy the weights from the original model to the evaluation model
    evaluation_model.set_weights(model.get_weights())
    Q_values = evaluation_model([state_repeated, action_space_flat])
    Q_values = tf.stop_gradient(Q_values)
    # print("shape of Q_value: ", Q_values.shape)
    # Q_value_numpy = Q_values.numpy()
    # Q_values = tf.reshape(Q_values, action_space_x.shape)
    
    # Q_values_normalized = tf.nn.softmax(Q_values).numpy()
    Q_values_normalized =  tf.nn.softmax(Q_values/0.2, axis=0)

    # # Convert Q-values from tf.Tensor to numpy if necessary
    if isinstance(Q_values, tf.Tensor):
        Q_values = Q_values.numpy()
        Q_values_normalized = Q_values_normalized.numpy()
    
    # # Normalize Q-values to range [0, 1]
    Q_min = Q_values.min()
    Q_max = Q_values.max()
    # print("Q_min: ", Q_min, " Q_max: ",Q_max)

    Q_min = Q_values_normalized.min()
    Q_max = Q_values_normalized.max()
    # print("Q_values_normalized_min: ", Q_min, " Q_max: ",Q_max)
    Q_values_normalized = (Q_values_normalized - Q_min) / (Q_max - Q_min)
    
    # Plotting in 2D
    # plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.grid(True)
    # plt.plot(Q_values)
    # im = plt.imshow(Q_values_normalized, cmap='Greys', origin='lower', extent=(x_min, x_max, y_min, y_max))
    # plt.colorbar(im, fraction=0.046, pad=0.04)
    # print("action_space_flat: ", action_space_flat.shape)
    plt.scatter(action_space_flat[:, 0], action_space_flat[:, 1],s=10, c= -1 * Q_values_normalized, cmap='gist_earth') # gnuplot2
    # plt.scatter(action_space_flat[:, 0], action_space_flat[:, 1],s=10, c= +1 * Q_values_normalized, cmap='turbo', alpha= 1)
    # plt.imshow(Q_values_normalized, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis', aspect='auto')
    
    # plt.colorbar(label='Normalized Q-value')
    
    # draw teacher_action here a_0 = a_0, a_1 = -1 * a_1
    # modified_teacher_action = [teacher_action[0], -1 * teacher_action[1]]
    modified_teacher_action = [a_positive[0],  a_positive[1]]

    # # Plotting the teacher action with a distinct marker, e.g., a red star
    plt.scatter(a_negative[0], a_negative[1], color='black', marker='o', s=100, label='robot Action')

    plt.scatter(modified_teacher_action[0], modified_teacher_action[1], color='red', marker='*', s=100, label='Teacher Action')


    # Compute the middle point
    mid_point = (np.array(a_negative) + np.array(a_positive)) / 2

    # Compute the perpendicular vector to (a_positive - a_negative)
    v = np.array(a_positive) - np.array(a_negative)
    perpendicular_v = np.array([-v[1], v[0]])  # Rotate vector by 90 degrees

    # Normalize the perpendicular vector
    perpendicular_v = perpendicular_v / np.linalg.norm(perpendicular_v)

    # Define the line endpoints for visualization (scale for display purposes)
    line_scale = 1.5  # Scale factor for extending the line
    line_start = mid_point - line_scale * perpendicular_v
    line_end = mid_point + line_scale * perpendicular_v

    # Add line to the plot
    plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'blue', linestyle='--', linewidth=1, label='Perpendicular Line')

    # circle = plt.Circle((0, 0), 0.4, color='r', fill=False, linewidth=2, linestyle='--',label='Circle (radius=0.4)')
    # plt.gca().add_patch(circle)

    # Setting the limits
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.xlabel('Action Dimension 1')
    plt.ylabel('Action Dimension 2')
    
    # plt.axhline(0, color='black', linewidth=0.5)
    # plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Normalized Q-value Function Visualization for a Given State')
    plt.savefig(saved_path)
    plt.close() # Close the figure
    return None
    # plt.show()




def visualize_normalized_q_values_tf_2d_contour(
    model, state, a_negative, a_positive, saved_path,saved_data_path = None, sampled_action =None, resolution=200, draw_desiredA = True
):
    """
    Visualizes the normalized Q-values for a given state over a specified action space
    for a Q-function implemented as a TensorFlow model in 2D, using a contour map.
    
    Parameters:
    - model: A TensorFlow model that computes Q-values given a state and an action.
    - state: The state for which to visualize the Q-values.
    - a_negative: A reference action (e.g., robot's action).
    - a_positive: Another reference action (e.g., teacher's action).
    - saved_path: Path to save the visualization.
    - resolution: The number of points to sample along each dimension of the action space.
    """
    # Action space bounds
    (x_min, x_max), (y_min, y_max) = (-1, 1), (-1, 1)

    # Generate a grid of actions in 2D space
    action_space_x = tf.linspace(x_min, x_max, resolution)
    action_space_y = tf.linspace(y_min, y_max, resolution)
    action_space_x, action_space_y = tf.meshgrid(action_space_x, action_space_y)
    action_space_flat = tf.stack([tf.reshape(action_space_x, [-1]), tf.reshape(action_space_y, [-1])], axis=1)

    state_repeated = tf.tile(state, (action_space_flat.shape[0], 1))

    # Predict Q-values using the model
    evaluation_model = tf.keras.models.clone_model(model)

    # Copy the weights from the original model to the evaluation model
    evaluation_model.set_weights(model.get_weights())
    Q_values = evaluation_model([state_repeated, action_space_flat])
    Q_values = tf.stop_gradient(Q_values)
    

    # Normalize Q-values using softmax
    Q_values_normalized = tf.nn.softmax(Q_values / 0.2, axis=0).numpy()

    Q_min = Q_values_normalized.min()
    Q_max = Q_values_normalized.max()
    # print("Q_values_normalized_min: ", Q_min, " Q_max: ",Q_max)
    Q_values_normalized = (Q_values_normalized - Q_min) / (Q_max - Q_min)
    Q_values_normalized = tf.reshape(Q_values_normalized, action_space_x.shape).numpy()

    # Create a contour map
    fig, ax = plt.subplots(figsize=(8, 8))
    # contour = ax.contourf(action_space_x, action_space_y, Q_values_normalized, levels=50, cmap='viridis')
    contour = ax.contour(action_space_x, action_space_y, Q_values_normalized, levels=5, cmap='viridis', linewidths=2.5)

    # Add labels to the contours for clarity
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Q-value', rotation=270, labelpad=15)

    if draw_desiredA:
        # Draw a_negative and a_positive
        ax.scatter(a_negative[0], a_negative[1], color='black', marker='o', s=100, label='Robot Action')
        ax.scatter(a_positive[0], a_positive[1], color='red', marker='*', s=100, label='Teacher Action')

        # Draw the perpendicular line
        mid_point = (np.array(a_negative) + np.array(a_positive)) / 2
        v = np.array(a_positive) - np.array(a_negative)
        perpendicular_v = np.array([-v[1], v[0]]) / np.linalg.norm(v)  # Perpendicular vector
        line_scale = 1.5
        line_start = mid_point - line_scale * perpendicular_v
        line_end = mid_point + line_scale * perpendicular_v
        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'blue', linestyle='--', linewidth=1, label='Perpendicular Line')

    if sampled_action is not None:
        ax.scatter(
            sampled_action[:, 0],  # x-coordinates of sampled actions
            sampled_action[:, 1],  # y-coordinates of sampled actions
            color='green',         # Color for the sampled actions
            marker='x',            # Marker style for the sampled actions
            s=50,                  # Size of the marker
            label='Sampled Actions'  # Legend label
        )
    
    # Formatting
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Action Dimension 1')
    ax.set_ylabel('Action Dimension 2')
    ax.set_aspect('equal', adjustable='box')

    # # Remove ticks and labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    ax.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(saved_path)
    plt.close()

    # Save the data
    if saved_data_path is not None:
        np.savez(
            saved_data_path,
            x=action_space_x.numpy(),
            y=action_space_y.numpy(),
            q_values=Q_values_normalized,
            a_negative=a_negative,
            a_positive=a_positive,
            sampled_action=sampled_action
        )


# used to combine two images together
def combine_images(img_path1, img_path2, output_path):
    image1 = Image.open(img_path1)
    image2 = Image.open(img_path2)

    # Assuming you want to stack them horizontally
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)

    combined_image = Image.new('RGB', (total_width, max_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))

    combined_image.save(output_path)
    return None
