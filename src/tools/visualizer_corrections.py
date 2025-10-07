import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np

class RobotTrajectoryVisualizer:
    def __init__(self, action_dim=3):
        self.action_dim = action_dim
        self.trajectories = []

    def record_step(self, state, action, correction):
        if not self.trajectories:
            self.trajectories.append([])
        if correction is None:
            correction = (0, 0, 0)
        self.trajectories[-1].append((state, action, correction))

    def reset(self):
        self.trajectories.append([])

    def visualize(self, episode_id):
        if episode_id >= len(self.trajectories):
            print(f"Episode {episode_id} does not exist.")
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        states, actions, corrections = zip(*self.trajectories[episode_id])

        if self.action_dim == 3:
            xs, ys, zs = zip(*states)
            ax.plot(xs, ys, zs, label='Trajectory')
            ax.scatter(xs, ys, zs, c='r', marker='o', s = 100)  # End effector positions
            max_range = np.array([max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]).max() / 2.0
            mid_x = (max(xs) + min(xs)) * 0.5
            mid_y = (max(ys) + min(ys)) * 0.5
            mid_z = (max(zs) + min(zs)) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            for (x, y, z), action, correction in zip(states, actions, corrections):
                if any(correction):  # If there is a non-zero correction
                    cx, cy, cz = correction
                    ax.quiver(x, y, z, 0.1 * cx, 0.1 * cy, 0.1 * cz, color='blue', length=0.02, normalize=True)
                    a_x, ay, az = action
                    ax.quiver(x, y, z, a_x, ay, az, color='green', length=0.04, normalize=True)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        # ax.set_aspect('auto')
        # ax.set_box_aspect([1,1,1])  # Equal aspect ratio

        plt.title(f"Robot Trajectory for Episode {episode_id}")
        plt.legend()
        plt.show()

    def save(self, filename, episode_id):
        if episode_id >= len(self.trajectories):
            print(f"Episode {episode_id} does not exist.")
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        states, corrections = zip(*self.trajectories[episode_id])

        if self.action_dim == 3:
            xs, ys, zs = zip(*states)
            ax.plot(xs, ys, zs, label='Trajectory')
            ax.scatter(xs, ys, zs, c='r', marker='o')  # End effector positions

            for (x, y, z), correction in zip(states, corrections):
                if any(correction):  # If there is a non-zero correction
                    cx, cy, cz = correction
                    ax.quiver(x, y, z, cx, cy, cz, color='blue', length=0.1, normalize=True)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.title(f"Robot Trajectory for Episode {episode_id}")
        plt.legend()

        # Create directories if not exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
