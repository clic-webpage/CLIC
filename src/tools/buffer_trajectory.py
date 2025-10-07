# trajectory_buffer.py
import pickle

import argparse
import os
import pickle

import imageio
import cv2
import pdb
import h5py
import numpy as np

import zarr

class TrajectoryBuffer:
    """
    A buffer for storing trajectories. Each trajectory is a list of transitions,
    where each transition is a dictionary containing:
        - 'obs': The observation at the current timestep.
        - 'action': The action taken by the agent.
        - 'done': A boolean flag indicating if the episode is done.
        - 'state': Additional state information (if any).
        - 'timestep': The timestep within the episode.
        - 'h': The feedback or correction signal (if any).
    """

    def __init__(self):
        # List of trajectories (each trajectory is a list of transitions)
        self.trajectories = []
        # Current trajectory (list of transitions)
        self.current_trajectory = []

    def add_transition(self, obs, teacher_action, robot_action, done, timestep,
        no_robot_action, no_teacher_action, if_success = False,
     episode_id = None):
        """
        Add a single transition to the current trajectory.
        
        Parameters:
            obs: Observation data.
            action: Action taken by the agent.
            done: Boolean indicating if the episode is finished.
            state: Additional state information.
            timestep: The current timestep in the episode.
            no_robot_action: if true, the robot_action is set to zeros by default. 
        """
        transition = {
            "obs": obs,
            "robot_action": robot_action,
            'teacher_action': teacher_action,
            "done": done,
            # "state": state,
            "timestep": timestep,
            "no_teacher_action": no_teacher_action,
            "no_robot_action": no_robot_action,
            "episode_id": episode_id, 
            'if_success': if_success,
        }
        self.current_trajectory.append(transition)

    def finish_trajectory(self):
        """
        Ends the current trajectory and appends it to the list of saved trajectories.
        Resets the current trajectory for the next episode.
        """
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []

    def save_to_file(self, filename):
        """
        Appends all trajectories in self.trajectories to an HDF5 file.
        Each call will only append newly collected trajectories.
        """
        if not self.trajectories:
            return  # Nothing to save
        filename = filename + ".hdf5"
        with h5py.File(filename, 'a') as f:
            current_num_episodes = len(f.keys())

            for idx, traj in enumerate(self.trajectories):
                group = f.create_group(f"episode_{current_num_episodes + idx:04d}")

                # Prepare storage
                obs_dict = {}
                robot_actions, teacher_actions, dones, timesteps = [], [], [], []
                no_robot_actions, no_teacher_actions = [], []
                if_success = []

                for t in traj:
                    # observation dict
                    for k, v in t['obs'].items():
                        obs_dict.setdefault(k, []).append(v)

                    robot_actions.append(t['robot_action'])
                    teacher_actions.append(t['teacher_action'])
                    dones.append(t['done'])
                    timesteps.append(t['timestep'])
                    no_robot_actions.append(t['no_robot_action'])
                    no_teacher_actions.append(t['no_teacher_action'])
                    if_success.append(t['if_success'])

                # Save observation
                obs_group = group.create_group("observation")
                for k, v_list in obs_dict.items():
                    arr = np.stack(v_list, axis=0)
                    obs_group.create_dataset(k, data=arr, compression="gzip")

                # Save actions and flags
                group.create_dataset("robot_actions", data=np.stack(robot_actions), compression="gzip")
                group.create_dataset("teacher_actions", data=np.stack(teacher_actions), compression="gzip")
                group.create_dataset("dones", data=np.array(dones))
                group.create_dataset("timesteps", data=np.array(timesteps))
                group.create_dataset("no_robot_actions", data=np.array(no_robot_actions))
                group.create_dataset("no_teacher_actions", data=np.array(no_teacher_actions))
                group.create_dataset('if_success', data = np.array(if_success))


        self.clear()

    def load_from_file(self, filename, traj_id):
        """
        Loads all trajectories from an HDF5 file into self.trajectories.
        """
        self.trajectories = []  # Clear existing data

        with h5py.File(filename, 'r') as f:
            # Sort groups by name (episode_0000, episode_0001, ...)
            episode_keys = sorted(f.keys())

            if traj_id < 0 or traj_id >= len(episode_keys):
                raise IndexError(f"Trajectory index {traj_id} out of range (0 to {len(episode_keys)-1})")

            ep_key_i = episode_keys[traj_id]

            # for ep_key in episode_keys:
            for ep_key in [ep_key_i]:
                group = f[ep_key]

                # Load observations
                obs_group = group["observation"]
                obs_keys = list(obs_group.keys())
                obs_data = {k: obs_group[k][()] for k in obs_keys}  # each k: (T, ...)

                # Load other arrays
                robot_actions = group["robot_actions"][()]
                teacher_actions = group["teacher_actions"][()]
                dones = group["dones"][()]
                timesteps = group["timesteps"][()]
                no_robot_actions = group["no_robot_actions"][()]
                no_teacher_actions = group["no_teacher_actions"][()]
                if_success =  group["if_success"][()]

                T = len(timesteps)

                # Reconstruct trajectory as a list of transitions
                trajectory = []
                for t in range(T):
                    obs_t = {k: obs_data[k][t] for k in obs_keys}
                    transition = {
                        "obs": obs_t,
                        "robot_action": robot_actions[t],
                        "teacher_action": teacher_actions[t],
                        "done": dones[t],
                        "timestep": timesteps[t],
                        "no_robot_action": no_robot_actions[t],
                        "no_teacher_action": no_teacher_actions[t],
                        "episode_id": ep_key,
                        "if_success": if_success[t],
                    }
                    trajectory.append(transition)
                # import pdb; pdb.set_trace()
                # self.trajectories.append(trajectory)
        return trajectory

    def count_trajectories_in_hdf5(self, filename):
        """
        Returns the number of saved trajectories (episodes) in an HDF5 file.
        Assumes each episode is stored in a group named 'episode_xxxx'.
        """
        with h5py.File(filename, 'r') as f:
            # Count groups that look like 'episode_XXXX'
            count = sum(1 for key in f.keys() if key.startswith('episode_'))
        return count

    def clear(self):
        """
        Clears all trajectories from the buffer.
        """
        self.trajectories = []
        self.current_trajectory = []


def convert_hdf5_to_zarr(hdf5_path, zarr_path, compressor=None):
    """
    Converts an HDF5 file to a Zarr directory store.

    Parameters:
        hdf5_path: str, path to the input HDF5 file
        zarr_path: str, path to output Zarr directory
        compressor: zarr compressor (e.g., zarr.Blosc()), or None for no compression
    """
    # Open HDF5 for reading
    with h5py.File(hdf5_path, 'r') as f_in:
        # Create (or overwrite) Zarr store
        if os.path.exists(zarr_path):
            print(f"Removing existing Zarr store at {zarr_path}")
            import shutil
            shutil.rmtree(zarr_path)

        root = zarr.open(zarr_path, mode='w')

        # Recursive function to copy
        def copy_group(h5_group, zarr_group):
            # Copy attributes (if needed)
            for k, v in h5_group.attrs.items():
                zarr_group.attrs[k] = v

            for name, item in h5_group.items():
                if isinstance(item, h5py.Dataset):
                    print(f"Copying dataset: {item.name} -> {zarr_group.name}/{name}")
                    data = item[()]  # Load dataset into memory
                    zarr_group.array(
                        name,
                        data=data,
                        chunks=True,
                        compressor=compressor
                    )
                elif isinstance(item, h5py.Group):
                    print(f"Creating group: {item.name}")
                    sub_group = zarr_group.create_group(name)
                    copy_group(item, sub_group)

        # Start recursive copy
        copy_group(f_in, root)

    print(f"Conversion complete: {hdf5_path} -> {zarr_path}")


def create_preference_batch(traj1, traj2, segment_length):
    # Determine the maximum valid starting index in each trajectory.
    max_idx1 = len(traj1) - segment_length + 1
    max_idx2 = len(traj2) - segment_length + 1

    if max_idx1 < 1 or max_idx2 < 1:
        raise ValueError("Trajectories are too short for the given segment_length.")


    min_traj_length = min(len(traj1), len(traj2))
    segments1 = [traj1[i:i + segment_length] for i in range(0, min_traj_length, segment_length)]
    segments2 = [traj2[i:i + segment_length] for i in range(0, min_traj_length, segment_length)]


    # Split each trajectory into segments.
    segments1 = [traj1[i:i + segment_length] for i in range(0, len(traj1), segment_length)]
    segments2 = [traj2[i:i + segment_length] for i in range(0, len(traj2), segment_length)]

    # Use the minimum number of segments available from both trajectories.
    num_segments = min(len(segments1), len(segments2))
    preference_data = []

    for i in range(num_segments):
        seg1 = segments1[i]
        seg2 = segments2[i]

        preference_label =  1 # seg2 is prefred

        # Append the tuple (segment from traj1, segment from traj2, preference label)
        preference_data.append((seg1, seg2, preference_label))

    return preference_data




class TrajectoryBuffer_pickle:
    """
    A buffer for storing trajectories. Each trajectory is a list of transitions,
    where each transition is a dictionary containing:
        - 'obs': The observation at the current timestep.
        - 'action': The action taken by the agent.
        - 'done': A boolean flag indicating if the episode is done.
        - 'state': Additional state information (if any).
        - 'timestep': The timestep within the episode.
        - 'h': The feedback or correction signal (if any).
    """

    def __init__(self):
        # List of trajectories (each trajectory is a list of transitions)
        self.trajectories = []
        # Current trajectory (list of transitions)
        self.current_trajectory = []

    def add_transition(self, obs, teacher_action, robot_action, done, timestep,
        no_robot_action, no_teacher_action,
     episode_id = None):
        """
        Add a single transition to the current trajectory.
        
        Parameters:
            obs: Observation data.
            action: Action taken by the agent.
            done: Boolean indicating if the episode is finished.
            state: Additional state information.
            timestep: The current timestep in the episode.
            no_robot_action: if true, the robot_action is set to zeros by default. 
        """
        transition = {
            "obs": obs,
            "robot_action": robot_action,
            'teacher_action': teacher_action,
            "done": done,
            # "state": state,
            "timestep": timestep,
            "no_teacher_action": no_teacher_action,
            "no_robot_action": no_robot_action,
            "episode_id": episode_id, 
        }
        self.current_trajectory.append(transition)

    def finish_trajectory(self):
        """
        Ends the current trajectory and appends it to the list of saved trajectories.
        Resets the current trajectory for the next episode.
        """
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []

    def save_to_file(self, filename):
        """
        Saves the entire buffer (all trajectories) to a file using pickle.

        Parameters:
            filename (str): The file path where the buffer will be saved.
        """
        filename = filename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.trajectories, f)

    

    def load_from_file(self, filename):
        """
        Loads trajectories from a file into the buffer.

        Parameters:
            filename (str): The file path from where the buffer will be loaded.
        """
        with open(filename, "rb") as f:
            self.trajectories = pickle.load(f)

 

    def clear(self):
        """
        Clears all trajectories from the buffer.
        """
        self.trajectories = []
        self.current_trajectory = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64,
                        help="Resolution for rendering")
    parser.add_argument("--output_dir", type=str, default="gifs",
                        help="Directory to save GIFs")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for the GIF")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)


    buffer = TrajectoryBuffer()

    # buffer.load_from_file("trajectory_buffer_tested.pkl")
    buffer.load_from_file('trajectory_buffer.pkl')

    # from main_init import env_eval
    from env.metaworld_env.metaworld import MetaWorldSawyerEnv
    # task = "hammer-v2-goal-observable"
    task = "drawer-open-v2"
    task = task.strip('"')
    env_eval = MetaWorldSawyerEnv(task)

    trajectories = buffer.trajectories

    '''create the preference data'''
    preference_data_all = []
    for traj_idx in range(0, len(trajectories)-1, 2):
        traj1 = trajectories[traj_idx]
        traj2 = trajectories[traj_idx + 1]
        try:
            pref_batch = create_preference_batch(traj1, traj2, segment_length=64)
            # You could further process pref_batch (e.g., convert segments to torch tensors)
            preference_data_all.extend(pref_batch)
            print(f"Processed preference batch for trajectory pair ({traj_idx}, { traj_idx + 1}).")
        except ValueError as e:
            print(f"Skipping trajectory pair ({ traj_idx}, {traj_idx + 1}): {e}")

    # save the preference_data_all
    save_path = "preference_data_all.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(preference_data_all, f)

    '''Visualize the preference data'''
    # For each preference batch, we render both segments and then combine the frames side by side.
    for idx, (seg1, seg2, pref_label) in enumerate(preference_data_all):
        seg1_frames = []
        seg2_frames = []
        env_eval.reset()
        # Render frames for the first segment
        for transition in seg1:
            # state_q, state_v = transition["state"]
            # pdb.set_trace()
            env_eval.set_state(transition["state"])
            env_eval.render_mode = 'rgb_array'
            env_eval.camera_name = "corner2"
            img1 = env_eval.render()
            # pdb.set_trace()
            img1 = cv2.resize(img1, (0, 0), fx=0.6, fy=0.6)
            # img1 = cv2.rotate(img1, cv2.ROTATE_180)
            seg1_frames.append(img1)
        
        # Render frames for the second segment
        for transition in seg2:
            # state_q, state_v = transition["state"]
            env_eval.set_state(transition["state"])
            env_eval.render_mode = 'rgb_array'
            env_eval.camera_name = "corner2"
            img2 = env_eval.render()
            img2 = cv2.resize(img2, (0, 0), fx=0.6, fy=0.6)
            # img2 = cv2.rotate(img2, cv2.ROTATE_180)
            seg2_frames.append(img2)
        
        # Ensure both segments have the same number of frames by taking the minimum length
        n_frames = min(len(seg1_frames), len(seg2_frames))
        combined_frames = []
        for i in range(n_frames):
            # Convert the frames to BGR if needed (OpenCV expects BGR)
            frame_left = cv2.cvtColor(seg1_frames[i], cv2.COLOR_RGB2BGR)
            frame_right = cv2.cvtColor(seg2_frames[i], cv2.COLOR_RGB2BGR)
            # Concatenate the two frames horizontally
            combined_frame = cv2.hconcat([frame_left, frame_right])
            combined_frames.append(combined_frame)
        
        if not combined_frames:
            print(f"No frames to process for preference segment {idx}.")
            continue

        # Determine the dimensions of the combined frame
        height, width, _ = combined_frames[0].shape
        video_path = os.path.join(args.output_dir, f"preference_video_{idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
        for frame in combined_frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"Saved preference video for segment {idx} at: {video_path}")

    print("All preference videos have been processed and saved.")

    '''visualize the trajectory data'''
    # Iterate over each trajectory
    for traj_idx in range(len(trajectories)):
        env_eval.reset()
        # pdb.set_trace()
        trajectory = trajectories[traj_idx]
        frames = []
        print(f"Processing trajectory {traj_idx} with {len(trajectory)} transitions...")
        for transition in trajectory:
            # Use the 'state' field stored in the transition to set the environment's state.
            # Make sure that your environment's `set_state` method is compatible with the saved state.
            # env.set_state(transition["state"])
            state_q, state_v = transition["state"]
            # print("state: ", state_q)
            env_eval.set_env_state((state_q, state_v))
            # Render the image from the environment; returns an (H, W, 3) NumPy array.
            env_eval.render_mode = 'rgb_array'
            env_eval.camera_name = "corner2"
            img = env_eval.render()
            # img = cv2.resize(img, (0, 0), fx=0.267, fy=0.267)
            img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
            img = cv2.rotate(img, cv2.ROTATE_180)
            frames.append(img)
            
        # Determine the frame dimensions from the first frame
        height, width, _ = frames[0].shape
        # Define the video file name
        video_path = os.path.join(args.output_dir, f"trajectory_{traj_idx}.mp4")
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
        # Write each frame to the video file
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        
        # Release the video writer
        video_writer.release()
        print(f"Saved video for trajectory {traj_idx} at: {video_path}")

    print("All trajectories have been processed and GIFs are saved.")
