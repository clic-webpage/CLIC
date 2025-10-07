import h5py
import numpy as np
import os

class HDF5TrajectoryBuffer:
    """
    A trajectory buffer for saving transition data directly to HDF5 with support for dict-based observations.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.h5file = h5py.File(self.file_path, 'a')  # append mode
        self.episode_idx = 0

    def add_trajectory(self, transitions):
        """
        Saves one trajectory to the HDF5 file. Each transition is a dict containing observation dict and other fields.
        """
        group_name = f"episode_{self.episode_idx:04d}"
        ep_group = self.h5file.create_group(group_name)

        # First, unpack and group data
        obs_dict = {}
        actions, dones, states, timesteps, h_signals = [], [], [], [], []

        for t in transitions:
            # Accumulate low_dim data
            for k, v in t['obs'].items():
                obs_dict.setdefault(k, []).append(v)
            actions.append(t['action'])
            dones.append(t['done'])
            states.append(t['state'])
            timesteps.append(t['timestep'])
            h_signals.append(t['h'] if t['h'] is not None else np.zeros_like(t['action']))

        # Save each observation component separately
        obs_group = ep_group.create_group("observation")
        for k, v_list in obs_dict.items():
            arr = np.stack(v_list, axis=0)
            obs_group.create_dataset(k, data=arr, compression="gzip")

        # Save the rest
        ep_group.create_dataset("actions", data=np.stack(actions), compression="gzip")
        ep_group.create_dataset("dones", data=np.array(dones))
        ep_group.create_dataset("states", data=np.stack(states), compression="gzip")
        ep_group.create_dataset("timesteps", data=np.array(timesteps))
        ep_group.create_dataset("h", data=np.stack(h_signals), compression="gzip")

        self.episode_idx += 1

    def close(self):
        self.h5file.close()
