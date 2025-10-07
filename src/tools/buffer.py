import random
import pickle
import numpy as np
import tensorflow as tf
import os

# from env.metaworld_env.sawyer_hammer_v2_policy import SawyerHammerV2Policy


from collections import deque

import h5py


import shutil
from tqdm import tqdm 

class HDF5Buffer:
    def __init__(
        self,
        filename: str,
        field_shapes: dict,
        min_size: int,
        max_size: int,
        dtype_map: dict = None,
        compression: str = "lzf"
    ):
        """
        Args:
            filename: path to .h5 file (will be created/opened).
            field_shapes: dict mapping field_name -> tuple(shape,),
                e.g. {
                    'agentview_image': (3,84,84),
                    'robot0_eye_in_hand_image': (3,84,84),
                    'robot0_eef_pos': (3,),
                    'robot0_eef_quat': (4,),
                    'robot0_gripper_qpos': (2,),
                    'robot_action': (10,),
                    'teacher_action': (10,)
                }
            min_size, max_size: same semantics as before.
            dtype_map: optional dict mapping field_name -> numpy dtype.
            compression: HDF5 compression filter.
        """
        self.filename    = filename
        self.min_size    = min_size
        self.max_size    = max_size
        self.field_shapes = field_shapes
        self.dtype_map   = dtype_map or {}
        self.compression = compression

        # open/create HDF5 file
        self.f = h5py.File(filename, 'a')
        self.datasets = {}
        for name, shape in field_shapes.items():
            dtype = self.dtype_map.get(name, 'float32')
            if name not in self.f:
                self.datasets[name] = self.f.create_dataset(
                    name,
                    shape=(max_size, *shape),
                    dtype=dtype,
                    chunks=(1, *shape),
                    compression=self.compression
                )
            else:
                self.datasets[name] = self.f[name]

        # pointer & count trackers
        if 'ptr' not in self.f:
            self.ptr_ds   = self.f.create_dataset('ptr',   data=0)
        else:
            self.ptr_ds   = self.f['ptr']
        if 'count' not in self.f:
            self.count_ds = self.f.create_dataset('count', data=0)
        else:
            self.count_ds = self.f['count']

    def full(self) -> bool:
        return int(self.count_ds[()]) >= self.max_size

    def initialized(self) -> bool:
        return int(self.count_ds[()]) >= self.min_size

    def length(self) -> int:
        return int(self.count_ds[()])

    def add(self, step: tuple):
        """
        step: a tuple (obs_dict, action_robot, action_teacher)
            - obs_dict: dict mapping all non-action field names to arrays/scalars
            - action_robot: array or list matching field_shapes['robot_action']
            - action_teacher: array or list matching field_shapes['teacher_action']
        """
        # obs_dict, action_robot, action_teacher = step
        obs_dict, action_teacher, action_robot = step
        i = int(self.ptr_ds[()])

        # write observation fields
        for name in self.field_shapes:
            if name in ('robot_action', 'teacher_action'):
                continue
            self.datasets[name][i] = obs_dict[name]

        # write actions
        print("action_robot: ", action_robot, " action_teacher: ", action_teacher)
        self.datasets['robot_action'][i]  = np.array(action_robot, dtype=self.datasets['robot_action'].dtype)
        self.datasets['teacher_action'][i] = np.array(action_teacher, dtype=self.datasets['teacher_action'].dtype)

        # advance pointer & count
        self.ptr_ds[...]   = (i + 1) % self.max_size
        self.count_ds[...] = min(self.count_ds[()] + 1, self.max_size)

    def sample(self, batch_size: int):
        """
        Returns a list of tuples (obs_dict, action_robot, action_teacher)
        matching the structure given to `add`.
        """
        n = self.length()
        idxs = np.random.randint(0, n, size=batch_size)
        batch = []
        for idx in idxs:
            # reconstruct observation dict
            obs_dict = {
                name: self.datasets[name][idx]
                for name in self.field_shapes
                if name not in ('robot_action', 'teacher_action')
            }
            action_robot  = self.datasets['robot_action'][idx]
            action_teacher = self.datasets['teacher_action'][idx]
            # batch.append((obs_dict, action_robot, action_teacher))
            batch.append((obs_dict, action_teacher, action_robot))
        return batch

    def save_to_file(self, filename: str):
        """
        Copy the underlying HDF5 file to `filename`.
        """
        self.f.flush()
        self.f.close()
        shutil.copy(self.filename, filename)
        self.f = h5py.File(self.filename, 'a')
        self._rebind_handles()

    def load_from_file(self, filename: str):   
        # this will make a copy of the existing buffer.h5 (filename)
        # usually is used when the data is too big to be directly loaded into a list, such as real-world exps
        """
        Overwrite current buffer with the HDF5 file at `filename`.
        """
        self.f.close()
        shutil.copy(filename, self.filename)
        self.f = h5py.File(self.filename, 'a')
        self._rebind_handles()

    def _rebind_handles(self):
        self.datasets = {name: self.f[name] for name in self.field_shapes}
        self.ptr_ds   = self.f['ptr']
        self.count_ds = self.f['count']

    def close(self):
        self.f.close()


    #  ingest previously-saved TrajectoryBuffer HDF5 files (defined in buffer_trajectory.py)
    def ingest_trajectory_hdf5(
        self,
        traj_filename: str,
        *,
        chunk_size: int = 2048,
        skip_no_robot_action: bool = False,
        # skip_no_teacher_action: bool = False,
        skip_no_teacher_action: bool = True,
        show_progress: bool = True,
    ) -> int:
        """
        Stream transitions from <traj_filename>.hdf5 into *this* buffer's file
        (`self.f`) in chunks, without keeping the whole data in RAM.

        Parameters
        ----------
        traj_filename : str
            Base name used in TrajectoryBuffer.save_to_file ('.hdf5' added
            automatically).
        chunk_size : int
            How many transitions to copy in one I/O block.  Tune for your SSD /
            network FS; 2-8 k is usually a good balance between speed & memory.
        skip_no_robot_action / skip_no_teacher_action : bool
            Drop transitions whose flags are True.
        show_progress : bool
            If True, show a tqdm bar while copying.

        Returns
        -------
        n_written : int
            Number of transitions actually written into the replay buffer.
        """
        from hydra.utils import get_original_cwd
        from hydra.core.hydra_config import HydraConfig
        # if use hydra, define path as follows
        if HydraConfig and getattr(HydraConfig, "is_initialized", None) and HydraConfig.is_initialized():
            project_root = get_original_cwd()
            # import pdb; pdb.set_trace()
            full_path = os.path.join(project_root, traj_filename)
            src_path = full_path 
        else: # if not use hydra, simply use the traj_filename
            src_path = traj_filename
        print("load hdf5 buffer from trajectory dataset: ", src_path)
        n_written = 0

        with h5py.File(src_path, "r") as src:
            episode_keys = sorted(k for k in src if k.startswith("episode_"))

            # Pre-extract dest handles so attribute lookups stay cheap
            dst_obs_dsets = {
                name: self.datasets[name]
                for name in self.field_shapes
                if name not in ("robot_action", "teacher_action")
            }
            dst_robot = self.datasets["robot_action"]
            dst_teacher = self.datasets["teacher_action"]

            cur_ptr = int(self.ptr_ds[()])
            cur_cnt = int(self.count_ds[()])

            pbar = tqdm(total=len(episode_keys), disable=not show_progress,
                        desc="Episodes") if show_progress else None

            for ep in episode_keys:
                g = src[ep]

                # Source handles
                # s_obs  = {k: g["observation"][k] for k in g["observation"]}
                s_obs = {}
                for k in g["observation"]:
                    ds_np = g["observation"][k][()]          # load the whole array for key k
                    exp_shape = self.field_shapes.get(k)
                    # Guard: if we don’t know the expected shape, just keep it
                    if exp_shape is None:
                        s_obs[k] = ds_np
                        continue

                    per_sample_shape = ds_np.shape[1:]
                    need_aug = (
                        per_sample_shape == tuple(exp_shape[1:])
                        and (len(per_sample_shape) == len(exp_shape) - 1)
                    )


                    if need_aug:                      # (T, C, H, W)  →  make (T, 2, C, H, W)
                        T = ds_np.shape[0]
                        stacked = np.empty((T, 2, *ds_np.shape[1:]), dtype=ds_np.dtype)
                        stacked[:, 1, ...] = ds_np           # current frame in slot 1
                        stacked[0, 0, ...] = ds_np[0]        # t==0: duplicate itself
                        if T > 1:
                            stacked[1:, 0, ...] = ds_np[:-1] # slot 0 gets previous frame
                        s_obs[k] = stacked
                    else:
                        # Already has leading stack dimension (T, 2, C, H, W) or non-image
                        s_obs[k] = ds_np

                s_r    = g["robot_actions"]
                s_t    = g["teacher_actions"]
                s_no_r = g["no_robot_actions"]
                s_no_t = g["no_teacher_actions"]
                # import pdb; pdb.set_trace()

                T = s_r.shape[0]
                start = 0
                while start < T:
                    end = min(start + chunk_size, T)

                    # Apply skip masks if needed --------------------------------
                    if skip_no_robot_action or skip_no_teacher_action:
                        # build a boolean mask for the slice
                        skip_mask = np.zeros(end - start, dtype=bool)
                        if skip_no_robot_action:
                            skip_mask |= s_no_r[start:end]
                        if skip_no_teacher_action:
                            skip_mask |= s_no_t[start:end]
                            # import pdb; pdb.set_trace()

                        keep_idx = np.nonzero(~skip_mask)[0]
                        if keep_idx.size == 0:
                            start = end
                            continue  # nothing worth copying in this slice

                        # Fancy-index the arrays into an in-RAM chunk
                        # (h5py must materialise them for selection anyway)
                        obs_chunk = {
                            k: s_obs[k][start:end][keep_idx] for k in s_obs
                        }
                        r_chunk = s_r[start:end][keep_idx]
                        t_chunk = s_t[start:end][keep_idx]
                    else:
                        # Straight, contiguous slice; pull once per field
                        obs_chunk = {k: s_obs[k][start:end] for k in s_obs}
                        r_chunk   = s_r[start:end]
                        t_chunk   = s_t[start:end]

                    n_chunk = r_chunk.shape[0]
                    if n_chunk == 0:
                        start = end
                        continue

                    # Where to write in destination --------------------------------
                    dst_end = cur_ptr + n_chunk
                    wrap = dst_end > self.max_size
                    if wrap:
                        first_part = self.max_size - cur_ptr
                        second_part = n_chunk - first_part
                    else:
                        first_part, second_part = n_chunk, 0

                    # -------- write FIRST part (no wrap or pre-wrap region) -------
                    sl = slice(cur_ptr, cur_ptr + first_part)
                    for k, arr in obs_chunk.items():
                        dst_obs_dsets[k][sl] = arr[:first_part]
                    dst_robot[sl]   = r_chunk[:first_part]
                    dst_teacher[sl] = t_chunk[:first_part]

                    # -------- optional wrap-around write --------------------------
                    if wrap and second_part:
                        sl2 = slice(0, second_part)
                        for k, arr in obs_chunk.items():
                            dst_obs_dsets[k][sl2] = arr[first_part:]
                        dst_robot[sl2]   = r_chunk[first_part:]
                        dst_teacher[sl2] = t_chunk[first_part:]

                    # Update pointers ---------------------------------------------
                    cur_ptr = (cur_ptr + n_chunk) % self.max_size
                    cur_cnt = min(cur_cnt + n_chunk, self.max_size)
                    n_written += n_chunk

                    if self.full():            # stop once buffer saturated
                        break

                    start = end

                if pbar:
                    pbar.update(1)

                if self.full():
                    break

            if pbar:
                pbar.close()

            # persist updated ptr / count to the file
            self.ptr_ds[...]   = cur_ptr
            self.count_ds[...] = cur_cnt
            self.f.flush()
        # import pdb; pdb.set_trace()
        return n_written


    def ingest_trajectory_hdf5_Ta(
        self,
        traj_filename: str,
        *,
        Ta: int = 16,
        chunk_size: int = 2048,
        require_all_teacher_present: bool = True,
        pad_tail: bool = True,
        show_progress: bool = True,
    ) -> int:
        """
        Stream length-Ta windows from <traj_filename>.hdf5 into this buffer.

        For each episode with length T:
        - Build windows of actions [t : t+Ta) for both robot & teacher.
        - Save the observation at index t+1 (mimics train_single_repetition's
            data_id=1 logic, where obs aligns with the 2nd element of the window).
        - A window is valid if teacher actions are present over the whole window
            (ALL no_teacher_actions==False) when require_all_teacher_present=True;
            otherwise at least one step present (ANY).
        - If pad_tail=True and t+Ta > T, pad with the last available action.

        NOTE:
        Your buffer must have action shapes (Ta, act_dim).

        Returns
        -------
        n_written : int
            Number of Ta-windows written.
        """
        from hydra.utils import get_original_cwd
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig and getattr(HydraConfig, "is_initialized", None) and HydraConfig.is_initialized():
            project_root = get_original_cwd()
            src_path = os.path.join(project_root, traj_filename)
        else:
            src_path = traj_filename

        print(f"[Ta={Ta}] load trajectory dataset from: {src_path}")
        n_written = 0

        # Pre-bind destination handles
        dst_obs_dsets = {
            name: self.datasets[name]
            for name in self.field_shapes
            if name not in ("robot_action", "teacher_action")
        }
        dst_robot = self.datasets["robot_action"]
        dst_teacher = self.datasets["teacher_action"]

        cur_ptr = int(self.ptr_ds[()])
        cur_cnt = int(self.count_ds[()])

        sum_of_feedback = 0
        sum_of_padding = 0

        with h5py.File(src_path, "r") as src:
            episode_keys = sorted(k for k in src if k.startswith("episode_"))
            pbar = tqdm(total=len(episode_keys), disable=not show_progress,
                        desc=f"Episodes (Ta={Ta})") if show_progress else None

            for ep in episode_keys:
                g = src[ep]

                # ---- Load & shape observations (reuse your frame-stack augmentation) ----
                s_obs = {}
                for k in g["observation"]:
                    ds_np = g["observation"][k][()]  # shape (T, ...)
                    exp_shape = self.field_shapes.get(k)

                    if exp_shape is None:
                        s_obs[k] = ds_np
                        continue

                    per_sample_shape = ds_np.shape[1:]
                    need_aug = (
                        per_sample_shape == tuple(exp_shape[1:])
                        and (len(per_sample_shape) == len(exp_shape) - 1)
                    )
                    if need_aug:
                        T = ds_np.shape[0]
                        stacked = np.empty((T, 2, *ds_np.shape[1:]), dtype=ds_np.dtype)
                        stacked[:, 1, ...] = ds_np
                        stacked[0, 0, ...] = ds_np[0]
                        if T > 1:
                            stacked[1:, 0, ...] = ds_np[:-1]
                        s_obs[k] = stacked
                    else:
                        s_obs[k] = ds_np

                # ---- Load actions & masks ----
                s_r    = g["robot_actions"][()]          # (T, A)
                s_t    = g["teacher_actions"][()]        # (T, A)
                s_no_r = g["no_robot_actions"][()]       # (T,) bool
                s_no_t = g["no_teacher_actions"][()]     # (T,) bool
                if_success_list = g['if_success'][()]

                T, act_dim = s_r.shape
                if T == 0:
                    if pbar: pbar.update(1)
                    continue

                # ---- Build list of valid window start indices ----
                starts = []
                # observation index is t+1; require it be within [0, T-1]
                # so t must be in [ -1, T-2 ], but we start from 0 and clamp obs index.
                for t in range(T):  # consider all starts; we'll clamp/pad as needed
                    end = t + Ta
                    window_mask = s_no_t[t:min(end, T)]
                    if require_all_teacher_present:
                        # TODO the traj_data should also save if_success iterm so that we can do padding for the success expisodes
                        if_success = if_success_list[-1] ## to do, load from data
                        ok_teacher = (window_mask.size == Ta and not window_mask.any()) \
                                    or (pad_tail and not window_mask.any() and if_success) \
                                    or (not pad_tail and end <= T and not window_mask.any())
                        
                        # ok_teacher = (window_mask.size == Ta and not window_mask[1:].any()) \
                        #             or (not pad_tail and end <= T and not window_mask[1:].any())
                    else:
                        ok_teacher = not window_mask.all()  # at least one present
                    
                    # if sum_of_feedback > 20695 and t >= T -3:
                    #     import pdb; pdb.set_trace()

                    if end <= T:
                        # exact window in-range
                        if ok_teacher:
                            starts.append(t)
                    else:
                        # tail window
                        if pad_tail and ok_teacher:
                            starts.append(t)
                            sum_of_padding = sum_of_padding + 1
                        # else drop

                sum_of_feedback += len(starts)
                print("sum_of_feedback: ", sum_of_feedback, " ep: ", ep, " sum_of_padding: ", sum_of_padding, " if_success: ", if_success)
                if not starts:
                    if pbar: pbar.update(1)
                    continue

                # ---- Stream in blocks of starts to limit RAM ----
                for block_beg in range(0, len(starts), chunk_size):
                    block_starts = starts[block_beg:block_beg + chunk_size]
                    n_chunk = len(block_starts)

                    # Prepare destination slices (handle wrap)
                    dst_end = cur_ptr + n_chunk
                    wrap = dst_end > self.max_size
                    first_part = (self.max_size - cur_ptr) if wrap else n_chunk
                    second_part = n_chunk - first_part if wrap else 0

                    # Allocate temp arrays for actions
                    robot_chunk   = np.empty((n_chunk, Ta, act_dim), dtype=dst_robot.dtype)
                    teacher_chunk = np.empty((n_chunk, Ta, act_dim), dtype=dst_teacher.dtype)

                    # Build windows + choose observation index = t+1 (clamped to T-1)
                    # We also build per-field obs selection for this block.
                    obs_idx_list = []
                    for i, t in enumerate(block_starts):
                        end = t + Ta
                        in_range = min(end, T)
                        win_len = in_range - t
                        # fill actions
                        robot_chunk[i, :win_len]   = s_r[t:in_range]
                        teacher_chunk[i, :win_len] = s_t[t:in_range]
                        if win_len < Ta:
                            # pad with the last available action
                            robot_chunk[i, win_len:]   = s_r[in_range - 1]
                            teacher_chunk[i, win_len:] = s_t[in_range - 1]
                        # obs index: t+1 (data_id=1 in your training), clamp to [0, T-1]
                        obs_idx_list.append(min(t + 1, T - 1))

                    obs_idx_arr = np.asarray(obs_idx_list, dtype=np.int64)

                    # Write FIRST part
                    sl = slice(cur_ptr, cur_ptr + first_part)
                    # observations: fancy-index each obs field
                    for k, arr in s_obs.items():
                        dst_obs_dsets[k][sl] = arr[obs_idx_arr[:first_part]]
                    dst_robot[sl]   = robot_chunk[:first_part]
                    dst_teacher[sl] = teacher_chunk[:first_part]

                    # Optional wrap-around write
                    if wrap and second_part:
                        sl2 = slice(0, second_part)
                        for k, arr in s_obs.items():
                            dst_obs_dsets[k][sl2] = arr[obs_idx_arr[first_part:]]
                        dst_robot[sl2]   = robot_chunk[first_part:]
                        dst_teacher[sl2] = teacher_chunk[first_part:]

                    # Update pointers
                    cur_ptr = (cur_ptr + n_chunk) % self.max_size
                    cur_cnt = min(cur_cnt + n_chunk, self.max_size)
                    n_written += n_chunk

                    if self.full():
                        break

                if pbar: pbar.update(1)
                if self.full():
                    break

            if pbar:
                pbar.close()

        # persist updated ptr / count
        self.ptr_ds[...]   = cur_ptr
        self.count_ds[...] = cur_cnt
        self.f.flush()
        return n_written



class Buffer:
    def __init__(self, min_size, max_size):
        self.buffer = []
        self.min_size, self.max_size = min_size, max_size

    def full(self):
        return len(self.buffer) >= self.max_size

    def initialized(self):
        return len(self.buffer) >= self.min_size

    def add(self, step):
        if self.full():
            self.buffer.pop(0)
        self.buffer.append(step)

    def sample(self, batch_size):
        return [random.choice(self.buffer) for _ in range(batch_size)]

    def length(self):
        return len(self.buffer)

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_from_file(self, filename):
        with open(filename, 'rb') as file:
            self.buffer = pickle.load(file)



# class Buffer_uniform_sampling:
#     def __init__(self, min_size, max_size):
#         self.buffer = []
#         self.buffer_temp = []  # Temporary buffer for sampling
#         self.min_size, self.max_size = min_size, max_size

#     def full(self):
#         return len(self.buffer) >= self.max_size

#     def initialized(self):
#         return len(self.buffer) >= self.min_size

#     def add(self, step):
#         if self.full():
#             self.buffer.pop(0)
#         self.buffer.append(step)
#         self.buffer_temp.append(step)  # Keep temp buffer synchronized

#     def sample(self, batch_size):
#         # Check if buffer_temp has enough samples left, else reset it to the full buffer
#         if len(self.buffer_temp) < batch_size:
#             # Reset the temporary buffer and shuffle
#             self.reset_temp_buffer()
        
#         # Sample without replacement
#         batch_indices = random.sample(range(len(self.buffer_temp)), batch_size)
#         batch = [self.buffer_temp[i] for i in batch_indices]

#         # Remove sampled items by indices (in reverse order to avoid index shifting)
#         for i in sorted(batch_indices, reverse=True):
#             del self.buffer_temp[i]
        
#         return batch

#     def length(self):
#         return len(self.buffer)

#     def save_to_file(self, filename):
#         with open(filename, 'wb') as file:
#             pickle.dump(self.buffer, file)

#     def load_from_file(self, filename):
#         with open(filename, 'rb') as file:
#             self.buffer = pickle.load(file)
#             self.buffer_temp = self.buffer.copy()  # Sync temp buffer with loaded buffer

#     def reset_temp_buffer(self):
#         """Clears the temporary buffer used for sampling."""
#         self.buffer_temp = self.buffer.copy()
#         random.shuffle(self.buffer_temp)


class Buffer_uniform_sampling:
    """
    A replay-buffer that lets you draw batches without replacement,
    while keeping memory overhead tiny.

    ── Behaviour ──────────────────────────────────────────────
    • add(step)              – append a new step (drops oldest if full)
    • sample(batch_size)     – uniform batch; every element is visited
                               once before the buffer is reshuffled
    • full(), initialized()  – same semantics as before
    • save_to_file / load_from_file – unchanged
    """
    def __init__(self, min_size: int, max_size: int) -> None:
        self.buffer: list = []                # or deque(maxlen=max_size) if you prefer
        self.min_size, self.max_size = min_size, max_size

        # internal bookkeeping for sampling
        self._indices: list[int] = []         # current permutation of indices
        self._ptr: int = 0                    # cursor into _indices
        self._need_shuffle: bool = True       # flag to reshuffle before next sample

    # ── helpers ──────────────────────────────────────────────
    def _reshuffle(self) -> None:
        """Create a new random permutation of indices."""
        self._indices = list(range(len(self.buffer)))
        random.shuffle(self._indices)
        self._ptr = 0
        self._need_shuffle = False

    # ── public API ───────────────────────────────────────────
    def full(self) -> bool:
        return len(self.buffer) >= self.max_size

    def initialized(self) -> bool:
        return len(self.buffer) >= self.min_size

    def add(self, step) -> None:
        """Add a transition; if full, drop the oldest one."""
        if self.full():
            # drop-oldest: pop(0) is O(n); switch to deque if this is a bottleneck
            self.buffer.pop(0)
        self.buffer.append(step)

        # the buffer changed → reshuffle before the next sample epoch starts
        self._need_shuffle = True

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer currently holds {len(self.buffer)} elements, "
                f"batch_size={batch_size} is too large."
            )

        # prepare permutation if needed, or if current epoch is almost exhausted
        if self._need_shuffle or self._ptr + batch_size > len(self._indices):
            self._reshuffle()

        # slice the next chunk of indices and advance the cursor
        start, end = self._ptr, self._ptr + batch_size
        self._ptr = end
        idx_batch = self._indices[start:end]

        return [self.buffer[i] for i in idx_batch]

    def length(self) -> int:
        return len(self.buffer)

    # ── persistence ─────────────────────────────────────────
    def save_to_file(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_from_file(self, filename: str) -> None:

        # used to obtain the src path instead of hydra's temporary run directory
        from hydra.utils import get_original_cwd
        project_root = get_original_cwd()
        # import pdb; pdb.set_trace()
        full_path = os.path.join(project_root, filename)

        # sanity‐check (optional):
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Cannot find pickle file: {full_path!r}")
        with open(full_path, "rb") as f:
            self.buffer = pickle.load(f)

        # force a reshuffle so the first sample after loading behaves correctly
        self._need_shuffle = True

    
    def load_from_h5_buffer_file(self, filename: str) -> None:
        """
        Load transitions from an HDF5Buffer-formatted .h5 file into self.buffer.

        Each loaded item is a tuple: (obs_dict, action_teacher, action_robot).

        If the file contains more transitions than this buffer's max_size,
        only the most recent self.max_size transitions are kept (based on the
        circular buffer's ptr/count metadata).
        """
        # Resolve path similarly to the pickle loader
        # from hydra.utils import get_original_cwd
        # project_root = get_original_cwd()
        # full_path = os.path.join(project_root, filename)
        # if not os.path.exists(full_path):
        #     full_path = filename
        # if not os.path.exists(full_path):
        #     raise FileNotFoundError(f"Cannot find HDF5 buffer file: {filename!r}")

        full_path = filename
        with h5py.File(full_path, "r") as f:
            # Basic validation
            required_keys = {"robot_action", "teacher_action", "ptr", "count"}
            if not required_keys.issubset(set(f.keys())):
                missing = required_keys.difference(set(f.keys()))
                raise ValueError(
                    f"HDF5 file {full_path!r} is missing required datasets: {sorted(missing)}"
                )

            d_robot = f["robot_action"]
            d_teacher = f["teacher_action"]
            file_max_size = int(d_robot.shape[0])  # same length for all per-design
            count = int(f["count"][()])
            ptr = int(f["ptr"][()])

            # Nothing to load
            n_available = min(count, file_max_size)
            if n_available <= 0:
                self.buffer = []
                self._need_shuffle = True
                return

            # Determine chronological indices (oldest → newest) in the ring buffer
            if n_available < file_max_size:
                # Not full yet: valid data are at [0, n_available)
                chronological = list(range(n_available))
            else:
                # Full: oldest starts at ptr, then wraps
                chronological = list(range(ptr, file_max_size)) + list(range(0, ptr))

            # Keep only the most recent up to this buffer's capacity
            n_keep = min(n_available, self.max_size)
            indices = chronological[-n_keep:]

            # Pre-bind observation datasets (everything except actions and metadata)
            obs_names = [
                k for k in f.keys()
                if k not in ("robot_action", "teacher_action", "ptr", "count")
            ]
            obs_dsets = {k: f[k] for k in obs_names}

            loaded = []
            for i in indices:
                # Build obs dict lazily per index
                obs_dict = {k: obs_dsets[k][i] for k in obs_names}
                action_robot = d_robot[i][()]      # shape: (A,) or (Ta, A)
                action_teacher = d_teacher[i][()]  # shape: (A,) or (Ta, A)
                # Match your tuple convention used elsewhere
                loaded.append((obs_dict, action_teacher, action_robot))

            self.buffer = loaded
            # Ensure next sample() reshuffles
            self._need_shuffle = True



class Buffer_HGDagger_CLICdata:
    def __init__(self, min_size, max_size):
        self.buffer = []
        self.min_size, self.max_size = min_size, max_size

    def full(self):
        return len(self.buffer) >= self.max_size

    def initialized(self):
        return len(self.buffer) >= self.min_size

    def add(self, step):
        if self.full():
            self.buffer.pop(0)
        self.buffer.append(step)

    def sample(self, batch_size):
        import random
        return [random.choice(self.buffer) for _ in range(batch_size)]

    def length(self):
        return len(self.buffer)

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_from_file(self, filename):
        with open(filename, 'rb') as file:
            loaded_buffer = pickle.load(file)
        # Modify each entry from [s, h, a] to [s, h+a]
            
        print("Modified Buffer:")
        # for item in loaded_buffer:
        #     print(item)
        # modified_buffer = [[s, h + a] for s, h, a in loaded_buffer]
        modified_buffer = []
        for s, h, a in loaded_buffer:
            # Reshape 'h' and 'a' to ensure they have a shape of (3,) before adding
            h_reshaped = tf.reshape(h, (3,))
            a_reshaped = tf.reshape(a, (3,))
            # Add 'h' and 'a'
            combined = h_reshaped + a_reshaped
            modified_buffer.append([s, combined])
        self.buffer = modified_buffer
        # print("Modified Buffer:")
        # for item in modified_buffer:
        #     print(item)


# New class inheriting from Buffer
class velocity_control_Buffer(Buffer):
    def get_action_state_pair_with_highest_combined_norm(self):
        # Ensure there are action-state pairs in the buffer
        if not self.initialized():
            return None, None
        # Calculate the combined norm for each action (translation + orientation) and return the pair with the highest combined norm
        # highest_combined_norm_pair = max(
        #     self.buffer,
        #     key=lambda pair: np.linalg.norm(pair[0][0:3]) + np.linalg.norm(pair[0][3:6]))
        highest_combined_norm_pair = max(
            self.buffer,
            key=lambda pair: np.linalg.norm(pair[1][3:6]))
        print("highest_combined_norm_pair: ", highest_combined_norm_pair)
        return highest_combined_norm_pair


class Buffer_IWR:
    def __init__(self, min_size, max_size):
        self.buffer = []
        self.buffer_no_intervention = []
        self.min_size, self.max_size = min_size, max_size

    def buffer_full(self):
        return len(self.buffer) >= self.max_size
    
    def buffer_no_intervention_full(self):
        return len(self.buffer_no_intervention) >= self.max_size

    def initialized(self):
        return self.length() >= self.min_size

    def add(self, step, intervention_flag = False):
        if self.buffer_full():
            self.buffer.pop(0)
        if self.buffer_no_intervention_full():
            self.buffer_no_intervention.pop(0)
        if intervention_flag:
            self.buffer.append(step)
        else:
            self.buffer_no_intervention.append(step)

    def sample_buffer(self, batch_size):
        return [random.choice(self.buffer) for _ in range(batch_size)]
    
    def sample_buffer_no_intervention(self, batch_size):
        return [random.choice(self.buffer_no_intervention) for _ in range(batch_size)]

    # def sample(self, batch_size):
    #     # Calculate the total number of samples in both buffers
    #     total_samples = len(self.buffer) + len(self.buffer_no_intervention)
    #     print("len(self.buffer): ", len(self.buffer), " len(self.buffer_no_intervention): ", len(self.buffer_no_intervention))
    #     # If there are no samples to sample from, return an empty list
    #     if total_samples == 0:
    #         return []
        
    #     # Calculate the number of samples to draw from each buffer based on their proportions
    #     num_samples_intervention = int(round((len(self.buffer_no_intervention) / total_samples) * batch_size))
    #     num_samples_no_intervention = batch_size - num_samples_intervention  # Ensure the total adds up to batch_size
        
    #     # [To fix]


    #     if num_samples_intervention > len(self.buffer):
    #         num_samples_intervention = len(self.buffer)
    #         num_samples_no_intervention = batch_size - num_samples_intervention 
    #     if num_samples_no_intervention > len(self.buffer_no_intervention):
    #         num_samples_no_intervention = len(self.buffer_no_intervention)
    #         num_samples_intervention = batch_size - num_samples_no_intervention

    #     # Sample from each buffer
    #     intervention_sampled = [random.choice(self.buffer) for _ in range(num_samples_intervention)] if self.buffer else []
    #     no_intervention_sampled = [random.choice(self.buffer_no_intervention) for _ in range(num_samples_no_intervention)] if self.buffer_no_intervention else []
        
    #     # Combine the samples from both buffers
    #     samples = intervention_sampled + no_intervention_sampled

    #     print("num_samples_intervention: ", num_samples_intervention, " num_samples_no_intervention: ", num_samples_no_intervention)
        
    #     # Shuffle the combined list to mix intervention and no-intervention samples
    #     random.shuffle(samples)
        
    #     return samples

    def sample(self, batch_size):
        # Calculate the total number of samples in both buffers
        total_samples = len(self.buffer) + len(self.buffer_no_intervention)
        print("len(self.buffer): ", len(self.buffer), " len(self.buffer_no_intervention): ", len(self.buffer_no_intervention))
        # If there are no samples to sample from, return an empty list
        if total_samples == 0:
            return []
        
        # Calculate the number of samples to draw from each buffer based on their proportions
        # num_samples_intervention = int(batch_size/2)
        # num_samples_no_intervention = batch_size - num_samples_intervention  # Ensure the total adds up to batch_size
        
        num_samples_intervention = int(round((len(self.buffer) / total_samples) * batch_size))
        # if intervention data is more than non-intervention data, do not reweight
        if num_samples_intervention < int(batch_size/2):
            num_samples_intervention = int(batch_size/2)
        num_samples_no_intervention = batch_size - num_samples_intervention  # Ensure the total adds up to batch_size
        
        if num_samples_intervention > len(self.buffer):
            num_samples_intervention = len(self.buffer)
            num_samples_no_intervention = batch_size - num_samples_intervention 
        if num_samples_no_intervention > len(self.buffer_no_intervention):
            num_samples_no_intervention = len(self.buffer_no_intervention)
            num_samples_intervention = batch_size - num_samples_no_intervention

        # Sample from each buffer
        intervention_sampled = [random.choice(self.buffer) for _ in range(num_samples_intervention)] if self.buffer else []
        no_intervention_sampled = [random.choice(self.buffer_no_intervention) for _ in range(num_samples_no_intervention)] if self.buffer_no_intervention else []
        
        # Combine the samples from both buffers
        samples = intervention_sampled + no_intervention_sampled

        print("num_samples_intervention: ", num_samples_intervention, " num_samples_no_intervention: ", num_samples_no_intervention)
        
        # Shuffle the combined list to mix intervention and no-intervention samples
        random.shuffle(samples)
        
        return samples
    
    # def sample(self, batch_size):
    #     # Calculate the total number of samples in both buffers
    #     total_samples = len(self.buffer) + len(self.buffer_no_intervention)
    #     print("len(self.buffer): ", len(self.buffer), " len(self.buffer_no_intervention): ", len(self.buffer_no_intervention))
    #     # If there are no samples to sample from, return an empty list
    #     if total_samples == 0:
    #         return []
        
    #     # Calculate the number of samples to draw from each buffer based on their proportions
    #     num_samples_intervention = int(batch_size/2)
    #     num_samples_no_intervention = batch_size - num_samples_intervention  # Ensure the total adds up to batch_size
        
    #     # [To fix]

        
    #     if num_samples_intervention > len(self.buffer):
    #         num_samples_intervention = len(self.buffer)
    #         num_samples_no_intervention = batch_size - num_samples_intervention 
    #     if num_samples_no_intervention > len(self.buffer_no_intervention):
    #         num_samples_no_intervention = len(self.buffer_no_intervention)
    #         num_samples_intervention = batch_size - num_samples_no_intervention

    #     # Sample from each buffer
    #     intervention_sampled = [random.choice(self.buffer) for _ in range(num_samples_intervention)] if self.buffer else []
    #     no_intervention_sampled = [random.choice(self.buffer_no_intervention) for _ in range(num_samples_no_intervention)] if self.buffer_no_intervention else []
        
    #     # Combine the samples from both buffers
    #     samples = intervention_sampled + no_intervention_sampled

    #     print("num_samples_intervention: ", num_samples_intervention, " num_samples_no_intervention: ", num_samples_no_intervention)
        
    #     # Shuffle the combined list to mix intervention and no-intervention samples
    #     random.shuffle(samples)
        
    #     return samples

    def length(self):
        # return len(self.buffer)
        return len(self.buffer) + len(self.buffer_no_intervention)

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_from_file(self, filename):
        with open(filename, 'rb') as file:
            self.buffer = pickle.load(file)


    


class Buffer_GroupedState(Buffer):
    def __init__(self, min_size, max_size, similarity_threshold):
        super().__init__(min_size, max_size)
        self.grouped_data = {}
        self.state_id_map = {}  # Map to keep track of state IDs
        self.next_state_id = 0  # Counter for the next state ID
        self.similarity_threshold = similarity_threshold

    def is_similar(self, state1, state2):  # absolute difference
        distance = np.linalg.norm(np.array(state1) - np.array(state2))
        return distance <= self.similarity_threshold
    
    # def is_similar(self, state1, state2):  # for metaworld's hammer task, leverage the relative difference 
    #     # "hand_pos": obs[:3],
    #     # "gripper": obs[3],
    #     # "hammer_pos": obs[4:7],
    #     # "unused_info": obs[7:],
    #     # goal_pos: np.array([0.24, 0.71, 0.11]) + np.array([-0.19, 0.0, 0.05])
      
    #     # Extract relevant positions from state1
    #     hand_pos_1 = np.array(state1)[0, :3]
    #     hammer_pos_1 = np.array(state1)[0, 4:7]
    #     goal_pos_1 = np.array([0.24, 0.71, 0.11]) + np.array([-0.19, 0.0, 0.05])  # Assuming goal position is constant

    #     # Extract relevant positions from state2
    #     hand_pos_2 = np.array(state2)[0, :3]
    #     hammer_pos_2 = np.array(state2)[0, 4:7]
    #     goal_pos_2 = np.array([0.24, 0.71, 0.11]) + np.array([-0.19, 0.0, 0.05])  # Assuming goal position is constant

    #     # Compute relative distances

    #     # print("shape of hand_pose: ", hand_pos_1.shape)
    #     rel_dist_hand_hammer_1 = np.linalg.norm(hand_pos_1 - hammer_pos_1)
    #     rel_dist_hand_goal_1 = np.linalg.norm(hand_pos_1 - goal_pos_1)
    #     rel_dist_hammer_goal_1 = np.linalg.norm(hammer_pos_1 - goal_pos_1)

    #     rel_dist_hand_hammer_2 = np.linalg.norm(hand_pos_2 - hammer_pos_2)
    #     rel_dist_hand_goal_2 = np.linalg.norm(hand_pos_2 - goal_pos_2)
    #     rel_dist_hammer_goal_2 = np.linalg.norm(hammer_pos_2 - goal_pos_2)

    #     # Compute the differences between the relative distances of state1 and state2
    #     diff_hand_hammer = np.abs(rel_dist_hand_hammer_1 - rel_dist_hand_hammer_2)
    #     diff_hand_goal = np.abs(rel_dist_hand_goal_1 - rel_dist_hand_goal_2)
    #     diff_hammer_goal = np.abs(rel_dist_hammer_goal_1 - rel_dist_hammer_goal_2)

    #     # Use a combined metric for similarity, could be a simple sum or a more complex function
    #     total_difference = diff_hand_hammer + diff_hand_goal + diff_hammer_goal

    #     return total_difference <= self.similarity_threshold

    def get_state_id_into_buffer(self, state):
        state_id = self.get_state_id(state)
        if state_id is None:
            # No similar state found, assign new ID
            new_state_id = self.next_state_id
            self.next_state_id += 1
            self.state_id_map[new_state_id] = state
            return new_state_id
        else:
            return state_id

    def get_state_id(self, state):
        if isinstance(state, tf.Tensor):
            state_np = tuple(state.numpy())
        for state_id, existing_state in self.state_id_map.items():
            if self.is_similar(state_np, existing_state):
                return state_id
        return None
    
    def number_of_groups(self):
        return len(self.grouped_data)

    def add(self, step):
        state, action, corrected_action = step
        state_id = self.get_state_id_into_buffer(state)

        super().add(step)  # Call the add method of the base class

        # Add to grouped data
        if state_id not in self.grouped_data:
            self.grouped_data[state_id] = []
        # Convert TensorFlow tensors to NumPy arrays
        if isinstance(state, tf.Tensor):
            state = state.numpy()
        if isinstance(action, tf.Tensor):
            action = action.numpy()
        if isinstance(corrected_action, tf.Tensor):
            corrected_action = corrected_action.numpy()
        self.grouped_data[state_id].append((state, action, corrected_action))
        # self.grouped_data[state_id].append(step)
        print("size of buffer: ", self.length())
        print("size of grouped buffer: ", self.number_of_groups())

    def get_this_state_feedback(self, state): # return the feedback data in this state
        state_id = self.get_state_id(state)
        if state_id is None:
            return None
        else:
            return self.grouped_data[state_id]
        
    def find_k_nearest_states(self, state, k=1):
        if isinstance(state, tf.Tensor):
            state_np = tuple(state.numpy())
        else:
            state_np = state

        distances = []
        for state_id, existing_state in self.state_id_map.items():
            distance = np.linalg.norm(np.array(state_np) - np.array(existing_state))
            # [!!] The k nearest states can also be quite far away from the given state (i.e. a new state haven't been seen before)
            #  Add a threhould to filter the states that are far away
            if distance < 0.2:
                distances.append((state_id, distance))

        # Sort the states by distance
        distances.sort(key=lambda x: x[1])

        # Select the k nearest states
        nearest_states_ids = [distances[i][0] for i in range(min(k, len(distances)))]

        # Retrieve the corresponding state data
        nearest_states_data = [self.grouped_data[state_id] for state_id in nearest_states_ids]

        return nearest_states_ids, nearest_states_data

    # def find_optimal_n(self, k_nearest_states_id, k_nearest_states_data):
    #     # Initialize a list to hold the state ID and its corresponding conflict ratio
    #     conflict_ratios = []
        
    #     grouped_state_action_data = []
    #     optimal_n = 0
    #     grouped_state_action_data_n = []

    #     policy_oracle = SawyerHammerV2Policy()
        
    #     # Iterate through each of the k nearest states
    #     for i in range(len(k_nearest_states_data)):
    #         state_id = k_nearest_states_id[i]
    #         state_data = k_nearest_states_data[i]
    #         # [(state, action, corrective_feedback), ...]
    #         grouped_state_action_data = state_data + grouped_state_action_data
    #         # Calculate the number of conflicted corrective feedbacks and total feedbacks
    #         conflicted_feedback_number = self.calculate_conflicted_number(grouped_state_action_data)

    #         # print("state_data[0][0]: ", state_data[0][0].shape)
    #         action_teacher_i = policy_oracle.get_action(np.squeeze(state_data[0][0]))
    #         action_max = np.max(abs(action_teacher_i))
    #         if action_max > 1:
    #             action_teacher_i =  action_teacher_i / action_max
            
    #         total_feedback = len(grouped_state_action_data)  # Assuming state_data is a list of feedbacks for the state
    #         # Calculate the conflict ratio
    #         if total_feedback > 0:
    #             conflict_ratio = conflicted_feedback_number / total_feedback
    #         else:
    #             conflict_ratio = 0  # Avoid division by zero
            
    #         # Append the state ID and its conflict ratio to the list
    #         conflict_ratios.append((state_id, conflict_ratio))
    #         formatted_ratio = "{:.2f}".format(conflict_ratio)
    #         print("i : ", i, " conflicted number: ", conflicted_feedback_number, " total_feedback: ", total_feedback, " conflict_ratio: ", formatted_ratio, " action_teacher_i: ",action_teacher_i)
    #         if conflict_ratio < 0.001:
    #             optimal_n = i
    #             grouped_state_action_data_n = grouped_state_action_data

    #     # Sort the list by conflict ratio in ascending order
    #     # conflict_ratios.sort(key=lambda x: x[1])

    #     # # Find the maximum n with the minimum conflict ratio
    #     # # This can be as simple as taking the first n elements from the sorted list
    #     # # Or you might have more complex criteria based on your specific needs
    #     # optimal_n = 0
    #     # for i in range(len(conflict_ratios)):
    #     #     # Define your criteria for selecting n here
    #     #     # For example, you might check if the conflict ratio meets a certain threshold
    #     #     # and increment optimal_n if it does
    #     return optimal_n, grouped_state_action_data_n

    def calculate_conflicted_number(self, state_action_data):
        conflicted_number = 0
        for i in range(len(state_action_data)):
            action_i = state_action_data[i][1]
            feedback_i = state_action_data[i][2] 
            for j in range(i + 1, len(state_action_data)):
                # check whether i-th corrective feedback conflict with j-th 
                action_j = state_action_data[j][1]
                feedback_j = state_action_data[j][2]
                flag_action_i_violates_j = self.check_inside_feasible_space_of_thsi_corrective_feedback(action_i, feedback_i, action_j, feedback_j)
                flag_action_j_violates_i = self.check_inside_feasible_space_of_thsi_corrective_feedback(action_j, feedback_j, action_i, feedback_i)
                if not flag_action_j_violates_i and not flag_action_i_violates_j:
                    conflicted_number = conflicted_number + 1

        return conflicted_number
    
    
    def check_inside_feasible_space_of_thsi_corrective_feedback(self, action_check, action_direction_check, feedback_action, feedback_corrective_direction):
        a_minus = feedback_action
        sphere_gamma = 0.5  # use 1, instead of small value such as 0.1
        e = 1
        a_plus = a_minus + sphere_gamma * e * feedback_corrective_direction  
        action_check_plus = action_check + sphere_gamma * e * action_direction_check
        # print("size of human_correction: ", human_correction.shape, " size of action_i", action_i.shape )
        action_feasible_flag = True if np.matmul(feedback_corrective_direction, (action_check_plus - 0.5 * a_minus - 0.5 * a_plus).T) >= 0 else False
        return action_feasible_flag
        
    