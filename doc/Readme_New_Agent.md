
# How to Add a New Agent

This guide provides instructions for integrating a new learning algorithm into this repository. Follow these steps to ensure your custom agent is compatible with the existing framework.

-----

## AGENT CLASS STRUCTURE

Your new agent should be defined as a Python class. The `__init__` method is the place to set up all the necessary components of your agent, such as the policy network, optimizer, and replay buffer.

### Key Components to Initialize:

  * **Model and Optimizer**: Define your neural network(s) and the corresponding optimizer(s). For an example using PyTorch, see the `CLIC` agent in `CLIC_torch.py`. For a TensorFlow/Keras example, refer to the `HG_DAGGER` agent in `HG_DAgger.py`.
  * **Replay Buffer**: Initialize a buffer to store interaction data. The buffer should have a defined minimum and maximum size.
  * **Hyperparameters**: Store all agent-specific parameters, such as learning rates, buffer sampling rates, and dimensions of action and observation spaces (`dim_a`, `dim_o`).

-----

## REQUIRED METHODS

Your agent class must implement the following methods to interact with the environment and the training loop.

### 1\. `action(observation)`

This method is responsible for selecting an action given the current observation from the environment.

  * **Input**: `observation` (the current state from the environment).
  * **Output**: An `action` to be executed.
  * **Example (`CLIC_torch.py`)**: The `CLIC` agent uses a Q-function to sample a set of candidate actions and selects the best one.
  * **Example (`HG_DAgger.py`)**: The `HG_DAGGER` agent performs a forward pass through its policy network to determine the action.

<!-- end list -->

```python
# In your agent class
def action(self, observation):
    # Convert observation to the appropriate tensor format
    # Perform a forward pass through your policy network
    # a = self.policy_model(processed_observation)
    # Return the final action as a NumPy array
    return final_action
```

### 2\. Training and Feedback Method

This method handles the agent's learning process. It is called at each step of the environment interaction. A good name for this function would be descriptive, like `train` or `update_policy`. In the provided examples, this functionality is within the `TRAIN_Q_value` method for the `CLIC` agent and `TRAIN_Policy` for `HG-DAgger`.

This function should perform three main tasks:

**A. Store New Data**
If new feedback (e.g., a human correction `h`) is available, add the relevant data tuple (e.g., `(observation, action, h, next_observation)`) to the replay buffer.

**B. Online Training (Optional)**
The agent can be configured to perform a training step at a fixed interval (e.g., every `self.buffer_sampling_rate` steps). This involves sampling a mini-batch from the buffer and running the loss calculation.

**C. End-of-Episode Training**
At the end of each episode, the agent should train for a specified number of iterations (`self.number_training_iterations`) to ensure the policy improves based on the collected data.

```python
# In your agent class
def train(self, action, h, observation, next_observation, t, done):
    # 1. Store data if feedback is provided
    if h is not None:
        self.buffer.add((observation, action, h, next_observation))
        # Optionally, perform an immediate training step
        if self.buffer.initialized():
            batch = self.buffer.sample(self.buffer_sampling_size)
            self.update_networks(batch)

    # 2. Train periodically
    if self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
        batch = self.buffer.sample(self.buffer_sampling_size)
        self.update_networks(batch)

    # 3. Train at the end of the episode
    if done and self.train_end_episode and self.buffer.initialized():
        for _ in range(self.number_training_iterations):
            batch = self.buffer.sample(self.buffer_sampling_size)
            self.update_networks(batch)

```

### 3\. Loss Calculation and Network Update

Create a method to handle the actual network updates. This function will be called by your main training method. It should:

  * Sample a batch of data from the buffer.
  * Calculate the loss based on your algorithm's objective function.
  * Perform backpropagation and update the network weights.

The `CLIC` agent, for example, has an `action_value_batch_update` method that computes a complex loss based on a desired action space, while the `HG_DAGGER` agent uses a simpler supervised learning approach in its `_batch_update` method.

```python
# In your agent class
def update_networks(self, batch):
    # Unpack the batch into states, actions, etc.
    # Convert data to tensors
    
    # Forward pass and loss calculation
    # e.g., predicted_actions = self.policy_model(states)
    # loss = self.loss_function(predicted_actions, true_actions)
    
    # Backpropagation
    # self.optimizer.zero_grad()
    # loss.backward()
    # self.optimizer.step()
```

-----

## INTEGRATION

After creating your agent class, you need to integrate it into the project.

### 1\. Add to `agent_selector.py`

Add a new condition in the `agent_selector` function to instantiate your agent. This is where you will pass the necessary configuration parameters to your agent's `__init__` method.

```python
# In agent_selector.py

def agent_selector(agent_type: str, config_agent: DictConfig):
    # ... other agents

    elif agent_type == 'Your_New_Agent':
        from agents.your_new_agent_file import YourNewAgentClass
        return YourNewAgentClass(
            dim_a=config_agent.dim_a,
            dim_o=config_agent.dim_o,
            # ... other parameters from the config
        )
```

### 2\. Create a Configuration File

Create a new YAML configuration file (e.g., `your_new_agent_config.yaml`) in the `config/` directory. This file will define all the hyperparameters for your agent, which will be loaded and passed to your agent by the `agent_selector`.

-----

By following these steps and using the provided `CLIC` and `HG-DAgger` agents as a reference, you can successfully integrate your own algorithms into the framework.