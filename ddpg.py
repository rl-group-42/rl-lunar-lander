"""
This is a PyTorch implementation of the Open AI's Deep Deterministic 
Policy Gradient (DDPG) algorithm for continuous space.

Below follows the procedure to implement DDPG using OpenAI's Spinning Up documentation:
https://spinningup.openai.com/en/latest/algorithms/ddpg.html
"""

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.backends import mps
import gymnasium as gym
import numpy as np
from collections import deque

# Use GPU if available for faster processing otherwise use CPU
def _get_torch_device():
    if cuda.is_available():
        return "cuda"
    elif mps.is_available():
        return "mps"
    else:
        return "cpu"

class ReplayMemory():
    """
    A replay buffer for storing experiences by a DDPG agent interacting with the environment.
    Agent can sample experiences from this buffer to improve learning.
    params:
        :max_mem_size: The maximum number of experiences the buffer can hold (default=1000000).
    """
    def __init__(self, max_mem_size=1000000):
        self._max_mem_size = max_mem_size
        # Initialize deque with maximum size
        self._state = deque(maxlen=max_mem_size)
        self._action = deque(maxlen=max_mem_size)
        self._new_state = deque(maxlen=max_mem_size)
        self._reward = deque(maxlen=max_mem_size)
        self._terminal = deque(maxlen=max_mem_size)
        self._mem_ctr = 0

    def _store_transition(self, state, action, new_state, reward, terminal):
        """Store experiences (s, a, s', r, t) from an environment interaction in deques"""
        self._state.append(state)
        self._action.append(action)
        self._new_state.append(new_state)
        self._reward.append(reward)
        self._terminal.append(terminal)

    def _sample_transition(self, batch_size):
        """
        Get a batch of experiences already stored in buffer given a batch size
        """
        # Return all experiences if memory size is smaller than batch size
        if len(self._state) < batch_size:
            batch_size = len(self._state)

        # Otherwise, convert deque to a list and sample a batch of experiences
        index_perm = np.random.permutation(np.arange(len(self._state)))[:batch_size]
        state = np.array(list(self._state))[index_perm]
        action = np.array(list(self._action))[index_perm]
        new_state = np.array(list(self._new_state))[index_perm]
        reward = np.array(list(self._reward))[index_perm]
        terminal = np.array(list(self._terminal))[index_perm]
        return state, action, new_state, reward, terminal

class Noise():
    """
    A noise function which encourages the agent to explore the environment in a 
    continous action space.
    The Ornstein-Uhlenbeck process is used as the noise function below:
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    params:
        :action_dim: number of actions agent can take in environment
        :theta, dt, sigma: hyperparameters which can be tune to encourage exploration
    """
    def __init__(self, action_dim, theta=0.2, dt=1e-2, sigma=0.2):
        self._action_dim = action_dim
        self._mu = np.zeros(self._action_dim) 
        self._theta = theta
        self._sigma = sigma
        self._dt = dt
        self.reset()

    def reset(self):
        """Reset to generate different noisy actions"""
        self._state = np.zeros(self._action_dim)

    def OUnoise(self):
        """OU noise formula: dxt = theta (mu - X) + sigma(dWt)"""
        dx = self._theta * (self._mu - self._state) * self._dt + self._sigma  * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        self._state += dx
        return self._state
        
    def __call__(self):
        return self.OUnoise()

class Network(nn.Module):
    """
    Builds a neural network. For DDPG specifically, it needs 4 networks for the critic, actor 
    and their target networks. By default there are two hidden layers and one output layer. 
    The network can be adjusted to what ever network configuration is needed.
    params:
        :input_dim: The size of the input to the neural network
        :output_dim: The size of the output to the neural network
        :output_activation: activation functions which can be chosen for the output of the network
        :hidden_dims: The number of nodes and how many hiden layers defined as a tuple
        :hidden_activations: activation functions which can be chosen for the hidden layers of the network
        :learning_rate: hyperparamter to control the learning of the network
    """
    def __init__(self, input_dim, output_dim, output_activations=tuple(), hidden_dims=(32, 32), hidden_activations=None, learning_rate=0.01):
        super(Network, self).__init__()

        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim

        # Default activation function is ReLU for hidden layers
        hidden_activations = [nn.ReLU() for _ in hidden_dims] if hidden_activations is None else hidden_activations

        # Create Sequential hidden layers given activation functions and hidden layer size
        layers = []
        prev_dim = input_dim
        for dim, activation in zip(hidden_dims, hidden_activations):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation)
            prev_dim = dim
        self._hidden_layers = nn.Sequential(*layers)

        # Define Sequential output layers given activation functions and hidden layer size
        self._output_layers = nn.ModuleList()
        if output_activations:
            for activation in output_activations:
                self._output_layers.append(
                    nn.Sequential(
                        nn.Linear(prev_dim, output_dim),
                        activation,
                        )
                )
        else:
            self._output_layers.append(nn.Linear(prev_dim, 1))
        # Adam Optimizer to optimize network's parameters
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """Pass input through network and return the network output"""
        x = self._hidden_layers(x)
        outputs = list(output_layer(x) for output_layer in self._output_layers)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

class Agent():
    """
    This class is the bulk of the DDPG algorithm implementation for the agent. The main agent evnronment 
    interaction is the _get_action, _store_transition, and the _learn function. Only the state and action
    is required for the agent to function.
    params:
        :env: OpenAi's environment object
        :continuity: A flag to set the environment to discrete or continuous action space
        :train_iteration: The number of times the agent will train
        :episodes: The number of iteractions with the environment
        :epoches: a weight which determines the size of update_epoches
        :optimal_run_episodes: The number of environment interactions when testing
        :render_episode: flag which states whether episode should be rendered
        :measure_after_iteration: indicates when to start testing during training
        :measurement: flag which states whether to test or not
        :gamma: The discount factor
        :tau: hyperparameter which control size of network's weights updates
        :learning_rate: hyperparamter to control the learning of the network
        :theta, dt, sigma: hyperparameters which can be tune to encourage exploration
        :batch_size: number of experiences to sample from memory
        :memory_size: The maximum number of experiences the replay memory can hold
        :update_epoches: The number of updates performed to networks when learning
    """
    def __init__(self, env, continuity=True, train_iterations=100000, episodes=4000, epoches=100,
                 optimal_run_episodes=5, render_episode=-1, measure_after_iterations=1, measurement=True,
                 gamma=0.99, tau=0.995, learning_rate=(0.001,),
                 batch_size=64, memory_size=1000, update_epoches=10,
                 theta=0.2, dt=1e-2, sigma=0.2):

        # Environment
        self._env = env
        self._state_dims = env.observation_space.shape[0]
        self._action_dims = env.action_space.shape[0]
        self._min_bounds = env.action_space.low
        self._max_bounds = env.action_space.high

        self._continuity = continuity
        self._train_iterations = train_iterations
        self._episodes = episodes
        self._epoches = epoches

        self._optimal_run_episodes = optimal_run_episodes
        self._render_episode = render_episode
        self._measure_after_iterations = measure_after_iterations
        self._measurement = measurement
        self._updates_number = 0

        self._gamma = gamma
        self._tau = tau
        self._learning_rate = learning_rate

        self._batch_size = batch_size
        self._update_epoches = update_epoches
        self._memory = ReplayMemory(memory_size)

        # Ornstein Uhlenbeck noise
        self.noise = Noise(self._action_dims, theta, dt, sigma)

        # Activation Functions
        activations = (nn.Tanh(),)

        # Torch device: GPU or CPU
        self._device = T.device(_get_torch_device())

        # Hidden layers
        hidden_layers = (32, 32)

        # 4 neural networks. Actor,critics, and their target networks
        self._actor = Network(self._state_dims, self._action_dims, activations, 
                              hidden_dims=hidden_layers, learning_rate=self._learning_rate[0]).to(self._device)
        self._critic = Network(self._state_dims + self._action_dims, 1, 
                               hidden_dims=hidden_layers, learning_rate=learning_rate[1]).to(self._device)
        self._target_actor = Network(self._state_dims, self._action_dims, activations, 
                                     hidden_dims=hidden_layers, learning_rate=self._learning_rate[2]).to(self._device)
        self._target_critic = Network(self._state_dims + self._action_dims, 1, 
                                      hidden_dims=hidden_layers, learning_rate=learning_rate[3]).to(self._device)

        # Explicitly set networks to training mode.
        #https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train
        self._actor.train()
        self._critic.train()
        self._target_actor.train()
        self._target_critic.train()

    def _get_numpy(self, tensor):
        """
        Helper function which converts tensors to numpy. This will help to add noise
        to action in continous space.
        """
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    def _get_tensor(self, nparray, grad=True):
        """
        Helper function which converts numpy to tensors. This will help to input 
        states and actions to neural networks.
        """
        return T.from_numpy(nparray).float().to(self._device).requires_grad_(grad)

    def _get_action(self, state, optimal=False):
        """
        Get an action by passing the state to the actor network. The action will be
        clipped and noise will be added.
        """
        self._actor.eval()
        state_tensor = self._get_tensor(np.array([state]), False).detach()

        # Deactivate gradient computation. Action does not need to be tracked for gradient updates
        with T.no_grad():
            action = self._actor(state_tensor)

        action = self._get_numpy(action)[0]

        if not optimal:
            #Add noise an clip action
            action = np.clip((action + self.noise()), self._min_bounds, self._max_bounds)

        self._actor.train()
        return action

    def _store_transition(self, state, action, new_state, reward, terminal):
        """
        Store experience from agent interaction with environment to replay memory
        """
        self._memory._store_transition(state, action, new_state, reward, terminal)

    def _gradient_descent_critic(self, state, action, new_state, reward, terminal):
        """
        Compute the gradent descent update to the critic networks. To later select actions from actor which moves
        closer to the target. This will later lead to the optimal policy
        params:
            :state, action, new_state, reward, terminal: experienced gained from agent interaction
        """
        # Convert experience to tensors to later pass through network
        action_tensor = self._get_tensor(action)
        state_tensor = self._get_tensor(state)
        new_state_tensor = self._get_tensor(new_state)
        terminal_tensor = self._get_tensor(terminal)
        reward_tensor = self._get_tensor(reward)

        # Compute the target value by passing the target_actions and the next_state
        # to the target critic networks
        target_actions = self._target_actor(new_state_tensor.detach())
        target_critic_value = self._target_critic(T.cat((new_state_tensor.detach(), target_actions.detach()), 1)).flatten()
        target = reward_tensor + self._gamma * target_critic_value * (1 - terminal_tensor)

        # Compute critic value to then compute the Mean Squared Error between
        # the critic value and target value
        critic_value = self._critic(T.cat((state_tensor, action_tensor), 1)).flatten()

        # MSE Loss function
        mse = T.nn.MSELoss()
        critic_loss = mse(critic_value, target)

        # Gradient step for critic network
        self._critic.optimizer.zero_grad()
        critic_loss.backward()
        self._critic.optimizer.step()

    def _gradient_ascent_actor(self, state):
        # Convert to tensor
        state_tensor = self._get_tensor(state)
        action_tensor = self._actor(state_tensor.detach())

        # Critic value
        critic_value = self._critic(T.cat((state_tensor, action_tensor), 1)).flatten()

        # Mean Loss
        actor_loss = T.mean(-critic_value)

        # Gradient step
        self._actor.optimizer.zero_grad()
        actor_loss.backward()
        self._actor.optimizer.step()

    def _update_target_actor_weights(self):        
        """
        Update network weights for the actor
        w' <- tau*w + (1 - tau)*w
        """
        for target_param, param in zip(self._target_actor.parameters(), self._actor.parameters()):
            # w2 <- tau*w1 + (1 - tau)*w2
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

    def _update_target_critic_weights(self):
        """
        Update network weights for the critic
        w' <- tau*w + (1 - tau)*w
        """
        for target_param, param in zip(self._target_critic.parameters(), self._critic.parameters()):
            # theta2 <- tau*theta1 + (1 - tau)*theta2
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

    def _learn(self):
        """
        The learning phase of the agent. After experience gained, learn by sampling experience and 
        updating neural networks based on this experience
        """
        for _ in range(self._update_epoches):
            samples = self._memory._sample_transition(self._batch_size)
            # Loop through each experience in minibatch
            state, action, new_state, reward, terminal = samples

            # Gradients updates
            self._gradient_descent_critic(state, action, new_state, reward, terminal)
            self._gradient_ascent_actor(state)

            self._update_target_actor_weights()
            self._update_target_critic_weights()

            self._updates_number += 1

    def train_run(self):
        """
        The training phase of the agent. The main bulk of our agent iteratively interacting and learning from 
        environment interactions and replay memory.
        """
        seed = 0
        latest_scores = deque(maxlen=100)
        total_reward = 0
        t=0
        terminal = True

        for iteration in range(1, self._train_iterations):
            for episode in range(self._episodes):

                if terminal:
                    # Record the final reward earned and reset current state to restart episode.
                    seed+=1
                    current_state = env.reset(seed=seed)[0]
                    latest_scores.append(total_reward)
                    # write_data(file_name_csv_training_episodes, iteration * episode, episode - t, total_reward)
                    t = episode
                    total_reward = 0
                    terminal = False

                # Take action and observe experience
                action = agent._get_action(current_state)
                new_state, reward, terminal, truncated, info = env.step(action)
                total_reward += reward
                # Terminal can be where no further action can be taken (terminal) or
                # when agent takes actions indefinitely (truncated) so must therefore stop
                terminal = terminal or truncated
                self._store_transition(current_state, action, new_state, reward, terminal)
                current_state = new_state

            # Randomly give different noise to action to encourage diverse exploration
            if np.random.rand() < 0.5:
                self.noise.reset()

            self._learn()
            print(f"Iteration:{iteration}, avg_total_reward:{np.mean(latest_scores)}")

            # After some time test the agent's performance
            if iteration % self._measure_after_iterations == 0 and self._measurement:
                self.optimal_run()

    def optimal_run(self, seed=0):
        """
        Test the agent's performance after some time training
        """
        print("\nTest\n")
        optimal_seed = seed
        self._optimal = True
        latest_scores = []
        rendered = False
        for episode in range(self._optimal_run_episodes):
            optimal_seed += 1
            render_condition = episode < self._render_episode
            # Render episode after a few episodes
            if render_condition:
                env = gym.make("LunarLander-v2", 
                                continuous=self._continuity,
                                #gravity=-10.0, 
                                #enable_wind=False, 
                                #wind_power=15.0, 
                                #turbulence_power=1.5, 
                                render_mode="human"
                                )
                rendered = True
            else:
                if rendered:
                    env.close()
                env = self._env
                rendered = False

            current_state = env.reset(seed=optimal_seed)[0]
            episode_score = 0
            terminal = False
            render_condition = episode % self._render_episode == 0

            # Interact with environment
            while not terminal:
                action = agent._get_action(current_state)
                # Take action and observe experience
                new_state, reward, terminal, truncated, info = env.step(action)
                terminal = terminal or truncated
                current_state = new_state
                episode_score += reward

            latest_scores.append(episode_score)

            # write_data(file_name_csv_optimal, episode_score)

            print(f"Episode:{episode}, avg_total_reward:{np.mean(latest_scores)}")
        self._optimal = False
        if rendered:
            env.close()

        # write_data(file_name_csv_optimal_average, self._updates_number, float(np.mean(latest_scores)))
        print("\nEnd Test\n")


"_______________TRAINING______________"
algorithm = "ddpg"

continuity = True
render = "rgb_array"

train_iterations = 65
episodes = 4000
epoches = 100

optimal_run_episodes = 5
render_episode = -1
measure_after_iterations = 1
measurement = True

gamma = 0.99
tau = 0.001
learning_rate = [0.000025, 0.00025, 0.000025, 0.00025]

batch_size = 64
memory_size = 100000
update_epoches= int(episodes / batch_size)

theta=0.2
dt=1e-2
sigma=0.2

# Initialize the Lunar Lander Environment
env = gym.make("LunarLander-v2", 
               continuous=continuity,
               render_mode=render
                #gravity=-10.0, 
                #enable_wind=False, 
                #wind_power=15.0, 
                #turbulence_power=1.5,
               )
# A wrapper around the env which records video
env = gym.wrappers.RecordVideo(env, "vids")

# Agent initialization and training
agent = Agent(env,
        continuity=continuity,
        train_iterations=train_iterations,
        episodes=episodes,
        epoches=epoches,
        optimal_run_episodes=optimal_run_episodes,
        render_episode=render_episode,
        measure_after_iterations=measure_after_iterations,
        measurement=measurement,
        gamma=gamma,
        tau=tau,
        learning_rate=learning_rate,
        batch_size=batch_size,
        memory_size=memory_size,
        update_epoches=update_epoches,
        theta=theta,
        dt=dt,
        sigma=sigma)
agent.train_run()

env.close()