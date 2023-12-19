import torch as T
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.backends import mps
import gym
import numpy as np
import torch.distributions as dist
from collections import deque

def _get_torch_device():
    if cuda.is_available():
        return "cuda"
    elif mps.is_available():
        return "mps"
    else:
        return "cpu"

class ReplayMemory():
    def __init__(self, max_mem_size=100):
        self._max_mem_size = max_mem_size
        self._memory = deque(maxlen=max_mem_size)  # Initialize deque with maximum size
        self._mem_ctr = 0

    def _store_transition(self, state, action, new_state, reward, terminal):
        transition = (state, action, new_state, reward, terminal)
        self._memory.append(transition)  # Append new transition
        
    def _sample_transition(self, batch_size):
        # Return all experiences if memory size is smaller than batch size
        if len(self._memory) < batch_size:
            return list(self._memory)
        
        # Otherwise, convert deque to a list and sample a batch of experiences
        samples = list(self._memory)
        batch_indexes = np.random.choice(len(samples), batch_size, replace=False)
        batch = [samples[i] for i in batch_indexes]
        return batch

class Noise():
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self._action_dim = action_dim
        self._mu = mu
        self._theta = theta
        self._sigma = sigma
        self._state = np.ones(self._action_dim) * self._mu
        self._reset()

    def _reset(self):
        self._state = np.ones(self._action_dim) * self._mu

    def __call__(self):
        # dxt = theta (mu - X) + sigma(dWt)
        dx = self._theta * (self._mu - self._state) + self._sigma * np.random.randn(self._action_dim)
        self._state += dx
        return self._state

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dims=(64, 32), hidden_activation=None,
                 output_activation=None, learning_rate=0.01):
        super(ActorNetwork, self).__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._hidden_dims = hidden_dims
        self._hidden_activation = nn.ReLU() if hidden_activation is None else hidden_activation
        self._output_activation = nn.Tanh() if output_activation is None else output_activation

        # Define Actor Network
        layers = []
        dims = (input_shape, ) + hidden_dims + (output_shape, )
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Hidden layer
            if i < len(dims) - 2:
                layers.append(self._hidden_activation)
            # Output layer
            else:
                layers.append(self._output_activation)

        self._network = nn.Sequential(*layers)
        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, state):
        actions = self._network(state)
        return actions

class CriticNetwork(nn.Module):
    def __init__(self, input_states, input_actions, hidden_dims=(64, 32), hidden_activation=None,
                 output_activation=None, learning_rate=0.01):
        super(CriticNetwork, self).__init__()
        self._input_states = input_states
        self._input_actions = input_actions
        self._hidden_dims = hidden_dims
        self._hidden_activation = nn.ReLU() if hidden_activation is None else hidden_activation
        self._output_activation = None if output_activation is None else output_activation
        
        # Defin Critic Network
        layers = []
        dims = (input_states + input_actions, ) + hidden_dims + (1, )
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Hidden Layer
            if i < len(dims) - 2:
                layers.append(self._hidden_activation)
            # Output Layer
            else:
                # No activation for output layer
                if self._output_activation == None:
                    continue
                else:
                    layers.append(self._output_activation)

        self._network = nn.Sequential(*layers)
        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, state, actions):
        state_action = T.cat((state, actions), 1)
        return self._network(state_action)

class Agent():
    def __init__(self, env, continuity=True, gamma=0.5, beta=0.1, learning_rate=0.0001, batch_size=10):
        # Reward Discount
        self._gamma = gamma
        # Weight Discount
        self._beta = beta
        
        # Alpha rate
        self._learning_rate = learning_rate
        
        # Replay Memory
        self._memory = ReplayMemory()
        self._batch_size = batch_size
        
        # Environment
        self._env = env
        self._continuity = continuity
        self._state_dims = env.observation_space.shape[0]
        self._action_dims = env.action_space.shape[0] if continuity else env.action_space.n
        self._min_bounds = env.action_space.low
        self._max_bounds = env.action_space.high
        
        # Ornstein Uhlenbeck noise
        self._noise = Noise(self._action_dims)
        
        # Torch device: GPU or CPU
        self._device = T.device(_get_torch_device())
        
        # Networks
        self._actor = ActorNetwork(self._state_dims, self._action_dims, learning_rate=self._learning_rate).to(self._device)
        self._critic = CriticNetwork(self._state_dims, self._action_dims, learning_rate=learning_rate).to(self._device)
        self._target_actor = ActorNetwork(self._state_dims, self._action_dims, learning_rate=self._learning_rate).to(self._device)
        self._target_critic = CriticNetwork(self._state_dims, self._action_dims, learning_rate=learning_rate).to(self._device)

    def _get_action(self, state):
        state_tensor = T.from_numpy(np.array([state])).float().to(self._device)
        
        action = self._actor(state_tensor).detach().numpy()[0]
        
        clipped_action = np.clip((action + self._noise()), self._min_bounds, self._max_bounds)
        
        return clipped_action

    def _store_transition(self, state, action, new_state, reward, terminal):
        self._memory._store_transition(state, action, new_state, reward, terminal)

    def _gradient_descent_critic(self, state, action, new_state, reward, terminal):
        # Convert to tensors
        action_tensor = T.from_numpy(np.array([action])).float().to(self._device)
        state_tensor = T.from_numpy(np.array([state])).float().to(self._device)
        new_state_tensor = T.from_numpy(np.array([new_state])).float().to(self._device)

        # Target reward for next state
        target_actions = self._target_actor(new_state_tensor)
        target_critic_value = self._target_critic(new_state_tensor, target_actions)
        target = reward + self._gamma * target_critic_value * (1- terminal)

        # Actual critic   
        critic_value = self._critic(state_tensor, action_tensor)

        # MSE Loss function
        mse = T.nn.MSELoss()
        critic_loss = mse(target, critic_value)

        # Gradient step
        self._critic.train()
        self._critic._optimizer.zero_grad()
        critic_loss.backward()
        self._critic._optimizer.step()

    def _gradient_ascent_actor(self, state):
        # Convert to tensor
        state_tensor = T.from_numpy(np.array([state])).float().to(self._device)
        action_tensor = T.from_numpy(np.array([self._get_action(state)])).float().to(self._device)
        
        # Critic value
        critic_value = self._critic(state_tensor, action_tensor)
        
        # Mean Loss
        actor_loss = -T.mean(critic_value)
        
        # Gradient step
        self._actor._optimizer.zero_grad()
        actor_loss.backward()
        self._actor._optimizer.step()
    
    def _update_target_actor_weights(self):
        for target_param, param in zip(self._target_actor.parameters(), self._actor.parameters()):
            # w2 <- beta*w1 + (1 - beta)*w2
            target_param.data.copy_(self._beta * param.data + (1 - self._beta) * target_param.data)

    def _update_target_critic_weights(self):
        for target_param, param in zip(self._target_critic.parameters(), self._critic.parameters()):
            # theta2 <- beta*theta1 + (1 - beta)*theta2
            target_param.data.copy_(self._beta * param.data + (1 - self._beta) * target_param.data)

    def _learn(self):
        samples = self._memory._sample_transition(self._batch_size)
        # Loop through each experience in minibatch
        for transition in samples:
            state, action, new_state, reward, terminal = transition
            
            # Gradients updates
            self._gradient_descent_critic(state, action, new_state, reward, terminal)
            self._gradient_ascent_actor(state)

            # Update the weights of the networks after episode has ended
            if terminal:
                self._update_target_actor_weights()
                self._update_target_critic_weights()


"_______________TRAINING______________"
env = gym.make("LunarLander-v2", continuous= True, render_mode="human")
continuity = True
n_games = 1500
best_score = float('-inf')
episode_score_history = []
agent = Agent(env, continuity)

for i in range(n_games):
    current_state = env.reset()[0]
    episode_score = 0
    terminal = False
    while not terminal:
        action = agent._get_action(current_state)
        # Take action and observe experience
        new_state, reward, terminal, truncated, info = env.step(action)
        terminal = terminal or truncated
        agent._store_transition(current_state, action, new_state, reward, terminal)
        current_state = new_state
        episode_score += reward
        # Learn form experience
        agent._learn()

    episode_score_history.append(episode_score)
    avg_score = np.mean(episode_score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
    print(f"Episode: {i}, score: {episode_score}, avg_score: {avg_score}, best_epi_score: {max(episode_score_history)}")
