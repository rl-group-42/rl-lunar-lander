import torch as T
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.backends import mps
import gymnasium as gym
import numpy as np
from collections import deque

def _get_torch_device():
    if cuda.is_available():
        return "cuda"
    elif mps.is_available():
        return "mps"
    else:
        return "cpu"

class ReplayMemory():
    def __init__(self, max_mem_size=1000000):
        self._max_mem_size = max_mem_size
        self._state = deque(maxlen=max_mem_size)  # Initialize deque with maximum size
        self._action = deque(maxlen=max_mem_size)  # Initialize deque with maximum size
        self._new_state = deque(maxlen=max_mem_size)  # Initialize deque with maximum size
        self._reward = deque(maxlen=max_mem_size)  # Initialize deque with maximum size
        self._terminal = deque(maxlen=max_mem_size)  # Initialize deque with maximum size
        self._mem_ctr = 0

    def _store_transition(self, state, action, new_state, reward, terminal):
        self._state.append(state)
        self._action.append(action)
        self._new_state.append(new_state)
        self._reward.append(reward)
        self._terminal.append(terminal)
        
    def _sample_transition(self, batch_size):
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
    def __init__(self, action_dim, theta=0.2, dt=1e-2, sigma=0.2):
        self._action_dim = action_dim
        self._mu = np.zeros(self._action_dim) 
        self._theta = theta
        self._sigma = sigma
        self._dt = dt
        self.reset()

    def reset(self):
        self._state = np.zeros(self._action_dim) 

    def __call__(self):
        # dxt = theta (mu - X) + sigma(dWt)
        dx = self._theta * (self._mu - self._state) * self._dt + self._sigma  * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        self._state += dx
        return self._state

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, output_activations=tuple(), hidden_dims=(64, 32), hidden_activations=None, learning_rate=0.01):
        super(Network, self).__init__()

        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim

        hidden_activations = [nn.ReLU() for _ in hidden_dims] if hidden_activations is None else hidden_activations

        # Hidden Layer
        layers = []
        prev_dim = input_dim
        for dim, activation in zip(hidden_dims, hidden_activations):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation)
            prev_dim = dim
        self._hidden_layers = nn.Sequential(*layers)

        # Define output layers
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
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self._hidden_layers(x)
        outputs = list(output_layer(x) for output_layer in self._output_layers)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

class Agent():
    def __init__(self, env, gamma=0.99, beta=0.995, learning_rate=(0.001,), update_epoches=10, theta=0.2, dt=1e-2, sigma=0.2, batch_size=64, memory_size=1000):
        # Reward Discount
        self._gamma = gamma
        # Weight Discount
        self._beta = beta
        
        # Alpha rate
        self._learning_rate = learning_rate
        self._update_epoches = update_epoches
        
        # Replay Memory
        self._memory = ReplayMemory(memory_size)
        self._batch_size = batch_size
        
        # Environment
        self._env = env
        self._state_dims = env.observation_space.shape[0]
        self._action_dims = env.action_space.shape[0]
        self._min_bounds = env.action_space.low
        self._max_bounds = env.action_space.high
        
        # Ornstein Uhlenbeck noise
        self.noise = Noise(self._action_dims, theta, dt, sigma)

        # Activation Functions
        activations = (nn.Tanh(),)
        
        # Torch device: GPU or CPU
        self._device = T.device(_get_torch_device())
        
        # Networks
        self._actor = Network(self._state_dims, self._action_dims, activations, learning_rate=self._learning_rate[0]).to(self._device)
        self._critic = Network(self._state_dims + self._action_dims, 1, learning_rate=learning_rate[1]).to(self._device)
        self._target_actor = Network(self._state_dims, self._action_dims, activations, learning_rate=self._learning_rate[2]).to(self._device)
        self._target_critic = Network(self._state_dims + self._action_dims, 1, learning_rate=learning_rate[3]).to(self._device)

        self._actor.train()
        self._critic.train()
        self._target_actor.train()
        self._target_critic.train()

    def _get_numpy(self, tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    def _get_tensor(self, nparray, grad=True):
        return T.from_numpy(nparray).float().to(self._device).requires_grad_(grad)

    def _get_action(self, state, optimal=False):
        self._actor.eval()
        state_tensor = self._get_tensor(np.array([state]), False).detach()
        
        with T.no_grad():
            action = self._actor(state_tensor)
        
        action = self._get_numpy(action)[0]

        if not optimal:
            action = np.clip((action + self.noise()), self._min_bounds, self._max_bounds)

        self._actor.train()
        return action

    def _store_transition(self, state, action, new_state, reward, terminal):
        self._memory._store_transition(state, action, new_state, reward, terminal)

    def _gradient_descent_critic(self, state, action, new_state, reward, terminal):
        # Convert to tensors
        action_tensor = self._get_tensor(action)
        state_tensor = self._get_tensor(state)
        new_state_tensor = self._get_tensor(new_state)
        terminal_tensor = self._get_tensor(terminal)
        reward_tensor = self._get_tensor(reward)

        # Target reward for next state
        target_actions = self._target_actor(new_state_tensor.detach())
        target_critic_value = self._target_critic(T.cat((new_state_tensor.detach(), target_actions.detach()), 1)).flatten()
        target = reward_tensor + self._gamma * target_critic_value * (1 - terminal_tensor)

        # Actual critic   
        critic_value = self._critic(T.cat((state_tensor, action_tensor), 1)).flatten()

        # MSE Loss function
        mse = T.nn.MSELoss()
        critic_loss = mse(critic_value, target)

        # Gradient step
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
        for target_param, param in zip(self._target_actor.parameters(), self._actor.parameters()):
            # w2 <- beta*w1 + (1 - beta)*w2
            target_param.data.copy_(self._beta * param.data + (1 - self._beta) * target_param.data)

    def _update_target_critic_weights(self):
        for target_param, param in zip(self._target_critic.parameters(), self._critic.parameters()):
            # theta2 <- beta*theta1 + (1 - beta)*theta2
            target_param.data.copy_(self._beta * param.data + (1 - self._beta) * target_param.data)

    def _learn(self):
        for _ in range(self._update_epoches):
            samples = self._memory._sample_transition(self._batch_size)
            # Loop through each experience in minibatch
            state, action, new_state, reward, terminal = samples
            
            # Gradients updates
            self._gradient_descent_critic(state, action, new_state, reward, terminal)
            self._gradient_ascent_actor(state)

            self._update_target_actor_weights()
            self._update_target_critic_weights()


"_______________TRAINING______________"
train_iterations = 100000
test_after_iteration = 30
prove_sample_size = 30
update_after_size = 1024
best_score = float('-inf')
episode_score_history = []
terminal = True
episode_score = 0

gamma = 0.99
beta = 0.995
learning_rate = [0.001, 0.0015, 0.001, 0.0015]

batch_size = 64
memory_size = 100000
update_epoches= int(update_after_size / batch_size)

theta=0.2
dt=1e-2
sigma=0.2

env = gym.make("LunarLander-v2", 
               continuous= True 
               #render_mode="human"
               )

agent = Agent(env,
              gamma=gamma,
              beta=beta,
              learning_rate=learning_rate,
              update_epoches=update_epoches,
              theta=theta,
              dt=dt,
              sigma=sigma,
              batch_size=batch_size,
              memory_size=memory_size)


for i in range(1, train_iterations):

    for j in range(update_after_size):

        if terminal:
            current_state = env.reset()[0]
            episode_score_history.append(episode_score)
            episode_score = 0
            terminal = False
            episode_score_history = episode_score_history[-100:]
            avg_score = np.mean(episode_score_history)

            if avg_score > best_score:
                best_score = avg_score
        
        action = agent._get_action(current_state)
        # Take action and observe experience
        new_state, reward, terminal, truncated, info = env.step(action)
        terminal = terminal or truncated
        agent._store_transition(current_state, action, new_state, reward, terminal)
        current_state = new_state
        episode_score += reward
  

    if np.random.rand() < 0.5:
        agent.noise.reset()

    agent._learn()
    print(f"Episode: {i}, score: {episode_score}, avg_score: {avg_score}, best_epi_score: {max(episode_score_history)}")

    if i % prove_sample_size==0:
        episode_score_history = []
        print()
        print("test")
        print()
        for j in range(prove_sample_size):

            current_state = env.reset()[0]
            episode_score = 0
            terminal = False

            while not terminal:
                action = agent._get_action(current_state, True)
                # Take action and observe experience
                new_state, reward, terminal, truncated, info = env.step(action)
                terminal = terminal or truncated
                current_state = new_state
                episode_score += reward

            episode_score_history.append(episode_score)
            avg_score = np.mean(episode_score_history)

            if avg_score > best_score:
                best_score = avg_score
            print(f"Episode: {j}, score: {episode_score}, avg_score: {avg_score}, best_epi_score: {max(episode_score_history)}")
        episode_score_history = []
        print()
        print("test end")
        print()
