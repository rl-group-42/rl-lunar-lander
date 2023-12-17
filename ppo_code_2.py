import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import torch.distributions as dist
import torch.nn.functional as F
from torch import cuda
from torch.backends import mps
from collections import deque

def _get_torch_device() -> str:
    if cuda.is_available():
        return "cuda"
    elif mps.is_available():
        return "mps"
    else:
        return "cpu"

class AgentMemory:
    def __init__(self, *timeline_keys):
        self.new_memory(timeline_keys if timeline_keys else ("state", "action", "terminal", "reward", "next_state", "log_probability", "value"))

    def new_memory(self, timeline_keys=None):
        self._timeline_keys = timeline_keys if timeline_keys else self._timeline_keys
        self._timeline = {key: None for key in self._timeline_keys}
        self._timeline["batch"] = np.empty((0, 2), dtype=object)
        self._timeline["batch_index"] = np.array([], dtype=int)
    
    def memorize(self, trajectories, keys=None, record_batch=True, transpose=True):
        if transpose:
            trajectories = list(zip(*trajectories))
        if keys is None:
            keys = self._timeline_keys
        for key_index in range(len(keys)):
            key = keys[key_index]
            a = np.array(trajectories[key_index])
            b = self._timeline[key]
            self._timeline[key] = np.concatenate((self._timeline[key], np.array(trajectories[key_index]))) if self._timeline[key] is not None else np.array(trajectories[key_index])
        if record_batch:
            last_index = self._timeline["batch"][-1][1] if self._timeline["batch"].size else 0
            self._timeline["batch"] = np.concatenate((self._timeline["batch"], np.array([[last_index, len(trajectories[0]) + last_index]])))
            self._timeline["batch_index"] = np.append(self._timeline["batch_index"], self._timeline["batch_index"][-1] + 1 if self._timeline["batch_index"].size else 0)
    
    def get_rollouts(self, key, section=None, randomize=False):
        if section is not None:
            section = section if isinstance(section[0], int) else (range(s[0], s[1]) for s in section)
            return np.array([element for s in section for element in self._timeline[key][s]])
        if randomize:
            return np.random.permutation(self._timeline[key])
        return self._timeline[key]
    

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
        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self._hidden_layers(x)
        outputs = list(output_layer(x) for output_layer in self._output_layers)
        if len(outputs) == 1:
            return outputs[0]
        return outputs
    
    def train_step(self, loss_function, x, *y):
        self.train()
        self._optimizer.zero_grad()
        prediction = self(x)
        loss = loss_function(prediction, *y)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    
class Agent:
    def __init__(self, env, continuity=True, optimal=False, gamma=0.9, lambda_gae=0.95, 
                 value_ratio_loss=0.001, entropy_loss_ratio=0.001,
                 learning_rates=(0.01, 0.2), policy_clip=0.2, training_iterations=1000, 
                 episodes=20, epoches=40, minibatch_size=256, memory_order=None):

        # Environment 
        self._env = env

        # Environment Style
        self._continuity = continuity
        self._optimal = optimal
        self._observation_space = env.observation_space
        self._action_space = env.action_space

        # Training Cycles
        self._training_iterations = training_iterations
        self._episodes = episodes
        self._epoches = epoches
        self._minibatch_size = minibatch_size

        # Memory
        self._memory_order = memory_order
        self._memory_extension = ("value", "advantage", "return")
        self._memory = AgentMemory(*(memory_order + self._memory_extension))

        # Discounts
        # Reward Discount
        self._gamma = gamma
        # Advantage Decay Rate
        self._lambda = lambda_gae
        # Critic Value Loss Ratio Coefficient
        self._value_loss_ratio = value_ratio_loss
        # Entropy Loss Ratio Coefficient
        self._entropy_loss_ratio = entropy_loss_ratio

        # Learning Rates
        # Alpha Rate
        self._learning_rates = learning_rates
        # Clip Epsilon
        self._policy_clip = policy_clip


        self._input_dim = self._observation_space.shape[0]
        self._output_dim = self._action_space.shape[0] if continuity else self._action_space.n

        activations = (nn.Tanh(), nn.Softplus()) if continuity else (nn.Softmax(dim=-1),)

        self._device = T.device(_get_torch_device())

        self._actor_network = Network(self._input_dim, self._output_dim, activations, learning_rate=learning_rates[0]).to(self._device)
        self._critic_network = Network(self._input_dim, 1, learning_rate=learning_rates[1]).to(self._device)

        print(self._actor_network, self._critic_network)

    def train_run(self):

        latest_scores = deque(maxlen=100)
        # Train Cycles
        for iteration in range(self._training_iterations):

            # Collecting Trajectories D_k = {tau_i}
            for episode in range(self._episodes):

                # Setup
                trajectory = []
                current_state = self._env.reset()[0]
                total_reward = 0
                terminal = False

                # Each trajectory is an episode
                while not terminal:

                    # Environment Interaction
                    action, log_probability = self._get_action(current_state)
                    new_state, reward, terminal_state, truncated, _  = self._env.step(action)
                    total_reward += reward
                    terminal = terminal_state or truncated
                    trajectory.append((current_state, action, log_probability, terminal, reward, new_state))
                    current_state = new_state
                
                latest_scores.append(total_reward)

                # Memorize
                self._memory.memorize(trajectory, self._memory_order)

                batch_section = self._memory.get_rollouts("batch")[-1]
                rewards = self._memory.get_rollouts("reward", [batch_section])
                last_reward = reward

                # Get Critic Values
                states = self._memory.get_rollouts("current_state", [batch_section])
                states = T.from_numpy(states).float().to(self._device)
                with T.no_grad():
                    values = self._critic_network(states)
                values = values.flatten()
                values = values.cpu().numpy() if values.is_cuda else values.numpy()

                # Compute Advantage and Return
                advantage_returns = self._advantages_returns_calculation(rewards, values, last_reward)

                # Memorize Extensions
                self._memory.memorize((values,) + advantage_returns, self._memory_extension, record_batch=False, transpose=False)

            # Train Models
            self._train_network()

            # Forget
            self._memory.new_memory()
            
            if iteration % 1 == 0:
                print(f"Iteration:{iteration}, total_reward:{np.mean(latest_scores)}")

    def _get_action(self, state):
        # Convert to Tensor
        state_tensor = T.from_numpy(np.array([state])).float().to(self._device)

        # Network Prediction of Action Probabilities pi_Theta_k(a_t|s_t)
        with T.no_grad():
            action_probabilities = self._actor_network(state_tensor)

        if self._optimal:
            # Optimal action is of max probability (mu for continuous space)
            action = action_probabilities[0] if self._continuity else T.argmax(action_probabilities[0], dim=1)
            log_probability = T.constant([1.], dtype=T.float32)
        else:
            # Action based on probability distribution (mu and sigma used in continuous space)
            distribution = dist.Normal(action_probabilities[0], action_probabilities[1]) if self._continuity else dist.Categorical(action_probabilities)
            action = distribution.sample()
            log_probability = T.sum(distribution.log_prob(action), dim=-1)

        # Numpy Conversion
        action = action.cpu().numpy() if action.is_cuda else action.numpy()
        log_probability = log_probability.cpu().numpy() if log_probability.is_cuda else log_probability.numpy()
        
        return action[0], log_probability[0]

    def _advantages_returns_calculation(self, rewards, values, last_reward):
        # Extend critic value with the last reward
        values = np.append(values, last_reward)

        # delta_t = r_t + (gamma * V(s_t+1)) - V(s_t)
        deltas = rewards + (self._gamma * values[1:]) - values[:-1]

        advantages = self._discounted_cummulative_sum(deltas, self._gamma * self._lambda)
        returns = self._discounted_cummulative_sum(rewards, self._gamma)

        return advantages, returns

    def _discounted_cummulative_sum(self, values, discount):
        discounted_sum = np.zeros_like(values)
        running_add = 0

        # [V_0 + D*V_1 +...+D^(T-1) * V_T-1, ...,V_T-2+(D * V_T-1), V_T-1]
        for index in reversed(range(len(values))):
            running_add = running_add * discount + values[index]
            discounted_sum[index] = running_add

        return discounted_sum
    
    def _train_network(self):
        for _ in range(self._epoches):
            batch_indicies = self._memory.get_rollouts("batch_index", randomize=True)
            batches = self._memory.get_rollouts("batch")[batch_indicies]
            states = self._memory.get_rollouts("current_state", batches)
            r = np.random.permutation(np.arange(0, len(states)))
            states = states[r]
            actions = self._memory.get_rollouts("action", batches)[r]
            advantages = self._memory.get_rollouts("advantage", batches)[r]
            returns = self._memory.get_rollouts("return", batches)[r]
            #values = self._memory. get_rollouts("value", batches)
            log_probabilities = self._memory.get_rollouts("log_probability", batches)[r]
            
            sample_size = len(states)
            index = 0
            while index < sample_size:

                end_index = index + self._minibatch_size
                if end_index >= sample_size:
                    end_index = sample_size - 1
                batch = range(index, end_index)

                state = T.from_numpy(states[batch]).float().to(self._device)
                action = T.from_numpy(actions[batch]).float().to(self._device)
                advantage = T.from_numpy(advantages[batch]).float().to(self._device)
                #value = T.from_numpy(values[batch]).float().to(self._device)
                reward_return = T.from_numpy(returns[batch].reshape((end_index - index, 1))).float().to(self._device)
                log_probability = T.from_numpy(log_probabilities[batch]).float().to(self._device)

                #value.requires_grad = True
                state.requires_grad = True
                action.requires_grad = True
                advantage.requires_grad = True
                reward_return.requires_grad = True
                log_probability.requires_grad = True

                critic_loss = self._critic_network.train_step(nn.MSELoss(), state, reward_return)
                self._actor_network.train_step(self._loss_function, state, action, advantage, log_probability, critic_loss)
                
                index += self._minibatch_size

    def _loss_function(self, predicted_action_probabilities, action, advantage, old_log_probabilities, critic_loss):
        
        distribution = dist.Normal(predicted_action_probabilities[0], predicted_action_probabilities[1]) if self._continuity else dist.Categorical(predicted_action_probabilities)
        current_log_probabilities = T.sum(distribution.log_prob(action), dim=-1)

        # Probability Ratio: r_t(Theta) = pi_Theta(a|s) / pi_Theta_k(a|s) = e^(pi_Theta(a|s) - pi_Theta_k(a|s))
        ratio = T.exp(current_log_probabilities - old_log_probabilities)

        # Importance Sampling Ratio: r_t(Theta) * A_t
        weighted_probabilities = ratio * advantage

        # Simplified Clipping g(epsilon, A_t) = ((1 + epsilon) if A_t >= 0 else if A_t < 0 (1 - epsilon)) * A_t
        clipped_probabilities = T.clamp(ratio, 1 - self._policy_clip, 1 + self._policy_clip)
        weighted_clipped_probabilities = clipped_probabilities * advantage

        # L_t^CLIP(Theta) = E_t[min(r_t(Theta)A_t, g(epsilon, A_t)]
        # Negative indicating loss
        actor_loss = -T.mean(T.min(weighted_probabilities, weighted_clipped_probabilities))

        # c_1 L_t^VF(Theta) = c_1 MSE = c_1 (V_Theta(s_t) - V_t^target)^2
        # Negative sign is not used since MSE is being used
        critic_loss = self._value_loss_ratio * critic_loss

        # c_2 S[pi_Theta](s_t) = c_2 Entropy Bonus
        if self._continuity:
            sigma = predicted_action_probabilities[0]
            variance = sigma.pow(2)
            entropy_loss = -self._entropy_loss_ratio * T.mean((T.log(2 * 3.14159 * variance) + 1) / 2)  
        else:
            entropy_loss = -self._entropy_loss_ratio * T.mean(T.sum(current_log_probabilities * T.log(current_log_probabilities + 1e-7), dim=-1))
        
        loss = actor_loss + critic_loss
        loss = loss + entropy_loss
        return loss
    
training_iterations = 10000
episodes = 10
epoches = 15
batch_size = 256
memory_order = ("current_state", "action", "log_probability", "terminal", "reward", "new_state")
gamma = 0.99
lambda_gae = 0.97
value_ratio_loss = 0.001
entropy_loss_ratio = 0.001
learning_rates = [0.0003, 0.001]
policy_clip = 0.15
continuity = True
optimal = False
env = gym.make("LunarLander-v2", 
         continuous=continuity
         #gravity=-10.0, 
         #enable_wind=False, 
         #wind_power=15.0, 
         #turbulence_power=1.5, 
         #render_mode="human"
         )

agent = Agent(env, 
              continuity=continuity, 
              optimal=optimal, 
              training_iterations=training_iterations,
              episodes=episodes,
              epoches=epoches,
              minibatch_size=batch_size,
              gamma=gamma,
              lambda_gae=lambda_gae,
              value_ratio_loss=value_ratio_loss,
              entropy_loss_ratio=entropy_loss_ratio,
              learning_rates=learning_rates,
              policy_clip=policy_clip,
              memory_order=memory_order)
agent.train_run()
env.close()