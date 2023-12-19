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

    def get_rollouts(self, key, section=None, randomize=False):
        if section is not None:
            section = section if isinstance(section[0], int) else (range(s[0], s[1]) for s in section)
            return np.array([element for s in section for element in self._timeline[key][s]])
        if randomize:
            return np.random.permutation(self._timeline[key])
        return self._timeline[key]
    
class Normalizer:
    def __init__(self, size):
        self._length = 0
        self._mean = np.zeros(size)
        self._std = np.ones(size)
    
    def update(self, values):
        values_count = values.shape[0]
        self._mean = ((self._mean * self._length) + values.sum(0)) / (self._length + values_count)
        self._std = np.sqrt(((np.square(self._std) * self._length) + (values.var(0) * values_count)) / (self._length + values_count))
        self._length += values_count
    
    def normalize(self, values):
        # consider the clip and the tensor use
        return (values - self._mean) / (self._std + 1e-8) 

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

class Agent:
    def __init__(self, env, continuity=True, optimal=False, gamma=0.9, lambda_gae=0.95, 
                 value_ratio_loss=0.001, entropy_loss_ratio=0.001, 
                 advantage_ratio=(0.5, 0.5), critic_ratio=(0.5, 0.5),
                 learning_rates=(0.0003, 0.001, 0.0004, 0.0011), policy_clip=0.2, 
                 training_iterations=1000, training_epoches=15,
                 rnd_steps=50, rnd_iterations=10, rnd_epoches=15,
                 normalization_episodes=150, minibatch_size=256, memory_order=None):

        # Environment 
        self._env = env

        # Environment Style
        self._continuity = continuity
        self._optimal = optimal
        self._observation_space = env.observation_space
        self._action_space = env.action_space

        # Training Cycles
        # K (length of rollouts)
        self._rnd_steps = rnd_steps
        # time-steps t is never used
        self._training_iterations = training_iterations
        # N <- Number of rollouts
        self._rnd_iterations = rnd_iterations
        # N_opt <- Number of optimization steps
        self._training_epoches = training_epoches
        self._rnd_epoches = rnd_epoches
        # M <- Number of inital steps for initlizing observation
        self._normalization_episodes = normalization_episodes
        self._minibatch_size = minibatch_size

        # Memory
        self._memory_order = memory_order
        self._memory_extension = ("intrinsic_reward",)
        self._memory = AgentMemory(*(memory_order + self._memory_extension))

        # Discounts
        # Reward Discount
        self._gamma = gamma
        # Advantage Decay Rate
        self._lambda = lambda_gae
        # Critic Value Loss Ratio Coefficient
        self._critic_ratio = critic_ratio # Ratio Divide
        self._value_loss_ratio = value_ratio_loss
        # Entropy Loss Ratio Coefficient
        self._entropy_loss_ratio = entropy_loss_ratio
        # Advantage Ratio Divide
        self._advantage_ratio = advantage_ratio

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
        self._intrinsic_critic_network = Network(self._input_dim, 1, learning_rate=learning_rates[1]).to(self._device)
        self._extrinsic_critic_network = Network(self._input_dim, 1, learning_rate=learning_rates[1]).to(self._device)
        self._rnd_predictor_network = Network(self._input_dim, 1, learning_rate=learning_rates[2]).to(self._device) # not sure about the output
        self._rnd_target_network = Network(self._input_dim, 1, learning_rate=learning_rates[3]).to(self._device) # not sure about the output

        print(self._actor_network)

        # Observation Normalization Parameters
        self._observation_normalizer = Normalizer(self._input_dim)
        self._reward_normalizer = Normalizer(1)
    
    def _get_numpy(self, tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    def _get_tensor(self, nparray, grad=True):
        return T.from_numpy(nparray).float().to(self._device).requires_grad_(grad)

    def initial_observations(self):
        terminal = True
        # for m = 1 to M do
        for episode in range(self._normalization_episodes):

            if terminal:
                # Setup
                # Sample state s_0 ∼ p_0(s_0) (after each episode)
                self._env.reset()
                terminal = False

            # sample a_t ∼ Uniform(a_t)
            action = self._action_space.sample()
            # sample s_t+1 ∼ p(s_t+1|s_t, a_t)
            new_state, _, terminal, truncated, _  = self._env.step(action)
            # sample s_t+1 ∼ p(s_t+1|s_t, a_t)
            self._observation_normalizer.update(new_state)
            # Episode termination
            terminal = terminal or truncated
    
    def train_run(self):

        latest_scores = deque(maxlen=100)
        total_reward = 0
        # Train Cycles
        for iteration in range(self._training_iterations):
            terminal = True
            trajectory = []

            for _ in range(self._rnd_iterations):
                # Collecting Trajectories D_k = {tau_i}
                for _ in range(self._rnd_steps):

                    if terminal:
                        # Setup
                        current_state = self._env.reset()[0]
                        latest_scores.append(total_reward)
                        total_reward = 0
                        terminal = False

                    # Environment Interaction
                    # sample a_t ∼ π(a_t|s_t)
                    action, log_probability = self._get_action(current_state)
                    # sample s_t+1, e_t ∼ p(s_t+1, e_t|s_t, a_t)
                    new_state, reward, terminal_state, truncated, _  = self._env.step(action)
                    total_reward += reward
                    terminal = terminal_state or truncated
                    # add s_t, s_t+1, a_t, e_t, i_t to optimization batch B_i
                    trajectory.append((current_state, action, log_probability, terminal, reward, new_state))
                    current_state = new_state

                # Memorize
                self._memory.memorize(trajectory, self._memory_order)

                batch_section = self._memory.get_rollouts("batch")[-1]

                states = self._memory.get_rollouts("state", [batch_section])
                next_states = self._memory.get_rollouts("new_state", [batch_section])

                self._train_rnd(states)

                intrinsic_rewards = self._get_intrinsic_reward(next_states)

                self._reward_normalizer.update(intrinsic_rewards)
                self._observation_normalizer.update(next_states)
                normalized_intrinsic_rewards = self._reward_normalizer.normalize(intrinsic_rewards)

                self._memory.memorize([normalized_intrinsic_rewards], self._memory_extension, record_batch=False, transpose=False)

            # Train Models
            self._train_network()

            # Forget
            self._memory.new_memory()
            
            if iteration % 1 == 0:
                print(f"Iteration:{iteration}, total_reward:{np.mean(latest_scores)}")

    def _get_action(self, state):
        # Convert to Tensor
        state_tensor = self._get_tensor(np.array([state]), False)

        # Network Prediction of Action Probabilities pi_Theta_k(a_t|s_t)
        with T.no_grad():
            action_probabilities = self._actor_network(state_tensor)

        if self._optimal:
            # Optimal action is of max probability (mu for continuous space)
            action = action_probabilities[0] if self._continuity else T.argmax(action_probabilities[0], dim=-1)
            log_probability = T.constant([1.], dtype=T.float32)
        else:
            # Action based on probability distribution (mu and sigma used in continuous space)
            distribution = dist.Normal(action_probabilities[0], action_probabilities[1]) if self._continuity else dist.Categorical(action_probabilities)
            action = distribution.sample()
            log_probability = T.sum(distribution.log_prob(action), dim=-1)
        # Numpy Conversion
        action = self._get_numpy(action)
        log_probability = self._get_numpy(log_probability)
        
        return (action[0], log_probability[0]) if self._continuity else (int(action), log_probability)
    
    def _train_rnd(self, states):
        states = self._observation_normalizer.normalize(states)

        for _ in range(self._rnd_epoches):
            state = np.random.permutation(states)

            target_state = self._get_tensor(state)
            prediction_state = self._get_tensor(state)

            target_result = self._rnd_target_network(target_state).flatten()
            prediction_result = self._rnd_predictor_network(prediction_state).flatten()

            loss = nn.MSELoss()(prediction_result, target_result) * 0.5

            self._rnd_predictor_network.optimizer.zero_grad()

            loss.backward()

            self._rnd_predictor_network.optimizer.step()
    
    def _get_intrinsic_reward(self, next_states):
        # Not sure if I should do this but I will normalize
        next_states = self._observation_normalizer.normalize(next_states)

        # Convert to Tensor
        next_states = self._get_tensor(next_states, False)

        with T.no_grad():
            # f(s_t+1)
            predictor_output = self._rnd_predictor_network(next_states)
            # ˆf(s_t+1)
            target_output = self._rnd_target_network(next_states)

        predictor_output = predictor_output.flatten()
        target_output = target_output.flatten()

        # Squared Euclidean norm = ‖ˆf(s_t+1) − f(s_t+1)‖^2
        # Not done, but the psudo asks for it
        norm = (target_output - predictor_output).pow(2)
        # Convert to Numpy
        norm = self._get_numpy(norm)

        return norm

    def _get_advantage(self, value, next_value, reward, terminal):
        deltas = reward + ((1 - terminal) * self._gamma * next_value) - value
        discounted_sum = np.zeros_like(deltas)
        running_add = 0

        # [V_0 + D*V_1 +...+D^(T-1) * V_T-1, ...,V_T-2+(D * V_T-1), V_T-1]
        for index in reversed(range(len(deltas))):
            running_add = deltas[index] + (1 - terminal[index]) * running_add * self._gamma * self._lambda
            discounted_sum[index] = running_add

        return discounted_sum
    
    def _train_network(self):
        states = self._memory.get_rollouts("state")
        actions = self._memory.get_rollouts("action")
        if not self._continuity:
            temp_actions = np.zeros((actions.size, self._output_dim))
            temp_actions[np.arange(actions.size), actions] = 1
            actions = temp_actions
        log_probabilities = self._memory.get_rollouts("log_probability")
        terminals = self._memory.get_rollouts("terminal")
        rewards = self._memory.get_rollouts("reward")
        in_rewards = self._memory.get_rollouts("intrinsic_reward")
        next_states = self._memory.get_rollouts("new_state")

        tensor_states = self._get_tensor(states, False).detach()
        tensor_next_states = self._get_tensor(next_states, False).detach()

        for _ in range(self._training_epoches):

            with T.no_grad():
                in_values = self._intrinsic_critic_network(tensor_states)
                in_next_values = self._intrinsic_critic_network(tensor_next_states)
                ex_values = self._intrinsic_critic_network(tensor_states)
                ex_next_values = self._intrinsic_critic_network(tensor_next_states)

            in_values, in_next_values = self._get_numpy(in_values.flatten()), self._get_numpy(in_next_values.flatten())
            ex_values, ex_next_values = self._get_numpy(ex_values.flatten()), self._get_numpy(ex_next_values.flatten())

            in_advantages = self._get_advantage(in_values, in_next_values, in_rewards, terminals)
            del in_next_values
            in_returns = in_advantages + in_values
            in_advantages = (in_advantages - in_advantages.mean()) / (in_advantages.std() + 1e-6)

            ex_advantages = self._get_advantage(ex_values, ex_next_values, rewards, terminals)
            del ex_next_values
            ex_returns = ex_advantages + ex_values
            ex_advantages = (ex_advantages - ex_advantages.mean()) / (ex_advantages.std() + 1e-6)

            advantages = (in_advantages * self._advantage_ratio[0]) + (ex_advantages * self._advantage_ratio[1])
            
            sample_size = len(states)
            index = 0
            while index < sample_size:

                end_index = index + self._minibatch_size
                if end_index >= sample_size:
                    end_index = sample_size - 1
                batch = range(index, end_index)

                permute = np.random.permutation(np.array(batch))

                state = self._get_tensor(states[permute])
                action = self._get_tensor(actions[permute])
                log_probability = self._get_tensor(log_probabilities[permute])
                advantage = self._get_tensor(advantages[permute])
                in_reward_return = self._get_tensor(in_returns[permute])
                ex_reward_return = self._get_tensor(ex_returns[permute])

                in_value = self._intrinsic_critic_network(state).flatten()
                ex_value = self._extrinsic_critic_network(state).flatten()
                action_probabilities = self._actor_network(state)

                in_critic_loss = nn.MSELoss()(in_value, in_reward_return)
                ex_critic_loss = nn.MSELoss()(ex_value, ex_reward_return)
                critic_loss = (in_critic_loss * self._critic_ratio[0]) + (ex_critic_loss * self._critic_ratio[1])

                loss = self._loss_function(action_probabilities, action, advantage, log_probability, critic_loss)

                self._actor_network.optimizer.zero_grad()
                self._intrinsic_critic_network.optimizer.zero_grad()
                self._extrinsic_critic_network.optimizer.zero_grad()
                
                loss.backward()

                self._actor_network.optimizer.step()
                self._intrinsic_critic_network.optimizer.step()
                self._extrinsic_critic_network.optimizer.step()

                index += self._minibatch_size

    def _loss_function(self, predicted_action_probabilities, action, advantage, old_log_probabilities, critic_loss):
        
        if self._continuity:
            distribution = dist.Normal(predicted_action_probabilities[0], predicted_action_probabilities[1])
            current_log_probabilities = T.sum(distribution.log_prob(action), dim=-1)
        else:
            current_log_probabilities = T.log(T.sum(predicted_action_probabilities * action, dim=-1))

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
            entropy_loss = -self._entropy_loss_ratio * T.mean(T.sum(predicted_action_probabilities * T.log(predicted_action_probabilities + 1e-7), dim=-1))
        
        loss = actor_loss + critic_loss
        loss = loss + entropy_loss
        return loss

training_iterations = 10000
rnd_iterations = 5
rnd_steps = 200
rnd_epoches = 5
epoches = 15
normalization_episodes = 1000
batch_size = 256
memory_order = ("state", "action", "log_probability", "terminal", "reward", "new_state")
gamma = 0.99
lambda_gae = 0.97
value_ratio_loss = 0.001
entropy_loss_ratio = 0.001
advantage_ratio = (0.1, 0.9)
critic_ratio = (0.1, 0.9)
learning_rates = (0.0003, 0.001, 0.0004, 0.0011)
policy_clip = 0.15
continuity = False
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
              rnd_iterations=rnd_iterations,
              rnd_steps=rnd_steps,
              rnd_epoches=rnd_epoches,
              training_epoches=epoches,
              normalization_episodes=normalization_episodes,
              minibatch_size=batch_size,
              gamma=gamma,
              lambda_gae=lambda_gae,
              advantage_ratio=advantage_ratio,
              critic_ratio=critic_ratio,
              value_ratio_loss=value_ratio_loss,
              entropy_loss_ratio=entropy_loss_ratio,
              learning_rates=learning_rates,
              policy_clip=policy_clip,
              memory_order=memory_order)

agent.train_run()
env.close()