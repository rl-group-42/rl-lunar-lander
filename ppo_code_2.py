import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import torch.distributions as dist
from torch import cuda
from torch.backends import mps
from collections import deque
import csv

number_exp = "network_1_128_c"
base_name = "C:/Users/HP/Documents/1 FILE 7658 Codex/University/RL/CODE/Data/"
file_name_csv_optimal_average = base_name + "ppo_test_csv_oprimal_average_" + str(number_exp) + ".csv"
file_name_csv_optimal =  base_name + "ppo_test_csv_oprimal_" + str(number_exp) + ".csv"
file_name_txt =  base_name + "ppo_test_txt_" + str(number_exp) + ".txt"
file_name_csv_training_episodes = base_name + "ppo_test_csv_training_episodes_" + str(number_exp) + ".csv"

def write_data(file_path, *data):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([*data])

def get_device() -> str:
    """ Get Device Name """
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
    
    def memorize(self, trajectories, keys=None, transpose=True):
        if transpose:
            trajectories = list(zip(*trajectories))
        if keys is None:
            keys = self._timeline_keys
        for key_index in range(len(keys)):
            key = keys[key_index]
            a = np.array(trajectories[key_index])
            b = self._timeline[key]
            self._timeline[key] = np.concatenate((self._timeline[key], np.array(trajectories[key_index]))) if self._timeline[key] is not None else np.array(trajectories[key_index])
    
    def get_rollouts(self, key, section=None, randomize=False):
        if section is not None:
            section = section if isinstance(section[0], int) else (range(s[0], s[1]) for s in section)
            return np.array([element for s in section for element in self._timeline[key][s]])
        if randomize:
            return np.random.permutation(self._timeline[key])
        return self._timeline[key]
    

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, output_activations=tuple(), hidden_dims=(128,), hidden_activations=None, learning_rate=0.01):
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
                 learning_rates=(0.01, 0.2), policy_clip=0.2, training_iterations=1000, 
                 optimal_run_episodes=30, render_episode=1, measure_after_iterations=10, measurement=True,
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
        self._updates_number = 0

        # Optimal Run
        self._optimal_run_episodes = optimal_run_episodes
        self._render_episode = render_episode
        self._measure_after_iterations = measure_after_iterations
        self._measurement = measurement

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

        self._device = T.device(get_device())

        self._actor_network = Network(self._input_dim, self._output_dim, activations, learning_rate=learning_rates[0]).to(self._device)
        self._critic_network = Network(self._input_dim, 1, learning_rate=learning_rates[1]).to(self._device)
        self._actor_network.train()
        self._critic_network.train()


        print(self._actor_network, self._critic_network)

    def _get_numpy(self, tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    def _get_tensor(self, nparray, grad=True):
        return T.from_numpy(nparray).float().to(self._device).requires_grad_(grad)

    def train_run(self):
        seed = 0
        latest_scores = deque(maxlen=100)
        total_reward = 0
        t=0
        terminal = True
        # Train Cycles
        for iteration in range(1, self._training_iterations):
            trajectory = []

            # Collecting Trajectories D_k = {tau_i}
            for episode in range(self._episodes):
                
                if terminal:
                    # Setup
                    seed += 1
                    current_state = self._env.reset(seed=seed)[0]
                    latest_scores.append(total_reward)
                    write_data(file_name_csv_training_episodes, iteration * episode, episode - t, total_reward)
                    t = episode
                    total_reward = 0
                    terminal = False

                # Environment Interaction
                action, log_probability = self._get_action(current_state)
                new_state, reward, terminal_state, truncated, _  = self._env.step(action)
                total_reward += reward
                terminal = terminal_state or truncated
                trajectory.append((current_state, action, log_probability, terminal, reward, new_state))
                current_state = new_state

            # Memorize
            self._memory.memorize(trajectory, self._memory_order)

            # Train Models
            self._train_network()

            # Forget
            self._memory.new_memory()

            print(f"Iteration:{iteration}, avg_total_reward:{np.mean(latest_scores)}")
            
            if iteration % self._measure_after_iterations == 0 and self._measurement:
                # Measure improvement
                self.optimal_run(seed)

    def optimal_run(self, seed=0):
        print("\nTest\n")
        optimal_seed = seed
        self._optimal = True
        latest_scores = []
        rendered = False
        for episode in range(self._optimal_run_episodes):
            optimal_seed += 1
            render_condition = episode < self._render_episode
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

            while not terminal:
                action, _ = agent._get_action(current_state)
                # Take action and observe experience
                new_state, reward, terminal, truncated, info = env.step(action)
                terminal = terminal or truncated
                current_state = new_state
                episode_score += reward

            latest_scores.append(episode_score)
            write_data(file_name_csv_optimal, episode_score)

            print(f"Episode:{episode}, avg_total_reward:{np.mean(latest_scores)}")
        self._optimal = False
        if rendered:
            env.close()

        write_data(file_name_csv_optimal_average, self._updates_number, float(np.mean(latest_scores)))
        print("\nEnd Test\n")


    def _get_action(self, state):
        # Convert to Tensor
        state_tensor = self._get_tensor(np.array([state]), False)

        # Network Prediction of Action Probabilities pi_Theta_k(a_t|s_t)
        self._actor_network.eval()
        with T.no_grad():
            action_probabilities = self._actor_network(state_tensor)
        self._actor_network.train()

        if self._optimal:
            # Optimal action is of max probability (mu for continuous space)
            action = action_probabilities[0] if self._continuity else T.argmax(action_probabilities[0], dim=-1)
            log_probability = T.tensor([1.], dtype=T.float32)
        else:
            # Action based on probability distribution (mu and sigma used in continuous space)
            distribution = dist.Normal(action_probabilities[0], action_probabilities[1]) if self._continuity else dist.Categorical(action_probabilities)
            action = distribution.sample()
            log_probability = T.sum(distribution.log_prob(action), dim=-1)
        # Numpy Conversion
        action = self._get_numpy(action)
        log_probability = self._get_numpy(log_probability)
        
        return (action[0], log_probability[0]) if self._continuity else (int(action), log_probability)
    
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
        next_states = self._memory.get_rollouts("new_state")

        tensor_states = self._get_tensor(states, False).detach()
        tensor_next_states = self._get_tensor(next_states, False).detach()

        for _ in range(self._epoches):

            with T.no_grad():
                values = self._critic_network(tensor_states)
                next_values = self._critic_network(tensor_next_states)

            
            values, next_values = self._get_numpy(values.flatten()), self._get_numpy(next_values.flatten())

            advantages = self._get_advantage(values, next_values, rewards, terminals)
            del next_values
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            sample_size = len(states)
            space_permutation = np.random.permutation(np.arange(sample_size))
            index = 0

            while index < sample_size:

                end_index = index + self._minibatch_size
                if end_index >= sample_size:
                    end_index = sample_size - 1
                batch = range(index, end_index)

                permute = space_permutation[batch]

                state = self._get_tensor(states[permute])
                action = self._get_tensor(actions[permute])
                log_probability = self._get_tensor(log_probabilities[permute])
                advantage = self._get_tensor(advantages[permute])
                reward_return = self._get_tensor(returns[permute])

                value = self._critic_network(state).flatten()
                action_probabilities = self._actor_network(state)

                critic_loss = nn.MSELoss()(value, reward_return)
                loss = self._loss_function(action_probabilities, action, advantage, log_probability, critic_loss.detach())

                self._critic_network.optimizer.zero_grad()
                critic_loss.backward()
                self._critic_network.optimizer.step()

                self._actor_network.optimizer.zero_grad()
                loss.backward()
                self._actor_network.optimizer.step()
                
                self._updates_number += 1

                index += self._minibatch_size


    def _get_advantage(self, value, next_value, reward, terminal):
        deltas = reward + ((1 - terminal) * self._gamma * next_value) - value
        discounted_sum = np.zeros_like(deltas)
        running_add = 0

        # [V_0 + D*V_1 +...+D^(T-1) * V_T-1, ...,V_T-2+(D * V_T-1), V_T-1]
        for index in reversed(range(len(deltas))):
            running_add = deltas[index] + (1 - terminal[index]) * running_add * self._gamma * self._lambda
            discounted_sum[index] = running_add

        return discounted_sum


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
        # Might need to be removed as a separate network is doing this
        critic_loss = self._value_loss_ratio * critic_loss

        # c_2 S[pi_Theta](s_t) = c_2 Entropy Bonus To ensure sufficient exploration
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
episodes = 1024 # Steps per gradient
epoches = 32 # epoches of complete batch updates
batch_size = 256 # Batches of the step per decent
# 32 * (1024 / 256) = gradient steps = 128

optimal_run_episodes = 10
render_episode = -1
measure_after_iterations = 1
measurement = True

memory_order = ("state", "action", "log_probability", "terminal", "reward", "new_state")
gamma = 0.999 # good
lambda_gae = 0.9 # Decrease
value_ratio_loss = 0. # Might remain 0
entropy_loss_ratio = 0.005 # check
learning_rates = [0.00025, 0.0005] # If learning slowly increase
policy_clip = 0.15 # if updates are high ddecrease
continuity = True
optimal = False
env = gym.make("LunarLander-v2", 
         continuous=continuity
         #gravity=-10.0, 
         #enable_wind=False, 
         #wind_power=15.0, 
         #turbulence_power=1.5, 
         )

write_data(file_name_txt, """
           training_iterations = 10000
            episodes = 1024 # Steps per gradient
            epoches = 32 # epoches of complete batch updates
            batch_size = 256 # Batches of the step per decent
            # 32 * (1024 / 256) = gradient steps = 128

            optimal_run_episodes = 10
            render_episode = -1
            measure_after_iterations = 1
            measurement = True

            memory_order = ("state", "action", "log_probability", "terminal", "reward", "new_state")
            gamma = 0.999 # good
            lambda_gae = 0.9 # Decrease
            value_ratio_loss = 0. # Might remain 0
            entropy_loss_ratio = 0.01 # If exploring too much decrease
            learning_rates = [0.00025, 0.0005] # If learning slowly increase
            policy_clip = 0.15 # if updates are high ddecrease
            continuity = True
            optimal = False
            env = gym.make("LunarLander-v2", 
                    continuous=continuity
                    #gravity=-10.0, 
                    #enable_wind=False, 
                    #wind_power=15.0, 
                    #turbulence_power=1.5, 
                    )
           """)


agent = Agent(env,
              continuity=continuity, 
              optimal=optimal, 
              training_iterations=training_iterations,
              episodes=episodes,
              epoches=epoches,
              optimal_run_episodes=optimal_run_episodes, 
              render_episode=render_episode, 
              measure_after_iterations=measure_after_iterations, 
              measurement=measurement,
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