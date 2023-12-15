import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam
import gymnasium as gym

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
            self._timeline["batch_index"] = np.append(self._timeline["batch_index"], self._timeline["batch_index"] + 1 if self._timeline["batch_index"].size else 0)
    
    def get_rollouts(self, key, section=None, randomize=False):
        if section is not None:
            section = section if isinstance(section[0], int) else (range(s[0], s[1]) for s in section)
            return np.array([element for s in section for element in self._timeline[key][s]])
        if randomize:
            return np.random.permutation(self._timeline[key])
        return self._timeline[key]
    
class Network:
    def __init__(self, input_dims, output_dim, output_activations=(None,), hidden_dims=(64, 32), hidden_activations="relu", learning_rate=0.01, loss_function=lambda *_: None):
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._hidden_activations = [hidden_activations for _ in hidden_dims] if isinstance(hidden_activations, str) else hidden_activations
        self._output_dim = output_dim
        self._output_activations = output_activations
        self._learning_rate = learning_rate
        self._loss_function = loss_function
    
    @property
    def build(self):
        input_layer = list(Input(shape=dim) for dim in self._input_dims)

        layer = input_layer[0]
        for dim, activation in zip(self._hidden_dims, self._hidden_activations):
            layer = Dense(dim, activation=activation)(layer)
        
        output_layer = Concatenate()(list(Dense(self._output_dim, activation=activation)(layer) for activation in self._output_activations))

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=self._learning_rate), loss=self._loss_function(*input_layer[1:])) #?
        model.summary()
        return model

class Agent:
    def __init__(self, env, continuity=True, optimal=False, gamma=0.9, lambda_gae=0.95, 
                 value_ratio_loss=0.001, entropy_loss_ratio=0.001,
                 learning_rates=(0.01, 0.2), policy_clip=0.2, training_iterations=1000, 
                 episodes=20, batch_size=15, memory_order=None):

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

        # Memory
        self._batch_size = batch_size
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

        activations = ("tanh", "softplus") if continuity else ("softmax",)
        input_dims = (self._input_dim, 1, 1, 1, self._output_dim, self._output_dim) if continuity else (self._input_dim,)

        self._actor_network = Network(input_dims, self._output_dim, activations, learning_rate=learning_rates[0], loss_function=self._loss_function).build
        self._old_actor_network = Network(input_dims, self._output_dim, activations, learning_rate=learning_rates[0], loss_function=self._loss_function).build
        self._critic_network = Network((self._input_dim,), 1, learning_rate=learning_rates[1], loss_function=lambda *_: "mean_squared_error").build

        self._empty_filler = lambda tensor: list(np.empty((len(tensor), dim)) for dim in input_dims[1:])
    
    def _loss_function(self, advantage, value, reward_return, old_log_probabilities, current_log_probabilities):
        def loss():
            # Surrogate losses 
            # L_t^CLIP+VF+S = E_t[L_t^CLIP(Theta) - c_1 L_t^VF(Theta) + c_2 S[pi_Theta](s_t)]

            # Probability Ratio: r_t(Theta) = pi_Theta(a|s) / pi_Theta_k(a|s) = e^(pi_Theta(a|s) - pi_Theta_k(a|s))
            ratio = keras.backend.exp(current_log_probabilities - old_log_probabilities)

            # Importance Sampling Ratio: r_t(Theta) * A_t
            weighted_probabilities = ratio * advantage

            # Simplified Clipping g(epsilon, A_t) = ((1 + epsilon) if A_t >= 0 else if A_t < 0 (1 - epsilon)) * A_t
            clipped_probabilities = 1 - self._policy_clip if advantage < 0 else 1 + self._policy_clip
            weighted_clipped_probabilities = clipped_probabilities * advantage

            # L_t^CLIP(Theta) = E_t[min(r_t(Theta)A_t, g(epsilon, A_t)]
            # Negative indicating loss
            actor_loss = - keras.backend.mean(keras.backend.minimum(weighted_probabilities, weighted_clipped_probabilities))
            
            # c_1 L_t^VF(Theta) = c_1 MSE = c_1 (V_Theta(s_t) - V_t^target)^2
            # Negative sign is not used since MSE is being used
            critic_loss = self._value_loss_ratio * keras.losses.MSE(value, reward_return)

            # c_2 S[pi_Theta](s_t) = c_2 Entropy Bonus
            if self._continuity:
                sigma = current_log_probabilities[:, self._output_dim:]
                variance = keras.backend.square(sigma)
                entropy_loss = -self._entropy_loss_ratio * keras.backend.mean((keras.backend.log(2 * np.pi * variance) + 1) / 2)  
            else:
                probabilities = keras.backend.softmax(current_log_probabilities)  
                entropy_loss = -self._entropy_loss_ratio * keras.backend.mean(keras.backend.sum(probabilities * keras.backend.log(probabilities + keras.backend.epsilon()), axis=-1))
            
            return actor_loss + critic_loss + entropy_loss
        return loss
        
    def train_run(self):
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
                
                # Memorize
                self._memory.memorize(trajectory, self._memory_order)

                batch_section = self._memory.get_rollouts("batch")[-1]
                rewards = self._memory.get_rollouts("reward", [batch_section])
                last_reward = reward

                # Get Critic Values
                states = self._memory.get_rollouts("current_state", [batch_section])
                states = tf.convert_to_tensor(states)
                values = self._critic_network.predict(states).flatten()

                # Compute Advantage and Return
                advantage_returns = self._advantages_returns_calculation(rewards, values, last_reward)

                # Memorize Extensions
                self._memory.memorize((values,) + advantage_returns, self._memory_extension, record_batch=False, transpose=False)

            # Train Models
            self._train_network()

            # Forget
            self._memory.new_memory()
            
            if iteration % 10 == 0:
                print(f"Iteration:{iteration}, total_reward:{total_reward}")

    

    def _get_action(self, state):
        # Convert to Tensor
        state = tf.convert_to_tensor([state])

        # Network Prediction of Action Probabilities pi_Theta_k(a_t|s_t)
        action_probabilities = self._actor_network.predict_on_batch([state, *self._empty_filler(state)])

        if self._optimal:

            # Optimal action is of max probability (mu for continous space)
            action = action_probabilities[0, 0:self._output_dim] if self._continuity else tf.math.argmax(action_probabilities)
            log_probability = np.ones_like(action)
        else:

            # Action based on probability distribution (mu and sigma used in continous space)
            distribution = tfp.distributions.Normal(action_probabilities[0, 0:self._output_dim], action_probabilities[0, self._output_dim:]) if self._continuity else tfp.distributions.Categorical(action_probabilities) 
            action = distribution.sample()
            log_probability = distribution.log_prob(action).numpy()

        # Numpy Conversion
        action = action.numpy()
        
        return action, log_probability
    
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
        batch_indicies = self._memory.get_rollouts("batch_index", randomize=True)[:self._batch_size]
        batches = self._memory.get_rollouts("batch")[batch_indicies]
        states = self._memory.get_rollouts("current_state", batches)
        actions = self._memory.get_rollouts("action", batches)
        advantages = self._memory.get_rollouts("advantage", batches)
        returns = self._memory.get_rollouts("return", batches)
        values = self._memory.get_rollouts("value", batches)
        log_probability = self._memory.get_rollouts("log_probability", batches)

        advantages = tf.convert_to_tensor(advantages)
        advantages = tf.squeeze(keras.utils.normalize(advantages))

        old_log_probability = self._old_action_log_probability_density(states, actions)

        self._actor_network.fit(
            x=[states, advantages, values, returns, old_log_probability, log_probability], verbose=0)
        self._critic_network.fit(
            x=states, y=returns, epochs=1, verbose=0)
        
        self._update_networks()

    def _old_action_log_probability_density(self, states, actions):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        action_probabilities = self._old_actor_network.predict([states, *self._empty_filler(states)])
        distribution = tfp.distributions.Normal(action_probabilities[0, 0:self._output_dim], action_probabilities[0, self._output_dim:]) if self._continuity else tfp.distributions.Categorical(action_probabilities)
        return distribution.log_prob(actions).numpy()
    
    def _update_networks(self):
        """Softupdate of the target network.
        In ppo, the updates of the 
        """
        alpha = self._learning_rates[0]
        actor_weights = np.array(self._actor_network.get_weights())
        actor_tartget_weights = np.array(self._old_actor_network.get_weights())
        new_weights = alpha*actor_weights + (1-alpha)*actor_tartget_weights
        self._old_actor_network.set_weights(new_weights)

training_iterations = 1000
episodes = 5
batch_size = int(episodes * 0.9)
memory_order = ("current_state", "action", "log_probability", "terminal", "reward", "new_state")
gamma = 0.9
lambda_gae = 0.95
value_ratio_loss = 0.001
entropy_loss_ratio = 0.001
learning_rates = [0.95, 0.9]
policy_clip = 0.1
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
              batch_size=batch_size,
              gamma=gamma,
              lambda_gae=lambda_gae,
              value_ratio_loss=value_ratio_loss,
              entropy_loss_ratio=entropy_loss_ratio,
              learning_rates=learning_rates,
              policy_clip=policy_clip,
              memory_order=memory_order)
agent.train_run()