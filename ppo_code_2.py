import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam

class AgentMemory:
    def __init__(self, *timeline_keys):
        self.new_memory(timeline_keys if timeline_keys else ("state", "action", "terminal", "reward", "next_state", "log_probability", "value"))

    def new_memory(self, *timeline_keys):
        self._timeline_keys = timeline_keys if timeline_keys else self._timeline_keys
        self._timeline = {key: np.array() for key in self._timeline_keys}
        self._timeline["batch"] = np.array()
    
    def memorize(self, episode_trajectories):
        trajectories = np.array(episode_trajectories).T
        for key_index in range(len(self._timeline_keys)):
            key = self._timeline_keys[key_index]
            self._timeline[key] = np.concatenate((self._timeline[key], trajectories[key_index]))
        self._timeline["batch"] = np.append(self._timeline["batch"], (self._timeline["batch"][-1][1] if self._timeline["batch"] else 0, len(episode_trajectories)))
    
    def get_rollouts(self, key, section=None, randomize=None):
        if section:
            return self._timeline[key][section[0]:section[1]]
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
    def __init__(self, input_dim, output_dim, space_continuity=True, optimal=False, learning_rates=(0.01, 0.2), reward_discount=0.99, advantage_decay=0.95, policy_clip=0.2, training_iterations=20, memory_order=None):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._learning_rates = learning_rates
        self._reward_discount = reward_discount
        self._advantage_decay = advantage_decay
        self._policy_clip = policy_clip
        self._training_iterations = training_iterations
        self._continuity = space_continuity
        self._optimal = optimal

        activations = ("tanh", "softplus") if space_continuity else ("softmax",)
        input_dims = (input_dim, 1, 2 * input_dim) if space_continuity else (input_dim,)

        self._actor_network = Network(input_dims, output_dim, activations, learning_rate=learning_rates[0], loss_function=self._loss_function).build
        self._critic_network = Network((input_dim,), 1, learning_rate=learning_rates[1], loss_function=lambda *_: "mean_squared_error").build

        self._memory = AgentMemory(memory_order)
    
    def _loss_function(self, old_action_probabilities, advantage):
        
        if self._continuity:
            def log_probability_density(action_probabilities, action):
                mu = action_probabilities[:, 0:self._output_dim]
                sigma = action_probabilities[:, self._output_dim:]
                variance = keras.backend.square(sigma)
                probability_density = 1. / keras.backend.sqrt(2. * np.pi * variance) * keras.backend.exp(-keras.backend.square(action - mu) / (2. * variance))
                return keras.backend.log(probability_density + keras.backend.epsilon())
        else:
            log_probability_density = lambda action_probabilities, action: keras.backend.log(action_probabilities[:, action] + keras.backend.epsilon())
        
        def loss(action, action_probabilities):
            log_probability_density_current = log_probability_density(action_probabilities, action)
            log_probability_density_old = log_probability_density(old_action_probabilities, action)
            # Calc ratio and the surrogates
            # ratio = prob / (old_prob + K.epsilon()) #ratio new to old
            ratio = keras.backend.exp(log_probability_density_current - log_probability_density_old)
            weighted_probabilities = ratio * advantage
            clipped_probabilities = keras.backend.clip(ratio, min_value=1 - self._policy_clip, max_value=1 + self._policy_clip)
            weighted_clipped_probabilities = clipped_probabilities * advantage
            # loss is the mean of the minimum of either of the surrogates
            loss_actor = - keras.backend.mean(keras.backend.minimum(weighted_probabilities, weighted_clipped_probabilities))
            # entropy bonus in accordance with move37 explanation https://youtu.be/kWHSH2HgbNQ
            sigma = action_probabilities[:, self.self._output_dim:]
            variance = keras.backend.square(sigma)
            loss_entropy = self.ENTROPY_LOSS_RATIO * keras.backend.mean(-(keras.backend.log(2*np.pi*variance)+1) / 2)  # see move37 chap 9.5
            # total bonus is all losses combined. Add MSE-value-loss here as well?
            return loss_actor + loss_entropy
        return loss
        

    

    def _get_action(self, state):
        state = tf.convert_to_tensor([state])
        action_probabilities = self._actor_network(state)
        if self._optimal:
            action = action_probabilities[:self._output_dim]
            log_probability = 1
        else:
            distribution = tfp.distributions.Normal(action_probabilities[:self._output_dim], action_probabilities[self._output_dim:]) if self._continuity else tfp.distributions.Categorical(action_probabilities) 
            action = distribution.sample()
            log_probability = distribution.log_prob(action).numpy()[0]
        action = action.numpy()[0]
        value = self._critic_network(state).numpy()[0]
        return action, log_probability, value
    
    def _train(self):
        pass

    
    def calculate_advantage():
        delta = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(
            deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr
    
    def _learn(self):

        for index in range(self._memory.batch_count):

            # Twice Back Propagation
            with tf.GradientTape(persistent=True) as tape:
                states = tf.convert_to_tensor([self._memory.get_rollouts("state", "batch")[index]])
                probabilities = tf.convert_to_tensor([self._memory.get_rollouts("probability", "batch")[index]])
                actions = tf.convert_to_tensor([self._memory.get_rollouts("action", "batch")[index]])

                action_probabilities = self._actor_network(states)
                distributions = tfp.distributions.Categorical(action_probabilities)
                new_probabilities = distributions.log_prob(actions)

                value = tf.squeeze(self._critic_network(states), 1)

                probability_ratio = tf.math.exp(new_probabilities - probabilities)

                weighted_probabilities = advantage[index] * probability_ratio
                clipped_probabilities = tf.clip_by_value(probability_ratio, 1 - self._policy_clip, 1 + self._policy_clip)
                weighted_clipped_probabilities= clipped_probabilities * advantage[index]

                actor_loss = -tf.math.minimum(weighted_probabilities, weighted_clipped_probabilities)
                returns = self._memory.get_rollouts("value", "batch")[index] * advantage[index]
                critic_loss = keras.losses.MSE(value, returns)
            actor



