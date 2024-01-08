import gym
import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import Normal
import torch.nn as nn
from collections import deque
import time


# this code is an implementation of the algorithm DroQ as described in the paper
# Dropout Q-Functions for Doubly Efficient Reinforcement Learning by
# Takuya Hiraoka et al.
# some inspiration was taken from Hiraoka's GitHub page found at
# https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning


# main agent
class Agent:
    # gamma = Discount factor
    # tau = network soft update param
    def __init__(self, env, max_steps=500000, batch_size=256,
                 learning_rate=0.001, hidden_units=[256, 256], memory_size=1e6,
                 gamma=0.98, tau=0.01, multi_step=1, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=4000, eval_runs=100, layer_norm=0,
                 target_entropy=None, target_drop_rate=0.0, critic_update_delay=1):

        self.env = env
        self.critic_update_delay = critic_update_delay
        self.target_drop_rate = target_drop_rate
        self.device = torch.device("cpu")

        # policy
        self.policy = Actor(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)

        # Q functions
        qfuncs = {"input_dim": self.env.observation_space.shape[0],
                  "output_dim": self.env.action_space.shape[0],
                  "hidden_units": hidden_units,
                  "layer_norm": layer_norm,
                  "drop_rate": self.target_drop_rate}
        self.critic = TwinnedQNetwork(**qfuncs).to(self.device)
        self.critic_target = TwinnedQNetwork(**qfuncs).to(self.device)
        if self.target_drop_rate <= 0.0:
            self.critic_target = self.critic_target.eval()
        # copy parameters of the learning network to the target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        # disable gradient calculations of the target network
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # optimizer
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=learning_rate)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=learning_rate)

        if target_entropy is None:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        else:
            self.target_entropy = torch.prod(torch.Tensor([target_entropy]).to(self.device)).item()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)

        self.memory = MultiStepMemory(
            memory_size, self.env.observation_space.shape,
            self.env.action_space.shape, self.device, gamma, multi_step)

        self.train_rewards = RunningMeanStats(log_interval)
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.max_steps = max_steps
        self.tau = tau
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.eval_runs = eval_runs
        self.multi_step = multi_step

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.max_steps:
                self.env = gym.wrappers.RecordVideo(self.env, 'video', episode_trigger=lambda episode: episode)
                self.eval_runs = 100
                self.evaluate()
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def act(self, state):
        if self.steps < self.start_steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = next_q1 + self.alpha * next_entropies  # discard q2
        # rescale rewards by num step
        target_q = (rewards / (self.multi_step * 1.0)) + (1.0 - dones) * self.gamma_n * next_q
        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()[0]

        while not done:
            action = self.act(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward
            if episode_steps >= 1000:  # max episode steps
                masked_done = False
            else:
                masked_done = done
                self.memory.append(state, action, reward, next_state, masked_done, episode_done=done)

            if len(self.memory) > self.batch_size and self.steps >= self.start_steps:
                self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
            state = next_state

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        logger.episodes.append([self.steps, episode_steps, episode_reward])

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}')

    def learn(self):
        self.learning_steps += 1

        # critic update
        if (self.learning_steps - 1) % self.critic_update_delay == 0:
            for _ in range(self.updates_per_step):
                batch = self.memory.sample(self.batch_size)
                weights = 1.
                q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch, weights)

                update_params(self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
                update_params(self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)

                if self.learning_steps % self.target_update_interval == 0:
                    for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
                        t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)

        # policy and alpha update
        batch = self.memory.sample(self.batch_size)
        weights = 1.

        policy_loss, entropies = self.calc_policy_loss(batch, weights)  # added by tH 20210705
        update_params(self.policy_optim, self.policy, policy_loss, self.grad_clip)

        entropy_loss = self.calc_entropy_loss(entropies, weights)
        update_params(self.alpha_optim, None, entropy_loss)
        self.alpha = self.log_alpha.exp()

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(states, sampled_action)
        q2 = q1  # discard q2
        if self.target_drop_rate > 0.0:
            q = 0.5 * (q1 + q2)
        else:
            q = torch.min(q1, q2)
        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = torch.mean((- q - self.alpha * entropy) * weights)

        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropy).detach() * weights)
        return entropy_loss

    def evaluate(self):
        episodes = self.eval_runs
        returns = np.zeros((episodes,), dtype=np.float32)
        for i in range(episodes):
            state = self.env.reset()[0]
            episode_reward = 0.
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = self.exploit(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                state = next_state
            returns[i] = episode_reward
            logger.optimaleps.append([self.steps, episode_reward])
        mean_return = np.mean(returns)

        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'reward: {mean_return:<5.1f}')
        print('-' * 60)

        logger.optimal.append([self.steps, mean_return])


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class QNetwork(BaseNetwork):
    def __init__(self, input_dim, output_dim, hidden_units=[], layer_norm=0, drop_rate=0.0):
        super(QNetwork, self).__init__()

        layer = []
        units = input_dim + output_dim

        for next_units in hidden_units:
            layer.append(nn.Linear(units, next_units))
            layer.append(nn.ReLU())
            units = next_units

        self.output = layer.append(nn.Linear(units, 1))

        self.Q = nn.Sequential(*layer)

        # override network architecture (dropout and layer normalization).
        new_q_networks = []
        for i, mod in enumerate(self.Q._modules.values()):
            new_q_networks.append(mod)
            if ((i % 2) == 0) and (i < (len(list(self.Q._modules.values()))) - 1):
                if drop_rate > 0.0:
                    new_q_networks.append(nn.Dropout(p=drop_rate))  # dropout
                if layer_norm:
                    new_q_networks.append(nn.LayerNorm(mod.out_features))  # layer norm
            i += 1
        self.Q = nn.Sequential(*new_q_networks)

    def forward(self, x):
        q = self.Q(x)
        return q

    def initialize_weights(self, initializer):
        def initialize(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                initializer(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        return initialize


class TwinnedQNetwork(BaseNetwork):

    def __init__(self, input_dim, output_dim, hidden_units=[],
                 initializer='xavier', layer_norm=0, drop_rate=0.0):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(
            input_dim, output_dim, hidden_units, layer_norm=layer_norm, drop_rate=drop_rate)
        self.Q2 = QNetwork(
            input_dim, output_dim, hidden_units, layer_norm=layer_norm, drop_rate=drop_rate)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


class MultiStepBuff:
    keys = ["state", "action", "reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.memory = {
            key: deque(maxlen=self.maxlen)
            for key in self.keys
        }

    def append(self, state, action, reward):
        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        return state, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([
            r * (gamma ** i) for i, r
            in enumerate(self.memory["reward"])])

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in MultiStepBuff.')
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory['state'])


class Memory:

    def __init__(self, capacity, state_shape, action_shape, device):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.is_image = len(state_shape) == 3
        self.state_type = np.uint8 if self.is_image else np.float32

        self.reset()

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=self.state_type)
        next_state = np.array(next_state, dtype=self.state_type)

        self.states[self._p] = state
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.next_states[self._p] = next_state
        self.dones[self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return self._sample(indices)

    def _sample(self, indices):
        if self.is_image:
            states = self.states[indices].astype(np.uint8)
            next_states = self.next_states[indices].astype(np.uint8)
            states = \
                torch.ByteTensor(states).to(self.device).float() / 255.
            next_states = \
                torch.ByteTensor(next_states).to(self.device).float() / 255.
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)

        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n

    def reset(self):
        self._n = 0
        self._p = 0

        self.states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type)
        self.actions = np.empty(
            (self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty(
            (self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type)
        self.dones = np.empty(
            (self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid], self.actions[valid], self.rewards[valid],
            self.next_states[valid], self.dones[valid])

    def load(self, batch):
        num_data = len(batch[0])

        if self._p + num_data <= self.capacity:
            self._insert(
                slice(self._p, self._p + num_data), batch,
                slice(0, num_data))
        else:
            mid_index = self.capacity - self._p
            end_index = num_data - mid_index
            self._insert(
                slice(self._p, self.capacity), batch,
                slice(0, mid_index))
            self._insert(
                slice(0, end_index), batch,
                slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, mem_indices, batch, batch_indices):
        states, actions, rewards, next_states, dones = batch
        self.states[mem_indices] = states[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]


class MultiStepMemory(Memory):

    def __init__(self, capacity, state_shape, action_shape, device,
                 gamma=0.99, multi_step=3):
        super(MultiStepMemory, self).__init__(
            capacity, state_shape, action_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done,
               episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if len(self.buff) == self.multi_step:
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self._append(state, action, reward, next_state, done)


# actor network using gaussian distribution
class Actor(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, input_dim, output_dim, hidden_units=[]):
        super(Actor, self).__init__()

        model = []
        units = input_dim

        for next_units in hidden_units:
            model.append(nn.Linear(units, next_units))
            model.append(nn.ReLU())
            units = next_units

        self.output = model.append(nn.Linear(units, output_dim * 2))

        self.network = nn.Sequential(*model).apply(
            self.initialize_weights(nn.init.xavier_uniform_))

    def forward(self, states):
        mean, log_std = torch.chunk(self.network(states), 2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, states):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        log_probs = normals.log_prob(xs) \
                    - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum()
        return actions, entropies, torch.tanh(means)

    def initialize_weights(self, initializer):
        def initialize(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                initializer(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        return initialize


class Logger:
    episodes = []
    optimal = []
    optimaleps = []

    def __init__(self):
        self.episodes.append(["t", "length", "reward"])
        self.optimal.append(["t", "avg"])
        self.optimaleps.append(["t", "reward"])

    def export(self, agent):
        name = "tests\\" + str(int(time.time()))
        with open(name + "_eps.csv", "a+") as file:
            for ep in self.episodes:
                file.write(",".join([str(x) for x in ep]) + "\n")
        with open(name + "_optim.csv", "a+") as file:
            for data in self.optimal:
                file.write(",".join([str(x) for x in data]) + "\n")
        with open(name + "_optim_eps.csv", "a+") as file:
            for data in self.optimaleps:
                file.write(",".join([str(x) for x in data]) + "\n")


env = gym.make("LunarLander-v2", continuous=True, render_mode="rgb_array")

agent = Agent(env)
logger = Logger()
agent.run()
logger.export(agent)
env.close()

env.close()
