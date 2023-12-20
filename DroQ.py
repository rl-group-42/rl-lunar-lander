import os
import gym
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler
from torch.distributions import Normal
import torch.nn as nn

from collections import deque
import itertools
import math
import random

class Agent:

    def __init__(self, env, num_steps=30000000, batch_size=256,
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=1000, target_update_interval=1,
                 eval_interval=1000, cuda=0, seed=0,
                 eval_runs=1, huber=0, layer_norm=0,
                 method=None, target_entropy=None, target_drop_rate=0.0, critic_update_delay=1):
        self.env = env

        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False

        self.method = method
        self.critic_update_delay = critic_update_delay
        self.target_drop_rate = target_drop_rate

        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")

        # policy
        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)

        # Q functions
        kwargs_q = {"num_inputs": self.env.observation_space.shape[0],
                    "num_actions": self.env.action_space.shape[0],
                    "hidden_units": hidden_units,
                    "layer_norm": layer_norm,
                    "drop_rate": self.target_drop_rate}
        self.critic = TwinnedQNetwork(**kwargs_q).to(self.device)
        self.critic_target = TwinnedQNetwork(**kwargs_q).to(self.device)
        if self.target_drop_rate <= 0.0:
            self.critic_target = self.critic_target.eval()
        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)

        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        if entropy_tuning:
            if not (target_entropy is None):
                self.target_entropy = torch.prod(torch.Tensor([target_entropy]).to(self.device)).item()
            else:
                # Target entropy is -|A|.
                self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            # replay memory without prioritied experience replay
            self.memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step)

        self.train_rewards = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        #
        self.eval_runs = eval_runs
        self.huber = huber
        self.multi_step = multi_step

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state[0]).unsqueeze(0).to(self.device)
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
            if self.method == "sac":
                next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
            elif self.method == "duvn":
                next_q = next_q1 + self.alpha * next_entropies
            else:
                raise NotImplementedError()
        target_q = (rewards / (self.multi_step * 1.0)) + (1.0 - dones) * self.gamma_n * next_q
        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            action = self.act(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    state, action, reward, next_state, masked_done,
                    self.device)
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = (0.5 * torch.abs(curr_q1 - target_q) + 0.5 * torch.abs(curr_q2 - target_q)).item()
                self.memory.append(state, action, reward, next_state, masked_done, error, episode_done=done)
            else:
                self.memory.append(state, action, reward, next_state, masked_done, episode_done=done)
            if self.is_update():
                self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models()
            state = next_state

        self.train_rewards.append(episode_reward)

        self.train_rewards.append(episode_reward)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}')

    def learn(self):
        self.learning_steps += 1

        if (self.learning_steps - 1) % self.critic_update_delay == 0:
            for _ in range(self.updates_per_step):
                if self.per:
                    # batch with indices and priority weights
                    batch, indices, weights = self.memory.sample(self.batch_size)
                else:
                    batch = self.memory.sample(self.batch_size)
                    # set priority weights to 1 when we don't use PER.
                    weights = 1.


                q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch, weights)

                update_params(self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
                update_params(self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)

                if self.learning_steps % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)

                if self.per:
                    # update priority weights
                    self.memory.update_priority(indices, errors.cpu().numpy())

        if self.per:
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            weights = 1.

        policy_loss, entropies = self.calc_policy_loss(batch, weights) # added by tH 20210705
        update_params(self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

    def calc_critic_4redq_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        curr_qs = self.critic.allQs(states, actions)

        target_q = self.calc_target_q(*batch)

        errors = torch.abs(curr_qs[0].detach() - target_q)
        mean_q1 = curr_qs[0].detach().mean().item()
        mean_q2 = curr_qs[1].detach().mean().item()

        losses = []
        for curr_q in curr_qs:
            losses.append(torch.mean((curr_q - target_q).pow(2) * weights))
        return losses, errors, mean_q1, mean_q2

    def calc_critic_loss(self, batch, weights):

        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        errors = torch.abs(curr_q1.detach() - target_q)
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
        if self.method == "duvn":
            q2 = q1 # discard q2
        if self.target_drop_rate > 0.0:
            q = 0.5 * (q1 + q2)
        else:
            q = torch.min(q1, q2)

        policy_loss = torch.mean((- q - self.alpha * entropy) * weights)

        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropy).detach() * weights)
        return entropy_loss

    def evaluate(self):
        episodes = self.eval_runs
        returns = np.zeros((episodes,), dtype=np.float32)

        sar_buf = [[] for _ in range(episodes) ]
        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                print(next_state)
                episode_reward += reward
                state = next_state

                sar_buf[i].append([state, action, reward])

            returns[i] = episode_reward

        mean_return = np.mean(returns)

        mc_discounted_return = [deque() for _ in range(episodes) ]
        for i in range(episodes):
            for re_tran in reversed(sar_buf[i]):
                if len(mc_discounted_return[i]) > 0:
                    mcret = re_tran[2] + self.gamma_n * mc_discounted_return[i][0]
                else:
                    mcret = re_tran[2]
                mc_discounted_return[i].appendleft(mcret)
        norm_coef = np.mean(list(itertools.chain.from_iterable(mc_discounted_return)))
        norm_coef = math.fabs(norm_coef) + 0.000001
        norm_scores = [[] for _ in range(episodes)]
        for i in range(episodes):
            states = np.array(sar_buf[i], dtype="object")[:, 0].tolist()
            actions = np.array(sar_buf[i], dtype="object")[:, 1].tolist()
            with torch.no_grad():
                state = torch.FloatTensor(states).to(self.device)
                action = torch.FloatTensor(actions).to(self.device)
                q1, q2 = self.critic(state, action)
                q = 0.5 * (q1 + q2)
                qs = q.to('cpu').numpy()
            for j in range(len(sar_buf[i])):
                score = (qs[j][0] - mc_discounted_return[i][j]) / norm_coef
                norm_scores[i].append(score)
        # calculate std
        flatten_norm_score = list(itertools.chain.from_iterable(norm_scores))
        mean_norm_score = np.mean(flatten_norm_score)
        std_norm_score = np.std(flatten_norm_score)
        print("mean norm score " + str(mean_norm_score))
        print("std norm score " + str(std_norm_score))

def to_batch(state, action, reward, next_state, done, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


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
    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier', layer_norm=0, drop_rate=0.0):
        super(QNetwork, self).__init__()


        self.Q = self.create_linear_network(
            num_inputs+num_actions, 1, hidden_units=hidden_units,
            initializer=initializer)

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

    def create_linear_network(self, input_dim, output_dim, hidden_units=[],
                              hidden_activation='relu', output_activation=None,
                              initializer='xavier_uniform'):
        model = []
        units = input_dim
        for next_units in hidden_units:
            model.append(nn.Linear(units, next_units))
            model.append(self.str_to_activation[hidden_activation])
            units = next_units

        model.append(nn.Linear(units, output_dim))
        if output_activation is not None:
            model.append(self.str_to_activation[output_activation])

        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        return nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).apply(initialize_weights)

class TwinnedQNetwork(BaseNetwork):

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier', layer_norm=0, drop_rate=0.0):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(
            num_inputs, num_actions, hidden_units, initializer, layer_norm=layer_norm, drop_rate=drop_rate)
        self.Q2 = QNetwork(
            num_inputs, num_actions, hidden_units, initializer, layer_norm=layer_norm, drop_rate=drop_rate)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


class RandomizedEnsembleNetwork(BaseNetwork):

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier', layer_norm=0, drop_rate=0.0, N=10):
        super(RandomizedEnsembleNetwork, self).__init__()

        self.N = N
        self.indices = list(range(N))
        for i in range(N):
            setattr(self, "Q"+str(i),
                    QNetwork(num_inputs, num_actions, hidden_units, initializer,
                             layer_norm=layer_norm, drop_rate=drop_rate))

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        random.shuffle(self.indices)
        q1 = getattr(self, "Q" + str(self.indices[0]))(x)
        q2 = getattr(self, "Q" + str(self.indices[1]))(x)

        return q1, q2

    def allQs(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        Qs = []
        for i in range(self.N):
            Qs.append(getattr(self, "Q" + str(i))(x))
        return Qs

    def averageQ(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        q = getattr(self, "Q" + str(0))(x)
        for i in range(1, self.N):
            q = q + getattr(self, "Q" + str(i))(x)
        q = q / self.N

        return q




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
        state = np.array(state[0], dtype=self.state_type)
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
                slice(self._p, self._p+num_data), batch,
                slice(0, num_data))
        else:
            mid_index = self.capacity-self._p
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



class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier', layer_norm=0, drop_rate=0.0):
        super(GaussianPolicy, self).__init__()

        self.policy = self.create_linear_network(
            num_inputs, num_actions*2, hidden_units=hidden_units,
            initializer=initializer)


        # override network architecture (dropout and layer normalization). TH 20210731
        """new_q_networks = []
        for i, mod in enumerate(self.Q._modules.values()):
            new_q_networks.append(mod)
            if ((i % 2) == 0) and (i < (len(list(self.Q._modules.values()))) - 1):
                if drop_rate > 0.0:
                    new_q_networks.append(nn.Dropout(p=drop_rate))  # dropout
                if layer_norm:
                    new_q_networks.append(nn.LayerNorm(mod.out_features))  # layer norm
            i += 1
        self.Q = nn.Sequential(*new_q_networks)
        """
    def forward(self, states):
        mean = states[0].mean()
        log_std = states[0].std().log()
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, states):
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        xs = normals.rsample()
        actions = torch.tanh(xs)
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        print(log_probs)
        entropies = -log_probs.sum()

        return actions, entropies, torch.tanh(means)


    def create_linear_network(self, input_dim, output_dim, hidden_units=[],
                              hidden_activation='relu', output_activation=None,
                              initializer='xavier_uniform'):

        conv_layer = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        nn.init.xavier_uniform_(conv_layer.weight, gain=1)
        nn.init.constant_(conv_layer.bias, 0)
        return conv_layer


class PrioritizedMemory(MultiStepMemory):

    def __init__(self, capacity, state_shape, action_shape, device,
                 gamma=0.99, multi_step=3, alpha=0.6, beta=0.4,
                 beta_annealing=0.001, epsilon=1e-4):
        super(PrioritizedMemory, self).__init__(
            capacity, state_shape, action_shape, device, gamma, multi_step)
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def append(self, state, action, reward, next_state, done, error,
               episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if len(self.buff) == self.multi_step:
                state, action, reward = self.buff.get(self.gamma)
                self.priorities[self._p] = self.calc_priority(error)
                self._append(state, action, reward, next_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self.priorities[self._p] = self.calc_priority(error)
            self._append(state, action, reward, next_state, done)

    def update_priority(self, indices, errors):
        self.priorities[indices] = np.reshape(
            self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        self.beta = min(1. - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(
            self.priorities[:self._n, 0], batch_size)
        indices = list(sampler)
        batch = self._sample(indices)

        p = self.priorities[indices] / np.sum(self.priorities[:self._n])
        weights = (self._n * p) ** -self.beta
        weights /= np.max(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        return batch, indices, weights

    def reset(self):
        super(PrioritizedMemory, self).reset()
        self.priorities = np.empty(
            (self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid], self.actions[valid], self.rewards[valid],
            self.next_states[valid], self.dones[valid], self.priorities[valid])

    def _insert(self, mem_indices, batch, batch_indices):
        states, actions, rewards, next_states, dones, priorities = batch
        self.states[mem_indices] = states[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
        self.priorities[mem_indices] = priorities[batch_indices]


env = gym.make("LunarLander-v2", continuous= True, render_mode="human")
n_games = 100
best_score = float('-inf')
episode_score_history = []
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = Agent(env)
agent.run()
