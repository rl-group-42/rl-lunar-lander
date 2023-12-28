import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import torch.distributions as dist
import torch.nn.functional as F
from torch import cuda
from torch.backends import mps
from collections import deque
import matplotlib.pyplot as plt

def get_device() -> str:
    """ Get Device Name """
    if cuda.is_available():
        return "cuda"
    elif mps.is_available():
        return "mps"
    else:
        return "cpu"
    
def typeprint(var):
    print(type(var))
    print(var)

# Stores experiences from the agent
# State - current state
# Action - action performed in that state
# Reward - reward gained from taking the action in the state
# Next State - the state resulting from taking the action from that state
# Done - done signal, true if the next state is terminal, if both legs are touched down
class ReplayBuffer:
    def __init__(self):
        self.buffer = []
    
    def new_memory(self, state, action, reward, next, done):
        self.buffer.append((state, action, reward, next, done))

    # returns a random permutation of a range with the length of the buffer
    def rand_key(self):
        memories = range(len(self.buffer))
        return np.random.permutation(np.array(memories))

    # returns a random selection of the memory
    def rand_sample(self, size):
        sample = []
        for i in self.rand_key()[:size]:
            sample.append(self.buffer[i])
        return sample
    
    def rand_sample_split(self, size):
        state, action, reward, next, done = [], [], [], [], []
        for item in self.rand_sample(size):
            state.append(item[0])
            action.append(item[1])
            reward.append(item[2])
            next.append(item[3])
            done.append(item[4])
        return np.array(state), np.array(action), np.array(reward), np.array(next), np.array(done)

    def rand_batches(self, size):
        key = self.rand_key()
        memories = len(key)
        index = 0
        batches = []
        while index < memories:
            batch = []
            for _ in range(min(size, memories - index)):
                batch.append(self.buffer[key[index]])
            batches.append(batch)



class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate):
        super(ActorNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.learning_rate = learning_rate

        # Set up layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim
        self.network = nn.Sequential(*layers)
        self.mean_layer = nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Tanh())
        self.std_layer = nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softplus())

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

    # gets a mean and standard deviation value for each action
    def forward(self, state):
        output = self.network(state)
        # typeprint(output)
        mean = self.mean_layer(output)
        # typeprint(mean)

        std = self.std_layer(output)

        return mean, std
    
    def sample_normal(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        sample = dist.sample()
        action = torch.tanh(sample)
        log_prob = dist.log_prob(sample)
        return action, log_prob # may need to do like cpu.detach.numpy
    
class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, learning_rate):
        super(CriticNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self.learning_rate = learning_rate

        # Set up layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

    def forward(self, input):
        output = self.network(input)
        return output


class SACAgent:
    def __init__(self, env: gym.Env, batch_size, discount, polyak, entropy):

        self.env = env
        self.state_space = env.observation_space
        self.action_space = env.action_space

        # size of minibatch when updating neuralnets
        self.batch_size = batch_size
        # # number of episodes in a training run
        # self.episodes = episodes
        # # when to start training
        # self.start_training = start_training
        # proportion of updated target network parameters kept on update
        self.polyak = polyak
        # entropy magnitude in loss
        self.entropy = entropy

        # future reward discount
        self.discount = discount

        self.memory = ReplayBuffer()

        self.device = torch.device(get_device())
        self.input_dim = self.state_space.shape[0]
        self.output_dim = self.action_space.shape[0]
        self.actor = ActorNet(self.input_dim, [128], self.output_dim, 0.01).to(self.device)
        self.critic1 = CriticNet(self.input_dim + self.output_dim, [128], 0.01).to(self.device)
        self.critic2 = CriticNet(self.input_dim + self.output_dim, [128], 0.01).to(self.device)
        self.critic1tar = CriticNet(self.input_dim + self.output_dim, [128], 0.01).to(self.device)
        self.critic2tar = CriticNet(self.input_dim + self.output_dim, [128], 0.01).to(self.device)

        # print(self.actor_net, self.critic_net)

    def get_tensor (self, array):
        return torch.from_numpy(array).float().to(self.device)
    
    def get_numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def get_action(self, state):
        state_tensor = self.get_tensor(state)
        action, std = self.actor.sample_normal(state_tensor)
        return self.get_numpy(action), self.get_numpy(std)
    
    def train(self, maxsteps):
        print("Starting episode 1 at timestep 0", end="")
        reward_hist = []
        total_reward = 0
        terminal = True
        for t in range(maxsteps):
            if terminal:
                current_state = self.env.reset()[0]
                reward_hist.append(total_reward)
                total_reward = 0
                print("\rStarting episode " + str(len(reward_hist)+1) + " at timestep " + str(t), end="")
            
            action, _ = self.get_action(current_state)
            next_state, reward, isterminal, truncated, _ = self.env.step(action)
            total_reward += reward
            terminal = 1 if isterminal or truncated else 0
            self.memory.new_memory(current_state, action, reward, next_state, terminal)
            current_state = next_state
            if t > self.batch_size:
                self.train_networks()
        print()
        return reward_hist

    def get_crit_in(self, action, state):
        return self.get_tensor(np.concatenate((action, state), axis=1))

    # memory: 0 state, 1 action, 2 reward, 3 next, 4 done
    def train_networks(self):
        states, actions, rewards, nexts, dones = self.memory.rand_sample_split(self.batch_size)
        # targets = []

        self.actor.optimizer.zero_grad()
        sample_actions, log_probs = self.get_action(nexts) # maybe need to get the tensor version for log probs
        log_probs = log_probs.sum(1)
        next1 = self.critic1tar.forward(self.get_crit_in(sample_actions, nexts))
        next2 = self.critic2tar.forward(self.get_crit_in(sample_actions, nexts))
        next_estimate = self.get_numpy(torch.min(next1, next2)).flatten()
        targets = rewards + self.discount*(1-dones) * \
            (next_estimate - (self.entropy * log_probs))
        targets = self.get_tensor(targets.reshape(-1, 1))

        self.critic1.optimizer.zero_grad()
        loss1 = self.critic1.forward(self.get_crit_in(actions, states))
        loss1 = torch.nn.functional.mse_loss(loss1, targets)
        loss1.backward()
        self.critic1.optimizer.step()

        self.critic2.optimizer.zero_grad()
        loss2 = self.critic2.forward(self.get_crit_in(actions, states))
        loss2 = torch.nn.functional.mse_loss(loss2, targets)
        loss2.backward()
        self.critic2.optimizer.step()

        curr1 = self.critic1.forward(self.get_crit_in(sample_actions, states))
        curr2 = self.critic2.forward(self.get_crit_in(sample_actions, states))
        curr = torch.min(curr1, curr2).flatten()
        log_probs_ten = self.get_tensor(log_probs)
        actorloss = torch.mean(curr - (self.entropy * log_probs_ten))
        actorloss.backward()
        self.actor.optimizer.step()
        self.polyak_update()

    def polyak_update(self):
        for target, real in zip(self.critic1tar.parameters(), self.critic1.parameters()):
            target.data.copy_(self.polyak * target.data + (1-self.polyak) * real.data)

        for target, real in zip(self.critic2tar.parameters(), self.critic2.parameters()):
            target.data.copy_(self.polyak * target.data + (1-self.polyak) * real.data)

    
    # env: gym.Env, batch_size, discount, start_training, polyak, entropy
def main():
    env = gym.make("LunarLander-v2", 
         continuous=True)
    # print(env.action_space)
    # print(env.action_space.shape)
    # print(env.observation_space)
    # print(env.observation_space.shape[0])
    # print(env.reset()[0])
    # print(type(env.reset()[0]))
    agent = SACAgent(env, 5, 0.99, 0.99, 0.01)
    rewards = agent.train(100000)
    # print(rewards)
    print(len(rewards))
    plt.plot(rewards)
    plt.show()
    # print(len(agent.memory.buffer))
    # print(agent.get_action(env.reset()[0]))

if __name__ == "__main__":
    main()