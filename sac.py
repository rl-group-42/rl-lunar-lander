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
            # layers.append(nn.LeakyReLU())
            layers.append(nn.ReLU())
            prev_dim = dim
        self.network = nn.Sequential(*layers)
        # self.mean_layer = nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Tanh())
        self.mean_layer = nn.Linear(prev_dim, output_dim)
        self.std_layer = nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softplus())

        # self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.optimizer = optim.SGD(self.parameters(), lr = learning_rate)

    # gets a mean and standard deviation value for each action
    def forward(self, state):
        output = self.network(state)
        mean = self.mean_layer(output)
        if torch.isnan(mean[0][0]):
            print("||ACTOR FORWARD PASS||")
            print(state)
            typeprint(output)
            typeprint(mean)
            for param in self.parameters():
                print(param)
        
        for paramin in self.parameters():
            # print("parameters")
            end = False
            param = paramin
            while len(param.shape) > 0:
                param = param[0]
            # print(len(param[0].shape))
            # print(param[0][0])
            if param.isnan():
                print(paramin)
                end = True
            if end: print("Params End")


        std = torch.clamp(self.std_layer(output), 0.01, 1)

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

        self.optimizer = optim.SGD(self.parameters(), lr = learning_rate)

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
        self.valloss = []
        self.crit1loss = []
        self.crit2loss = []
        self.actorloss = []

        self.device = torch.device(get_device())
        self.input_dim = self.state_space.shape[0]
        self.output_dim = self.action_space.shape[0]
        # hidden_layers = [128]
        hidden_layers = [128, 128]
        self.actor = ActorNet(self.input_dim, hidden_layers, self.output_dim, 0.001).to(self.device)
        self.value = CriticNet(self.input_dim, hidden_layers, 0.001).to(self.device)
        self.valuetar = CriticNet(self.input_dim, hidden_layers, 0.001).to(self.device)
        self.critic1 = CriticNet(self.input_dim + self.output_dim, hidden_layers, 0.001).to(self.device)
        self.critic2 = CriticNet(self.input_dim + self.output_dim, hidden_layers, 0.001).to(self.device)

        # print(self.actor_net, self.critic_net)

    def get_tensor (self, array):
        return torch.from_numpy(array).float().to(self.device)
    
    def get_numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def get_action(self, state):
        state_tensor = self.get_tensor(state)
        action, std = self.actor.sample_normal(state_tensor)
        return action, std
    
    def get_single_action(self, state):
        state_tensor = self.get_tensor(state).unsqueeze(0)
        action, std = self.actor.sample_normal(state_tensor)
        return self.get_numpy(action.squeeze()), std
    
    def train(self, maxsteps):
        print("Starting episode 1 at timestep 0", end="")
        reward_hist = []
        total_reward = 0
        terminal = True
        for t in range(maxsteps):
            if terminal:
                current_state = self.env.reset(seed=1)[0]
                reward_hist.append(total_reward)
                total_reward = 0
                print("\rStarting episode " + str(len(reward_hist)+1) + " at timestep " + str(t) + " previous score: " + str(reward_hist[-1]), end="")
            
            action, _ = self.get_single_action(current_state)
            next_state, reward, isterminal, truncated, _ = self.env.step(action)
            total_reward += reward
            terminal = 1 if isterminal or truncated else 0
            self.memory.new_memory(current_state, action, reward, next_state, terminal)
            current_state = next_state
            if t > self.batch_size and t%50 == 0:
                self.train_networks()
        print()
        return reward_hist

    def get_crit_in(self, action, state):
        return self.get_tensor(np.concatenate((action, state), axis=1))

    # memory: 0 state, 1 action, 2 reward, 3 next, 4 done
    def train_networks(self):
        states, actions, rewards, nexts, dones = self.memory.rand_sample_split(self.batch_size)
        states = self.get_tensor(states)
        actions = self.get_tensor(actions)
        rewards = self.get_tensor(rewards).unsqueeze(1)
        nexts = self.get_tensor(nexts)
        dones = self.get_tensor(dones)

        # getting sample actions using the current states
        s_actions, s_log_probs = self.actor.sample_normal(states)
        s_log_probs = torch.sum(s_log_probs, dim=1).unsqueeze(1)

        # getting value targets using critic q functions
        self.value.optimizer.zero_grad()
        sq_in = torch.cat((s_actions, states), 1)
        q_val_1 = self.critic1.forward(sq_in)
        q_val_2 = self.critic2.forward(sq_in)
        q_val = torch.min(q_val_1, q_val_2)
        val_targets = q_val - self.entropy * s_log_probs

        # getting value values and optimizing using value targets
        val = self.value.forward(states)
        val_loss = torch.nn.functional.mse_loss(val, val_targets)
        self.valloss.append(self.get_numpy(val_loss))
        val_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # getting q function targets using target value network
        n_val = self.valuetar.forward(nexts)
        q_targets = rewards + self.discount * n_val
        q_in = torch.cat((actions, states), 1)

        # optimizing first critic network
        self.critic1.optimizer.zero_grad()
        q_val1 = self.critic1.forward(q_in)
        q_loss1 = torch.nn.functional.mse_loss(q_val1, q_targets)
        self.crit1loss.append(self.get_numpy(q_loss1))
        q_loss1.backward(retain_graph=True)
        self.critic1.optimizer.step()

        # optimizing second critic network
        self.critic2.optimizer.zero_grad()
        q_val2 = self.critic2.forward(q_in)
        q_loss2 = torch.nn.functional.mse_loss(q_val2, q_targets)
        self.crit2loss.append(self.get_numpy(q_loss2))
        q_loss2.backward()
        self.critic2.optimizer.step()

        # optimizing actor network
        self.actor.optimizer.zero_grad()
        s_actions2, s_log_probs2 = self.actor.sample_normal(states)
        s_log_probs2 = torch.sum(s_log_probs2, dim=1).unsqueeze(1)
        sq_in2 = torch.cat((s_actions2, states), 1)
        q_actor_1 = self.critic1.forward(sq_in2)
        q_actor_2 = self.critic2.forward(sq_in2)
        q_actor = torch.min(q_actor_1, q_actor_2)
        actor_loss = self.entropy * s_log_probs - q_actor
        actor_loss = torch.mean(actor_loss)
        self.actorloss.append(self.get_numpy(actor_loss))
        actor_loss.backward()
        self.actor.optimizer.step()

        self.polyak_update()

    def polyak_update(self):
        for target, real in zip(self.valuetar.parameters(), self.value.parameters()):
            target.data.copy_(self.polyak * target.data + (1-self.polyak) * real.data)

    
    # env: gym.Env, batch_size, discount, start_training, polyak, entropy
def main():
    env = gym.make("LunarLander-v2", 
         continuous=True)
    torch.autograd.set_detect_anomaly(True)
    # print(env.action_space)
    # print(env.action_space.shape)
    # print(env.observation_space)
    # print(env.observation_space.shape[0])
    # print(env.reset()[0])
    # print(type(env.reset()[0]))
    agent = SACAgent(env, 100, 0.99, 0.995, 0.2)
    rewards = agent.train(50000)
    # print(rewards)
    print(len(rewards))
    plt.subplot(1, 2, 1)
    plt.title("Episode Rewards")
    plt.plot(rewards)
    plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.plot(agent.valloss, label="Value")
    plt.plot(agent.crit1loss, label="Critic 1")
    plt.plot(agent.crit2loss, label="Critic 2")
    plt.plot(agent.actorloss, label="Actor")
    plt.legend()
    plt.show()
    # print(len(agent.memory.buffer))
    # print(agent.get_action(env.reset()[0]))

if __name__ == "__main__":
    main()