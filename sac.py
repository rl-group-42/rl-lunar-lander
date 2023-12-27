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

def get_device() -> str:
    """ Get Device Name """
    if cuda.is_available():
        return "cuda"
    elif mps.is_available():
        return "mps"
    else:
        return "cpu"
    


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
        for i in self.rand_key():
            sample.append(self.buffer[i])
        return sample

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
    def __init__(self, input_dim, hidden_dims, output_dim, output_activation, learning_rate):
        super(ActorNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.output_activation = output_activation

        self.learning_rate = learning_rate

        # Set up layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(output_activation())
        self.network = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

    def forward(self, input):
        output = self.network(input)
        return output
    
class SACAgent:
    def __init__(self, env: gym.Env, batch_size, target_update, episodes, discount, start_training, polyak):

        self.env = env
        self.state_space = env.observation_space
        self.action_space = env.action_space

        # size of minibatch when updating neuralnets
        self.batch_size = batch_size
        # target network update rate
        self.target_update = target_update
        # number of episodes in a training run
        self.episodes = episodes
        # when to start training
        self.start_training = start_training
        # polyak proportion for target updates
        self.polyak = polyak

        # future reward discount
        self.discount = discount

        self.memory = ReplayBuffer()

        self.device = torch.device(get_device())
        self.input_dim = self.state_space.shape[0]
        self.output_dim = self.action_space.shape[0]
        self.actor_net = NeuralNet(self.input_dim, [128], self.output_dim, nn.Sigmoid, 0.01).to(self.device)
        self.critic_net1 = NeuralNet(self.input_dim + self.output_dim, [128], self.output_dim, nn.Sigmoid, 0.01).to(self.device)
        self.critic_net2 = NeuralNet(self.input_dim + self.output_dim, [128], self.output_dim, nn.Sigmoid, 0.01).to(self.device)
        # self.actor_net.train()
        # self.critic_net.train()

        print(self.actor_net, self.critic_net)

    def get_tensor (self, array):
        return torch.from_numpy(array).float().to(self.device)
    
    def get_numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def get_action(self, state):
        
        state_tensor = self.get_tensor(state)
        action = self.actor_net.forward(state_tensor)
        return self.get_numpy(action)
    
    def train(self):

        total_reward = 0
        terminal = True
        for t in range(100):
            if terminal:
                current_state = self.env.reset()[0]
            
            action = self.get_action(current_state)
            next_state, reward, isterminal, truncated, _ = self.env.step(action)
            total_reward += reward
            terminal = isterminal or truncated
            self.memory.new_memory(current_state, action, reward, next_state, terminal)
            current_state = next_state
            if t < self.start_training:
                # self.train_networks()


    # memory: 0 state, 1 action, 2 reward, 3 next, 4 done
    def train_networks(self):

        minibatch = self.memory.rand_sample(self.batch_size)
        targets = []
        for sample in minibatch:
            sample_action = 
            next1 = self.critic_net1
            next_estimate = min()
            target = sample[0] + self.discount*(1-sample[4])







    
def main():
    env = gym.make("LunarLander-v2", 
         continuous=True)
    print(env.action_space)
    print(env.action_space.shape)
    print(env.observation_space)
    print(env.observation_space.shape[0])
    agent = SACAgent(env)
    print(agent.get_action(env.reset()[0]))

if __name__ == "__main__":
    main()