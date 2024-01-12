# PPO and RND-PPO

## PPO 
Implementation of the PPO algorithm for the LunarLander environment (ppo_code_2.py)

To train the network, create an Agent object with the below parameters (provided values are the default):
```py

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
```
and call via agent.train_run()

The environment needs to be configured for this and provided to the agent as follows:
```py
env = gym.make("LunarLander-v2", continuous=continuity)
```
NB. It is possible to train the network on continuous or discrete action space by toggling the continuity boolean. 

## RND-PPO
Implementation of the RND-PPO algorithm for the LunarLander environment (rnd_code_4.py)

To train the network, create an Agent object with the below parameters (provided values are the default):
```py
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
```
and call via agent.train_run()

The environment needs to be configured for this and provided to the agent as follows:
```py
env = gym.make("LunarLander-v2", continuous=continuity)
```
NB. It is possible to train the network on continuous or discrete action space by toggling the continuity boolean. 
