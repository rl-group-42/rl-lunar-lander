# Deep Deterministic Policy Gradient (DDPG)
Implementation of the DDPG algorithm for the Lunar Lander environment (ddpg.py)

To train the network, create an Agent object with the below parameters (provided values are the default):
```python
continuity = True
render = "rgb_array"

train_iterations = 65
episodes = 4000
epoches = 100

optimal_run_episodes = 5
render_episode = -1
measure_after_iterations = 1
measurement = True

gamma = 0.99
tau = 0.001
learning_rate = [0.000025, 0.00025, 0.000025, 0.00025]

batch_size = 64
memory_size = 100000
update_epoches= int(episodes / batch_size)

theta=0.2
dt=1e-2
sigma=0.2

agent = Agent(env,
        continuity=continuity,
        train_iterations=train_iterations,
        episodes=episodes,
        epoches=epoches,
        optimal_run_episodes=optimal_run_episodes,
        render_episode=render_episode,
        measure_after_iterations=measure_after_iterations,
        measurement=measurement,
        gamma=gamma,
        tau=tau,
        learning_rate=learning_rate,
        batch_size=batch_size,
        memory_size=memory_size,
        update_epoches=update_epoches,
        theta=theta,
        dt=dt,
        sigma=sigma)
```
and call
```python
agent.train_run()
```
The environment needs to be considered as follows:
```python
env = gym.make("LunarLander-v2", continuous=continuity)
```

# Twin Delayed Deterministic Policy Gradient (TD3)
Implementation of the DDPG algorithm for the Lunar Lander environment (td3.py)

To train the network, create an Agent object with the below parameters (provided values are the default):
```python
continuity = True
render = "rgb_array"

train_iterations = 65
episodes = 5000
epoches = 100

optimal_run_episodes = 5
render_episode = -1
measure_after_iterations = 1
measurement = True

gamma = 0.99
tau = 0.0007
learning_rate = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

batch_size = 64
memory_size = 1000000
update_epoches= int(episodes / batch_size)

theta=0.2
dt=1e-2
sigma=0.15

min_noise_clip=-0.4
max_noise_clip=0.4
noise_scale=0.3

policy_delay=2

agent = Agent(env,
        continuity=continuity,
        train_iterations=train_iterations,
        episodes=episodes,
        epoches=epoches,
        optimal_run_episodes=optimal_run_episodes,
        render_episode=render_episode,
        measure_after_iterations=measure_after_iterations,
        measurement=measurement,
        gamma=gamma,
        tau=tau,
        learning_rate=learning_rate,
        batch_size=batch_size,
        memory_size=memory_size,
        update_epoches=update_epoches,
        theta=theta,
        dt=dt,
        sigma=sigma,
        min_noise_clip=min_noise_clip,
        max_noise_clip=max_noise_clip,
        noise_scale=noise_scale,
        policy_delay=policy_delay)
```
and call
```python
agent.train_run()
```
The environment needs to be considered as follows:
```python
env = gym.make("LunarLander-v2", continuous=continuity)
```
