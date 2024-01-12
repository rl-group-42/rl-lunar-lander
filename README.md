# DQN
Implementation of the DQN algorithm with Duelling Networks, Double Q-Learning, and Prioritised Experience Replay on the LunarLander environment

To train the network, call train(params, modifications=[], save_data=False) with

params: dictionary of parameters used by model. default (as provided):
```py
params = {
'n_hidden_neurons': 128,
'learning_rate': 5e-4,
'discount_factor': 0.99,
'max_buffer_size': 10_000,
'batch_size': 32,
'n_episodes': 10_000,
'exploration_chance_start': 1.0,
'exploration_chance_end': 1e-4,
'exploration_chance_decay': 0.99,
'target_update_freq': 20,
'finish_check_freq': 100,
'finish_score': 200,
'per_params': {
    'bias_factor_start': 0.5,
    'bias_factor_end': 1.0,
    'bias_increment': 1e-2,
    'priority_scale': 0.7,
    'td_error_clamp': 50 # 1/4 of target end score
    }
}
```

modifications (default = []): list of modifications to the base DQN algorithm
```py
modifications = ['double', 'duelling', 'per'] # set modifications to all 3 improvements on top of base DQN
```

save_data (default = False): save the hyperparameters used in a parseable txt file as well as score for each episode in format:
```txt
{dict of params}
---

score1@episode1:step_count1
score2@episode2:step_count2
...
scoreN@episodeN:step_countN
```

train() returns trained agent as DQNAgent object

visual_run_agent(agent) can be called to run the agent visually by passing a DQNAgent object
