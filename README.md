# Unity Agent


This is our Project ASP. Currently we have a PPO optimized for:

* the unity environment
* the pendulum environment

Due to lack of time, we haven't had the time to let the unity environment converge by hyperparameter optimization. However we do have a script on hyperparameter optimization and used it on the pendulum environment (because it's much faster). We decided to hand in a learning PPO for the unity environment, for which we don't know of it converges, as well as a working slightly altered PPO for pendulum which converges, just to show we could solve a continous environment. 

### Requirements

see requirements.txt

```
pip3 install -r requirements.txt
```

### Main Program

For the PPO optimized for Gym environment 'Pendulum-v0'

```
python3 main_pendulum.py
```

For the unity ml-agent,

```
python3 main.py executable
```

##### optional args:

the following arguments have default values, however you can experiment by setting them in the CLI

unity env PPO:

- batch_size - batch size for the agent update
- gamma - discout factor for episodes
- N - agent update interval
- n_epochs - number of times the agent updates on memory
- n_episodes - nr of episodes the agents trains
- alpha - learning rate of the actor
- beta - learning rate of the critic
- policy_clip - clip treshold of weight updates
- gae_lambda - discount factor of advantage estimation
- dev_episodes - development mean
- random - True if agent should be random insteaf of PPO, default is False


pendulum PPO:

- batch_size, default=32, type=int
- gamma, default=0.9, type=float
- N, default=1024, type=int
- n_epochs, default=10, type=int
- n_episodes, default=1000,  type=int
- dev_episodes, default=50,  type=int
- alpha, default=1e-4, type=float
- beta, default=3e-4, type=float
- policy_clip, default=0.2, type=float
- ac_dim, type=int, default=128
- max_grad_norm, type=float, default=0.5
- show, type=bool, default=False - render the training.


### Agents

- PPO Agent
- asyncronous A2C algorithm with temporal difference advantage.(for baseline comparison in discrete domain)

### PPO Implementation Notes:

##### Usage

For the PPO optimized for Gym environment 'Pendulum-v0'

```
python3 main_pendulum.py
```

For the unity ml-agent, for which training currently works, but does not converge

```
python3 main.py executable
```

- shuffle memories then take batch size chunks for a mini batch of stochastic gradient ascent
- Two distinct networks instead of shared inputs
- Track state, actions, reward, dones, values, log probs
- the values of those states according to critic network and the log of the probability selecting those actions
- shuffle memories and sample batches
- perform n_epochs of updates on each batch

##### Generalized Advantage Estimation(GAE)

- A way to calculate returns which reduces variance
- The smoothing is governed by lambda between 0 and 1
- lambda=1 gives highes accuracy, lower smoothes
- The PPO paper suggests lambda=0.95

##### Surrogate Policy Loss
- PPO loss function is based on the ratio of(new_prob/old_prob)
- In logarithmic space, (new_log_probs - old_log_probs).exp
- PPO policy loss is the minimum of two surrogate functions
- Surrogate1: ratio*advantage
- Surrogate2: clip(ratio,1-policy.clip,1+policy.clip)*advantage


### Baseline Run for Unity Execs

to run the baseline you'll need the stable_baselines package. This has a few prerequisites.
Follow these instructions:

https://stable-baselines.readthedocs.io/en/master/guide/install.html

after that:

```
python3 baseline.py path/to/exec --num_episodes 
```

to run baseline2.py, you'll need the baselines package from OpenAI. This has a few prerequisites.
Follow these instructions:

https://github.com/openai/baselines

after that:

```
python3 baselines2.py path/to/exec --num_episodes
```

### Hyperparameter Optimization Evoultionary Algorithm


In genetic, there is the class functionality to find the best hyperparams for an env and agent.
The best HP's will be saved at: **./best_hyperparams.txt**

**This only works with the converging environment - PPO pendulum**


```
python3 find_best_hyperparams.py PPO --pop_size 50 --cross_rate 0.3 --mut_rate 0.015 --elitism True --elite_size 3 --maximize True --gen_epochs 7
```

Running this will take a long time

#### args:

- species : PPO
- cross_rate : set with what prob individuals cross their genes
- mut_rate : set with what prob individuals mutate
- elitism : set if good indivs progress generations without tournament
- elite_size : how many of those progress w/o tournament
- maximize : maximize fitness=True or minimize=False
- gen_epochs: set the generation number

### Crawler

A 3D unity continous environment from Unity ML-Agents.

#### Parameters:

- observation space : Real valued vector with 172 parameters
- action space : Real valued vector with 20 parameters
- Agent Reward Function (independent): The reward function is geometric (normalized products of all rewards)


For every parameter, the actor networks builds a Normal Distribution on
a mu and sigma, which are the outputs of the actor network in our PPO agent.


### A2C Implementation Notes

This is an a2c with temporal difference advantage for discrete baseline comparison

##### Args:

- env : the environment in which the agents learns
- gamma : decay of future rewards
- nr_outputs : nr of dimensions in the action space of the actor
- alpha : lr of the actor
- beta : lr of the critic
- training_iter : how many training iters the agent does
- hidden_dim : hidden size of the networks

### Unity Environment

Version: Unity 2019.4.18f1

For Python API:

```
pip install mlagents==0.23.0
```

For GymWrapper

```
pip install gym-unity==0.23.0
```

Imports:

```
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper
```

Start Enviornment as follows:

```
env = UE(file_name='crawler_linux\Crawler', seed=1, side_channels=[])
env = UnityToGymWrapper(env)
env.reset()
```

Get observations and actions:

```
num_actions = env.action_size
observ_dim = env.observation_space.shape[0]
```

Troubleshooting:

For OsX/Linux: If your script exits with an error message, that some permissions ar not set:

```
chmod -R 755 /abs/path/to/UnityEnvironment
```

where abs path to should be replaced with a path to your executable

### Resources


##### Configure a virtual Environment
- set
```
$ python3 -m venv environment_name
```
- activate (Unix or Mac)
```
$ source environment_name/bin/activate
```
- activate (Windows)
```
$ environment_name\Scripts\activate.bat
```
- deactivate 
```
$ Source deactivate 
```
or (for anaconda)
```
$ conda deactivate
```
or 
```
§ deactive
```
See more in the [documentation](https://docs.python.org/3/tutorial/venv.html )

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md

https://github.com/Unity-Technologies/ml-agents/blob/master/gym-unity/README.md

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md