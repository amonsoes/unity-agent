# Unity Agent

### Requirements

see requirements.txt

### Agents

- asyncronous A2C algorithm with temporal difference advantage.
- PPO Agent

### A2C Implementation Notes

##### Usage:

```
python3 main.py --env CartPole-v1 --gamma 0.99 --nr_outputs 2 --alpha 0.0005 --beta 0.001 --training_iter 1500 --hidden_dim 64
```
##### Args:

- env : the environment in which the agents learns
- gamma : decay of future rewards
- nr_outputs : nr of dimensions in the action space of the actor
- alpha : lr of the actor
- beta : lr of the critic
- training_iter : how many training iters the agent does
- hidden_dim : hidden size of the networks

##### Peculiarities:

In order for the agent to act in the environment Cartpole, the distribution over the actions is categorical for now.
Change to Normal for more agnoistc agent.

Best observed Hyperparams are stored in default params. Those need to change as soon as we deploy the algorithm to the
new env.


### PPO Implementation Notes:
(folowing parameters works well for CartPole enviroment )
Implementation Notes

##### Usage

```
python3 main.py
```
- Memory Indices[0,1,...,19]
- Batches start at multiples of batch_size[0,5,10,15]
- shuffle memories then take batch size chunks for a mini batch of stochastic gradient ascent
- Two distinct networks instead of shared inputs
- Memory is fixed to length T(20)steps
- Track state, actions, reward, dones, values, log probs
 - the values of those states according to critic network and the log of the probability selecting those actions
 - shuffle memories and sample batches(5)
 - perform 4 epochs of updates on each batch

  ##### Hyperparameters

  Memory length T(should be much less than the length of episode), batch size, number of epochs, learning rate

  - gamma: discount factor in the calculation of our advantages(typically use 0.99)
  - alpha: learning rate (0.0003) for actor
  - beta: learning rate for critic
  - policy_clip: 0.1/0.2
  - batch_size=64
  - N: the number of steps before we perform an update(2048)
  - n_epochs: the number of epochs (10)
  - gae_lambda: the lambda parameter

##### parameter value im main.py benutzt

  (self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10)
  this parameters com from the values for continuous enviroments)


### Hyperparameter Optimization

In genetic, there is the class functionality to find the best hyperparams for an env and agent.
The best HP's will be saved at: **./best_hyperparams.txt**

```
python3 find_best_hyperparams.py species --pop_size 50 --cross_rate 0.3 --mut_rate 0.015 --elitism True --elite_size 3 --maximize True --gen_epochs 7
```

#### args:

- species : A2C or PPO
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



### Unity Environment

Version: Unity 2019.4.18f1

For Python API:

```
pip install mlagents==0.23.0
```

For GymWrapper

```
pip install gym-unity==0.22.0
```

Imports:

```
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
```

Start Enviornment as follows:

```
env = UE(file_name='crawler_linux\Crawler', seed=1, side_channels=[])
env.reset()
```



### Resources

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md

https://github.com/Unity-Technologies/ml-agents/blob/master/gym-unity/README.md

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md