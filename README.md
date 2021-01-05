# Unity Agent

### Requirements

see requirements.txt

### Agents

1st Version is an asyncronous A2C algorithm with temporal difference advantage.

#### Usage:

```
python3 main.py --env CartPole-v1 --gamma 0.99 --nr_outputs 2 --alpha 0.0005 --beta 0.001 --training_iter 1500 --hidden_dim 64
```
#### Args:

- env : the environment in which the agents learns
- gamma : decay of future rewards
- nr_outputs : nr of dimensions in the action space of the actor
- alpha : lr of the actor
- beta : lr of the critic
- training_iter : how many training iters the agent does
- hidden_dim : hidden size of the networks

#### Peculiarities:

In order for the agent to act in the environment Cartpole, the distribution over the actions is categorical for now.
Change to Normal for more agnoistc agent.

Best observed Hyperparams are stored in default params. Those need to change as soon as we deploy the algorithm to the
new env.

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



### Resources

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md

https://github.com/Unity-Technologies/ml-agents/blob/master/gym-unity/README.md

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md