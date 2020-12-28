# Unity Agent

### Requirements

see requirements.txt

### Agent

1st Version is an asyncronous A2C algorithm with temporal difference advantage.

#### Usage:

```
python3 main.py --env CartPole-v1 --gamma 0.99 --nr_outputs 2 --alpha 0.0005 --beta 0.0001 --training_iter 2000 --hidden_dim 64
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



### Resources

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md

https://github.com/Unity-Technologies/ml-agents/blob/master/gym-unity/README.md

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md