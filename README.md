# Unity Agent
 ### Hyper parameters
  Memory length T(should be much less than the length of episode), batch size, number of epochs, learning rate

  - gamma: discount factor in the calculation of our advantages(typically use 0.99)
  - alpha: learning rate (0.0003) for actor
  - beta: learning rate for critic
  - policy_clip: 0.1/0.2
  - batch_size=64
  - N: the number of steps before we perform an update(2048)
  - n_epochs: the number of epochs (10)
  - gae_lambda: this is the smoothing factor used in the GAE algorithm.
## PPO Upgrades
1.Generalized Advantage Estimation(GAE)
2.Surrogate Policy loss
3.Mini-Batch Updates
### GAE
A way to calculate returns which reduces variance
The smoothing is governed by lambda between 0 and 1
lambda gives highest accuracy, lower smoothes
The PPO Paper suggests lambda=0.95
### PPO Implementation Notes:
(folowing parameters works well for CartPole enviroment )
Implementation Notes
- Memory Indices[0,1,...,19]
- Batches start at multiples of batch_size[0,5,10,15]
- shuffle memories then take batch size chunks for a mini batch of stochastic gradient ascent
- Two distinct networks instead of shared inputs
- Memory is fixed to length T(20)steps
- Track state, actions, reward, dones, values, log probs
 - the values of those states according to critic network and the log of the probability selecting those actions
 - shuffle memories and sample batches(5)
 - perform 4 epochs of updates on each batch
 
### parameter value im main.py benutzt
  (self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10)
  this parameters com from the values for continuous enviroments)

   
 
### Resources

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md

https://github.com/Unity-Technologies/ml-agents/blob/master/gym-unity/README.md

https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md