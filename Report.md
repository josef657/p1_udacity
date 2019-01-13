# Report


## Learning algorithm


The simple  Deep Q Network (DQN) algorithm will be implemented  as described in the course.  The state is the  input for deep network, that has 3 fully connected layers :
 - input: state size  output 128
  - input: 128  output 64
   - input: 64  output action size 
   


### DQN Hyper Parameters  


### DQN Agent Hyper Parameters
![Reward Plot](https://github.com/josef657/p1_udacity/blob/master/Scoring.png?raw=true)

- BUFFER_SIZE (int): replay buffer size : 1e6
- BATCH_SIZE (int): mini batch size :128
- GAMMA (float): discount factor : 0.99
- TAU (float): for soft update of target parameters : 1e-3
- LR (float): learning rate for optimizer : 0.0001
- UPDATE_EVERY (int): how often to update the network : 3


##  Scoring plot




```
Episode 100	Average Score: 0.30
Episode 200	Average Score: 3.54
Episode 300	Average Score: 8.97
Episode 400	Average Score: 11.50
Episode 500	Average Score: 11.72
Episode 600	Average Score: 12.55
Episode 665	 Score: 13.0900
 Solved after  565 episodes!	Average Score: 13.0900

```

## Improvements

-Double Deep Q Networks
-Dueling Deep Q Networks
-RAINBOW Paper


