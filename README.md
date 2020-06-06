
# 3D_AirBattle
## Description
This repository is a three-dimensional simulating competitive environment. 

Two groups of agents with fire range are contained in our environment, and the members in group have their own distinctive color. Each agent is free to move in three-dimensional environment with boundary and  switch its orientation. The target of agent is that trying to aviod entering into the fire range of opponents and adjust its own fire range toward opponents to annihilate them. 

The current edition only supports the condition including one agent trained by intelligent learning algorithm and another one with fixed strategy.

## History Record 
### 20/06/06
Test our environment by
```
python3 testEnv.py
```
The project of class of AirBattle is completed and no apparent logistic problem is shown when our agent is controlled by random strategy. Meanwhile, the demo of env rendering can be seen by 
```
python3 renderExp.py
```

However, the live visualization of environment and the access to the RL control algorithm has not been finished. We will supplement these functions in the following developing edition. 
