# Continuous-Control

Deep Reinforcement Learning Nanodegree Project 2

(add gif file)

### Project description

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of  Each action is a vector with four numbers, 

- **State space** is `33` dimensional continuous vector, consisting of position, rotation, velocity, and angular velocities of the arm.

- **Action space** is `4` dimentional continuous vector, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

- **Solution criteria**: the environment is considered as solved when the agent gets an average score of **+30 over 100 consecutive episodes** (averaged over all agents in case of multiagent environment).

### Getting started

#### Configuration

PC configuration used for this project:
- OS: Mac OS 10.14 Mojave
- i7-8800H, 32GB, Radeon Pro 560X 4GB

#### Structure

All project files are stored in `/src` folder:
- `main.py` - main file where the program execution starts.
- `agent.py` - agent class implementation.
- `unity_env.py` - Unity Environment wrapper (borrowed from [here](https://github.com/Unity-Technologies/ml-agents/blob/master/gym-unity/gym_unity/envs/unity_env.py) and modified).
- `trainer.py` - trainer (interface between agent and environment) implementation.
- `replay_buffer.py` - memory replay buffer implementation.
- `models.py` - neural network implementations (PyTorch)

All project settings are stored in JSON file: `settings.json`. It is divided into 4 sections: 
- `general_params` - general, module agnostic parameters: mode (`train` or `test`), number of episodes, seed.
- `agent_params` - agent parameters: epsilon, gamma, learning rate, etc. This section also includes neural network configuration settings and memory replay buffer parameters.
- `trainer_params` - trainer parameters depending on the algorithm. They are responsible for any change of agent learning parameters. Agent can't change them.
- `env_params` - environment parameters: path, number of agents, etc.

#### Environment setup

- For detailed Python environment setup (PyTorch, the ML-Agents toolkit, and a few more Python packages) please follow these steps: [link](https://github.com/udacity/deep-reinforcement-learning#dependencies)

- Download pre-built Unity Environment:
  - [Linux - 20 agents](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip), [Linux - 1 agent](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
  - [Mac - 20 agents](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip), [Mac - 1 agent](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
  - [Win x32 - 20 agents](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip), [Win x32 - 1 agent](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
  - [Win x64 - 20 agents](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip), [Win x64 - 1 agent](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

 - Open `settings.json` and specify the relative path to the application file in `"path"` inside of `"env_params"`.


