import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models import Actor, Critic, GaussianPolicy, ValueFunction
from replay_buffer import ReplayBuffer, Buffer
from uo_process import UOProcess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentDDPG:
    def __init__(self, params):

        action_size = params['action_size']
        state_size = params['state_size']
        buf_params = params['buf_params']

        nn_params = params['nn_params']
        nn_params['nn_actor']['l1'][0] = state_size
        nn_params['nn_actor']['l3'][1] = action_size
        nn_params['nn_critic']['l1'][0] = state_size + action_size

        self.__actor_local = Actor(nn_params['nn_actor']).to(device)
        self.__actor_target = Actor(nn_params['nn_actor']).to(device)
        self.__critic_local = Critic(nn_params['nn_critic']).to(device)
        self.__critic_target = Critic(nn_params['nn_critic']).to(device)

        self.__action_size = action_size
        self.__state_size = state_size
        self.__memory = ReplayBuffer(buf_params)
        self.__t = 0

        self.gamma = params['gamma']
        self.learning_rate_actor = params['learning_rate_actor']
        self.learning_rate_critic = params['learning_rate_critic']
        self.tau = params['tau']

        self.__optimiser_actor = optim.Adam(self.__actor_local.parameters(), self.learning_rate_actor)
        self.__optimiser_critic = optim.Adam(self.__critic_local.parameters(), self.learning_rate_critic)
        self.__uo_process = UOProcess()
        # other parameters
        self.agent_loss = 0.0

    # Set methods
    def set_learning_rate(self, lr_actor, lr_critic):
        self.learning_rate_actor = lr_actor
        self.learning_rate_critic = lr_critic
        for param_group in self.__optimiser_actor.param_groups:
            param_group['lr'] = lr_actor
        for param_group in self.__optimiser_critic.param_groups:
            param_group['lr'] = lr_critic

    # Get methods
    def get_actor(self):
        return self.__actor_local

    def get_critic(self):
        return self.__critic_local

    # Other methods
    def step(self, state, action, reward, next_state, done):
        # add experience to memory
        self.__memory.add(state, action, reward, next_state, done)

        if self.__memory.is_ready():
            experiences = self.__memory.sample()
            self.__update(experiences)

    def choose_action(self, state, mode='train'):
        if mode == 'train':
            # state should be transformed to a tensor
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            self.__actor_local.eval()
            with torch.no_grad():
                action = self.__actor_local(state) + self.__uo_process.sample()
            self.__actor_local.train()
            return list(np.clip(action.cpu().numpy().squeeze(), -1, 1))
        elif mode == 'test':
            # state should be transformed to a tensor
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            self.__actor_local.eval()
            with torch.no_grad():
                action = self.__actor_local(state)
            self.__actor_local.train()
            return list(np.clip(action.cpu().numpy().squeeze(), -1, 1))
        else:
            print("Invalid mode value")

    def reset(self, sigma):
        self.__uo_process.reset(sigma)

    def __update(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        # update critic
        # ----------------------------------------------------------
        loss_fn = nn.MSELoss()
        self.__optimiser_critic.zero_grad()
        # form target
        next_actions = self.__actor_target(next_states)
        Q_target_next = self.__critic_target.forward(torch.cat((next_states, next_actions), dim=1)).detach()
        targets = rewards + self.gamma * Q_target_next * (1 - dones)
        # form output
        outputs = self.__critic_local.forward(torch.cat((states, actions), dim=1))
        mean_loss_critic = loss_fn(outputs, targets)  # minus added since it's gradient ascent
        mean_loss_critic.backward()
        self.__optimiser_critic.step()

        # update actor
        # ----------------------------------------------------------
        self.__optimiser_actor.zero_grad()
        predicted_actions = self.__actor_local(states)
        mean_loss_actor = - self.__critic_local.forward(torch.cat((states, predicted_actions), dim=1)).mean()
        mean_loss_actor.backward()
        self.__optimiser_actor.step()   # update actor

        self.__soft_update(self.__critic_local, self.__critic_target, self.tau)
        self.__soft_update(self.__actor_local, self.__actor_target, self.tau)

    @staticmethod
    def __soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class AgentPPO:
    def __init__(self, params):

        action_size = params['action_size']
        state_size = params['state_size']
        buf_params = params['buf_params']

        nn_params = params['nn_params']
        nn_params['nn_policy']['l1'][0] = state_size
        nn_params['nn_policy']['l3'][1] = action_size
        nn_params['nn_value_function']['l1'][0] = state_size

        self.__policy = GaussianPolicy(nn_params['nn_policy']).to(device)
        self.__value_fn = ValueFunction(nn_params['nn_value_function']).to(device)

        self.__action_size = action_size
        self.__state_size = state_size
        self.__memory = Buffer(buf_params)
        self.__t = 0

        self.gamma = params['gamma']
        self.learning_rate_policy = params['learning_rate_policy']
        self.learning_rate_value_fn = params['learning_rate_value_fn']
        self.tau = params['tau']

        self.updates_num = params['updates_num']
        self.ppo_epochs = params['ppo_epochs']
        self.baseline_epochs = params['baseline_epochs']
        self.ppo_eps = params['ppo_epsilon']

        self.__optimiser_policy = optim.Adam(self.__policy.parameters(), self.learning_rate_policy)
        self.__optimiser_value_fn = optim.Adam(self.__value_fn.parameters(), self.learning_rate_value_fn, weight_decay=1e-3)
        # other parameters
        self.agent_loss = 0.0

    # Set methods
    def set_learning_rate(self, lr_policy, lr_value_fn):
        self.learning_rate_policy = lr_policy
        self.learning_rate_value_fn = lr_value_fn
        for param_group in self.__optimiser_policy.param_groups:
            param_group['lr'] = lr_policy
        for param_group in self.__optimiser_value_fn.param_groups:
            param_group['lr'] = lr_value_fn

    # Get methods
    def get_actor(self):
        return self.__policy

    def get_critic(self):
        return self.__value_fn

    # Other methods
    def collect_trajectory(self, state, action, reward, next_state, done):
        # add experience to memory
        self.__memory.add(state, action, reward, next_state, done)

    def step(self):
        self.__update()

    def choose_action(self, state, mode='train'):
        if mode == 'train':
            # state should be transformed to a tensor
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            self.__policy.eval()
            with torch.no_grad():
                action, mean, std = self.__policy.sample_action(state)
            self.__policy.train()
            return list(np.clip(action.cpu().numpy().squeeze(), -1, 1)), mean.cpu().numpy(), std.cpu().numpy()
        elif mode == 'test':
            pass
        else:
            print("Invalid mode value")

    def reset(self, sigma):
        pass

    def __update(self, batch_size=256):

        states, actions, rewards, next_states, dones = list(self.__memory.get_data())

        # T = rewards.shape[0]
        # last_return = torch.zeros(rewards.shape[1]).float().to(device)
        # returns = torch.zeros(rewards.shape).float().to(device)
        #
        # for t in reversed(range(T)):
        #     last_return = rewards[t] + last_return * self.gamma * (1 - dones[t])
        #     returns[t] = last_return

        log_probs_old = self.__policy.evaluate_actions(states, actions).detach()

        discount = self.gamma ** np.arange(rewards.shape[0])
        rewards_cpu = rewards.cpu().numpy() * discount[:, np.newaxis]
        returns = rewards_cpu[::-1].cumsum(axis=0)[::-1]
        mean = np.mean(returns, axis=1)
        std = np.std(returns, axis=1) + 1.0e-10
        returns_normalized = torch.from_numpy((returns - mean[:, np.newaxis]) / std[:, np.newaxis]).float().\
            to(device).detach()

        trajectories = [states, actions, returns_normalized, next_states, rewards, log_probs_old]

        trajectory_batches = [list(torch.split(trajectory_batch, batch_size)) for trajectory_batch in trajectories]

        num_of_batches = len(trajectory_batches[0])

        # Critic update
        for i in range(num_of_batches):
            # for i in range(self.updates_num):

            # idx = np.random.randint(0, returns.shape[0], batch_size)
            states_batch = trajectory_batches[0][i]
            returns_batch = trajectory_batches[2][i]
            next_states_batch = trajectory_batches[3][i]
            rewards_batch = rewards[4][i]

            self.__optimiser_value_fn.zero_grad()
            value_pred_next = self.__value_fn(next_states_batch).squeeze().view(-1, 1)
            value_pred = self.__value_fn(states_batch).squeeze().view(-1, 1)
            loss_fn = nn.MSELoss()
            value_loss = loss_fn(value_pred, value_pred_next + rewards_batch.view(-1, 1))
            value_loss.backward()
            self.__optimiser_value_fn.step()


        # Policy update
        for i in range(num_of_batches):
            # for i in range(self.updates_num):

            # idx = np.random.randint(0, returns.shape[0], batch_size)
            states_batch = trajectory_batches[0][i]
            actions_batch = trajectory_batches[1][i]
            returns_batch = trajectory_batches[2][i]
            log_probs_old_batch = trajectory_batches[5][i]

            # states_batch = states[idx]
            # actions_batch = actions[idx]
            # returns_batch = returns[idx]
            # log_probs_old_batch = log_probs_old[idx]

            advantages = (returns_batch - self.__value_fn(states_batch).squeeze().detach()).view(-1, 1)

            self.__optimiser_policy.zero_grad()
            log_probs_batch = self.__policy.evaluate_actions(states_batch, actions_batch)
            ratio = torch.exp(log_probs_batch.view(-1, 1) - log_probs_old_batch.view(-1, 1))
            surrogate_fn = torch.min(ratio * advantages,
                                     torch.clamp(ratio * advantages, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps))

            policy_loss = - surrogate_fn.mean()
            policy_loss.backward()
            self.__optimiser_policy.step()
