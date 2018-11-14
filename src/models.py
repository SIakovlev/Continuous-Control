import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, params):

        super(Actor, self).__init__()
        self.fc1 = nn.Linear(params['l1'][0], params['l1'][1])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(params['l2'][0], params['l2'][1])
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):

        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        action = torch.tanh(self.fc3(action))

        return action


class Critic(nn.Module):
    """ Critic (Q value) Model."""

    def __init__(self, params):

        super(Critic, self).__init__()
        self.fc1 = nn.Linear(params['l1'][0], params['l1'][1])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(params['l2'][0], params['l2'][1])
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(params['l4'][0], params['l4'][1])
        nn.init.xavier_uniform_(self.fc4.weight)
        self.Q = nn.Linear(params['l5'][0], 1)
        nn.init.xavier_uniform_(self.Q.weight)

    def forward(self, state_action):

        q_value = F.relu(self.fc1(state_action))
        q_value = F.relu(self.fc2(q_value))
        q_value = F.relu(self.fc3(q_value))
        q_value = F.relu(self.fc4(q_value))
        q_value = self.Q(q_value)

        return q_value


class GaussianPolicy(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, params):

        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(params['l1'][0], params['l1'][1])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(params['l2'][0], params['l2'][1])
        nn.init.xavier_uniform_(self.fc2.weight)
        self.mean_head = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.mean_head.weight)
        self.std_head = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.std_head.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean_head(x))
        std = nn.Softplus(self.std_head(x))
        return mean, std

    def sample_action(self, states):
        mean, std = self.forward(states)
        dist = Normal(mean, std)
        action = dist.sample()
        return action.data

    def evaluate_action(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        return dist.log_prob(action)


class ValueFunction(nn.Module):
    """ Dueling Actor (Policy) Model."""

    def __init__(self, params):

        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(params['l1'][0], params['l1'][1])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(params['l2'][0], params['l2'][1])
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(params['l4'][0], params['l4'][1])
        nn.init.xavier_uniform_(self.fc4.weight)
        self.V = nn.Linear(params['l5'][0], 1)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, state):

        v_value = F.relu(self.fc1(state))
        v_value = F.relu(self.fc2(v_value))
        v_value = F.relu(self.fc3(v_value))
        v_value = F.relu(self.fc4(v_value))
        v_value = self.Q(v_value)

        return v_value
