import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, feature_dim]
        weights = self.attention(x)  # [batch_size, seq_len, 1] # Compute attention weights
        weights = F.softmax(weights, dim=1) # Normalize weights across sequence
        output = (x * weights).sum(dim=1)  # Weighted sum across sequence
        return output
    
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims = 256, lstm_dims = 128, fc2_dims = 256,
            name = 'critic', chkpt_dir = 'tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.lstm_dims = lstm_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.lstm = nn.LSTM(fc1_dims, lstm_dims, batch_first=True)
        self.attention = Attention(lstm_dims)
        self.fc2 = nn.Linear(lstm_dims, fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = action_value.unsqueeze(1)  # Add sequence dimension
        action_value, _ = self.lstm(action_value)
        action_value = self.attention(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims = 256, fc2_dims = 256,
            name = 'value', chkpt_dir = 'tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims = 256, lstm_dims = 128,
            fc2_dims = 256, n_actions = 2, name = 'actor', chkpt_dir = 'tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.lstm_dims = lstm_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6 # probably it is the entropy coefficient, which balances between exploration and exploitation in the policy

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        #LSTM
        self.lstm = nn.LSTM(self.fc1_dims, self.lstm_dims, batch_first=True)  # Adding LSTM layer
        self.attention = Attention(self.lstm_dims)  # Adding Attention layer
        self.fc2 = nn.Linear(self.lstm_dims, self.fc2_dims) #nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        # Reshape for LSTM: add a sequence dimension
        prob = prob.unsqueeze(1)  # Reshapes prob from [batch_size, features] to [batch_size, 1, features]    
        prob, _ = self.lstm(prob)  # Process output through LSTM
        #prob = prob[:, -1, :]  # Assuming only last timestep output is used #TO EDIT
        prob = self.attention(prob)  # Process output through Attention layer instead of taking last timestep
        #######################
        prob = self.fc2(prob)
        prob = F.tanh(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min = self.reparam_noise, max = 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize = True):
        mu, sigma = self.forward(state)    # sigma
        action = mu
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise) # entropy
        log_probs = log_probs.sum(1, keepdim = True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))