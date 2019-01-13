import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_size, action_size, seed,
                 f1_units=128, f2_units=64):
        """
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.f1 = nn.Linear(state_size, f1_units)
        self.f2 = nn.Linear(f1_units, f2_units)
        self.f3 = nn.Linear(f2_units, action_size)


    def forward(self, state):
        x = F.relu(self.f1(state))
        x = F.relu(self.f2(x))
        return self.f3(x)
