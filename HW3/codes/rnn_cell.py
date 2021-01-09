import torch
from torch import nn
import torch.nn.functional as F

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def init(self, batch_size, device):
        #return the initial state
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, incoming, state):
        # flag indicates whether the position is valid. 1 for valid, 0 for invalid.
        output = (self.input_layer(incoming) + self.hidden_layer(state)).tanh()
        new_state = output # stored for next step
        return output, new_state

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.input_layer = nn.Linear(input_size, hidden_size*3)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size*3, bias=False)
        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state
        return torch.zeros(batch_size, self.hidden_size, device=device)
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state
        r1, r2, r3 = self.input_layer(incoming).chunk(3, dim=-1)
        z1, z2, z3 = self.hidden_layer(state).chunk(3, dim=-1)
        r = torch.sigmoid(r1 + z1)
        z = torch.sigmoid(r2 + z2)
        n = torch.tanh(r3 + r * z3)
        output = (1 - z) * n + z * state
        new_state = output
        return output, new_state
        # TODO END

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.input_layer = nn.Linear(input_size, hidden_size*4)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size*4, bias=False)
        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state (which can be a tuple)
        return (
            torch.zeros(batch_size, self.hidden_size, device=device), 
            torch.zeros(batch_size, self.hidden_size, device=device)
        )
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state
        h, c = state
        i1, i2, i3, i4 = self.input_layer(incoming).chunk(4, dim=-1)
        h1, h2, h3, h4 = self.hidden_layer(h).chunk(4, dim=-1)
        i = torch.sigmoid(i1 + h1)
        f = torch.sigmoid(i2 + h2)
        g = torch.tanh(i3 + h3)
        o = torch.sigmoid(i4 + h4)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        output = new_h
        return output, (new_h, new_c)
        # TODO END
