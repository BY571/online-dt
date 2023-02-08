import torch
from torch import nn, Tensor

class RndPredictor(nn.Module):
    def __init__(self, obs_size, hidden_size, output_size=None) -> None:
        super().__init__()
        
        self.input_layer = nn.Linear(obs_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        if output_size == None:
            self.output = nn.Linear(hidden_size, obs_size)
        else:
            self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, obs: Tensor)-> Tensor:
        x = torch.relu(self.input_layer(obs))
        x = torch.relu(self.hidden1(x))
        return self.output(x)
    
class NoiseDecay():
    def __init__(self, total_steps=3_000_000, lin_decay_steps=2_000_000, start_noise=0.175, end_noise=0.075):
        self.total_steps = total_steps 
        self.lin_decay_steps = lin_decay_steps
        self.start_noise = start_noise
        self.end_noise = end_noise
  
    def get_current_noise(self, step):
        noise = self.start_noise - step * ((self.start_noise - self.end_noise)/self.lin_decay_steps)
        if step > self.lin_decay_steps:
            noise = self.end_noise
        return noise