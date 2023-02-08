

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