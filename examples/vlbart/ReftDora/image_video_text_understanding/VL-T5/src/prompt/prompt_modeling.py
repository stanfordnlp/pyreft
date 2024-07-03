import torch
import torch.nn as nn

class InputPrompts(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.prompt_len = config.prompt_len
        self.input_dim = config.input_dim
        self.mid_dim = config.mid_dim

        self.prefix_tokens = torch.arange(self.prompt_len).long()
        self.prefix_embedding = nn.Sequential(
            nn.Embedding(self.prompt_len, self.input_dim),
            nn.Linear(self.input_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.input_dim),
        )

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device) # (B, L)
        prefix_prompt = self.prefix_embedding(input_tokens) # (B, L, d_model * n_heads * n_layer)
        
        return prefix_prompt