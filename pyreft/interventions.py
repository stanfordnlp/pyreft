import torch
from collections import OrderedDict

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from transformers.activations import ACT2FN


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True # we must match!
        
        return


class LoreftIntervention_Scale(LoreftIntervention):
    """
    LoReFT with learned scalar gating (starts at 0 for identity).
    
    LoReFT(h) = h + scale * R^T(Wh + b − Rh)
    
    At init: scale = 0, so output = h (identity).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        delta = torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), 
            self.rotate_layer.weight.T
        )
        output = base + self.scale * delta
        return self.dropout(output.to(base.dtype))


class LoreftIntervention_SigmoidScale(LoreftIntervention):
    """
    LoReFT with sigmoid-bounded scalar gating [0, 2].
    
    LoReFT(h) = h + scale * R^T(Wh + b − Rh)
    where scale = 2 * sigmoid(scale_logit - 5)
    
    At init: scale ≈ 0.013 (near identity).
    Range [0, 2] allows identity (0), full (1), and reflection (2).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale_logit = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        delta = torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), 
            self.rotate_layer.weight.T
        )
        scale = 2.0 * torch.sigmoid(self.scale_logit - 5.0)
        output = base + scale * delta
        return self.dropout(output.to(base.dtype))


class LoreftIntervention_DataDepScale(LoreftIntervention):
    """
    LoReFT with data-dependent gating (per-sequence).
    
    LoReFT(h) = h + scale(h) * R^T(Wh + b − Rh)
    where scale(h) = 2 * sigmoid(gate_proj(pool(h)))
    
    Similar to Deep Delta Learning's gating mechanism.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dtype = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        self.gate_proj = torch.nn.Linear(self.embed_dim, 1, bias=True).to(dtype)
        # Initialize to output near 0 at start
        torch.nn.init.zeros_(self.gate_proj.weight)
        torch.nn.init.constant_(self.gate_proj.bias, -5.0)
    
    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        delta = torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), 
            self.rotate_layer.weight.T
        )
        # Pool over sequence dimension, compute gate
        pooled = base.mean(dim=1, keepdim=True)  # (B, 1, D)
        scale = 2.0 * torch.sigmoid(self.gate_proj(pooled))  # (B, 1, 1)
        output = base + scale * delta
        return self.dropout(output.to(base.dtype))


class LoreftIntervention_TokenScale(LoreftIntervention):
    """
    LoReFT with token-wise gating (most expressive).
    
    LoReFT(h) = h + scale(h) * R^T(Wh + b − Rh)
    where scale(h) = 2 * sigmoid(gate_proj(h)) per token
    
    Each token gets its own gating scalar.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dtype = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        self.gate_proj = torch.nn.Linear(self.embed_dim, 1, bias=True).to(dtype)
        # Initialize to output near 0 at start
        torch.nn.init.zeros_(self.gate_proj.weight)
        torch.nn.init.constant_(self.gate_proj.bias, -5.0)
    
    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        delta = torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), 
            self.rotate_layer.weight.T
        )
        # Per-token gate
        scale = 2.0 * torch.sigmoid(self.gate_proj(base))  # (B, T, 1)
        output = base + scale * delta
        return self.dropout(output.to(base.dtype))


class NoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class ConsreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)


class LobireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LobiReFT(h) = h + R^T(b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.learned_source, self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class DireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    DiReFT(h) = h + R^T(Wh + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.learned_source.weight.dtype)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(self.rotate_layer.weight.dtype), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class NodireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))

