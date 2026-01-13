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
        # Debug logging (off by default)
        self.debug = kwargs.get("debug", False)
        self._debug_logged = False
        self.metrics = {}
        # Save full parametrization state for training continuation (off by default for smaller files)
        self.save_for_training = kwargs.get("save_for_training", False)

    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        learned = self.act_fn(self.learned_source(base))
        diff = learned - rotated_base
        delta = torch.matmul(diff, self.rotate_layer.weight.T)

        # Store metrics for logging (only if debug=True)
        if self.debug:
            # Note: ||delta|| = ||diff|| since R has orthonormal columns
            diff_norm = diff.norm().item()
            base_norm = base.norm().item()
            b_norm = self.learned_source.bias.norm().item()
            self.metrics = {
                "base_norm": base_norm,
                "rotated_base_norm": rotated_base.norm().item(),
                "learned_norm": learned.norm().item(),
                "b_norm": b_norm,
                "diff_norm": diff_norm,
                "delta_base_ratio": diff_norm / (base_norm + 1e-8),
            }
            if not self._debug_logged:
                print(f"[DEBUG LoreftIntervention] First forward:")
                print(f"  base norm: {base_norm:.4f}, Rh norm: {self.metrics['rotated_base_norm']:.4f}")
                print(f"  Wh+b norm: {self.metrics['learned_norm']:.4f}, b norm: {b_norm:.4f}")
                print(f"  diff norm: {diff_norm:.4f}, delta/base ratio: {self.metrics['delta_base_ratio']:.4f}")
                self._debug_logged = True

        output = base + delta
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Save state for checkpoint.

        By default, only saves the computed orthogonal weight (for inference).
        If save_for_training=True, also saves internal parametrization state
        needed for training continuation without breaking orthogonality.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        # Save computed orthogonal weight (always, for inference)
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        # Optionally save internal parametrization state (for training continuation)
        if self.save_for_training:
            state_dict["rotate_layer_original"] = self.rotate_layer.parametrizations.weight.original.data
            state_dict["rotate_layer_base"] = self.rotate_layer.parametrizations.weight[0].base.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Load state from checkpoint. Properly restores parametrization internals
        to support training continuation without breaking orthogonality.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]

        # Recreate the parametrized layer
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)

        # Check if we have the full parametrization state (new format)
        if "rotate_layer_original" in state_dict and "rotate_layer_base" in state_dict:
            # Restore internal parametrization state for training continuation
            original = state_dict["rotate_layer_original"].to(self.learned_source.weight.device)
            base = state_dict["rotate_layer_base"].to(self.learned_source.weight.device)
            self.rotate_layer.parametrizations.weight.original.data.copy_(original)
            self.rotate_layer.parametrizations.weight[0].base.data.copy_(base)
        else:
            # Legacy format: only has computed weight, use old behavior
            # (works for inference but may break training continuation)
            self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w

        # Verify the loaded weight matches expected
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data, atol=1e-5), \
            f"Loaded weight mismatch: {torch.norm(self.rotate_layer.weight.data - overload_w.data)}"

        return


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

