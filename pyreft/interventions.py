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
        # Store metrics for wandb logging
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

            # Log on first forward
            if not self._debug_logged:
                print(f"[DEBUG LoreftIntervention] First forward:")
                print(f"  base norm: {base_norm:.4f}")
                print(f"  Rh norm: {self.metrics['rotated_base_norm']:.4f}")
                print(f"  Wh+b norm: {self.metrics['learned_norm']:.4f}")
                print(f"  b norm: {b_norm:.4f}")
                print(f"  diff norm: {diff_norm:.4f}")
                print(f"  delta/base ratio: {self.metrics['delta_base_ratio']:.4f}")
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


class LoreftIntervention_Scale(LoreftIntervention):
    """
    LoReFT with learned scalar gating (starts at 0 for identity).
    
    LoReFT(h) = h + scale * R^T(Wh + b − Rh)
    
    At init: scale = 0, so output = h (identity).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = torch.nn.Parameter(torch.zeros(1))
        self._debug_logged = False
    
    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        delta = torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), 
            self.rotate_layer.weight.T
        )
        # Debug: log scale value on first forward pass
        if not self._debug_logged:
            print(f"[DEBUG LoreftIntervention_Scale] First forward: scale={self.scale.item():.6f}, delta_norm={delta.norm().item():.4f}")
            self._debug_logged = True
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
        # Debug logging (off by default)
        self.debug = kwargs.get("debug", False)
        self._debug_logged = False
        self.metrics = {}

    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.learned_source.weight.dtype)
        learned = self.act_fn(self.learned_source(cast_base))
        delta = torch.matmul(learned.to(self.rotate_layer.weight.dtype), self.rotate_layer.weight.T)

        # Store metrics for logging (only if debug=True)
        if self.debug:
            # For DiReFT: diff = Wh + b (no subtraction), ||delta|| = ||learned|| since R is orthonormal
            learned_norm = learned.norm().item()
            base_norm = base.norm().item()
            b_norm = self.learned_source.bias.norm().item()
            self.metrics = {
                "base_norm": base_norm,
                "learned_norm": learned_norm,
                "b_norm": b_norm,
                "diff_norm": learned_norm,  # For DiReFT, diff = learned = Wh + b
                "delta_base_ratio": learned_norm / (base_norm + 1e-8),
            }
            if not self._debug_logged:
                print(f"[DEBUG DireftIntervention] First forward:")
                print(f"  base norm: {base_norm:.4f}")
                print(f"  Wh+b norm: {learned_norm:.4f}, b norm: {b_norm:.4f}")
                print(f"  delta/base ratio: {self.metrics['delta_base_ratio']:.4f}")
                self._debug_logged = True

        output = base + delta
        return self.dropout(output.to(base.dtype))


class MoeloreftIntervention(
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention
):
    """
    MoE-LoReFT(h) = h + R^T(∑_{i=1}^{e}(s_i * W_i * h) + b - Rh)

    Mixture-of-experts variant of LoReFT where W is replaced by a weighted
    combination of e expert W matrices, with top-k expert selection.

    Args:
        num_experts: Number of expert W matrices (default: 4)
        top_k: Number of experts to select per token (default: 2)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.num_experts = kwargs.get("num_experts", 4)
        self.top_k = kwargs.get("top_k", 2)
        low_rank_dim = kwargs["low_rank_dimension"]
        dtype = kwargs.get("dtype", torch.bfloat16)

        # Orthogonal rotation layer (shared across experts)
        rotate_layer = LowRankRotateLayer(self.embed_dim, low_rank_dim, init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)

        # Router: projects h to e scores
        self.router = torch.nn.Linear(self.embed_dim, self.num_experts, bias=False).to(dtype)

        # Expert W matrices: e separate linear layers (embed_dim -> low_rank_dim)
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(self.embed_dim, low_rank_dim, bias=False).to(dtype)
            for _ in range(self.num_experts)
        ])

        # Shared bias
        self.bias = torch.nn.Parameter(torch.zeros(low_rank_dim, dtype=dtype))

        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))
        self.act_fn = ACT2FN.get(kwargs.get("act_fn"), ACT2FN["linear"])

        # Debug logging
        self.debug = kwargs.get("debug", False)
        self._debug_logged = False
        self.metrics = {}

        # Track expert activation counts
        self.expert_counts = torch.zeros(self.num_experts)
        self.total_tokens = 0

    def forward(self, base, source=None, subspaces=None):
        batch_size, seq_len, _ = base.shape
        cast_base = base.to(self.router.weight.dtype)

        # Compute router scores: (B, T, e)
        router_logits = self.router(cast_base)
        router_probs = torch.softmax(router_logits, dim=-1)

        # Top-k selection: (B, T, k)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Renormalize top-k probs
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute weighted expert outputs
        # For efficiency, compute all expert outputs then select
        # expert_outputs: (B, T, e, low_rank_dim)
        expert_outputs = torch.stack([expert(cast_base) for expert in self.experts], dim=2)

        # Gather top-k expert outputs: (B, T, k, low_rank_dim)
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[-1])
        selected_outputs = torch.gather(expert_outputs, dim=2, index=top_k_indices_expanded)

        # Weighted sum: (B, T, low_rank_dim)
        weighted_output = (selected_outputs * top_k_probs.unsqueeze(-1)).sum(dim=2)

        # Apply activation and add bias
        learned = self.act_fn(weighted_output + self.bias)

        # LoReFT: h + R^T(learned - Rh)
        rotated_base = self.rotate_layer(base)
        diff = learned.to(rotated_base.dtype) - rotated_base
        delta = torch.matmul(diff, self.rotate_layer.weight.T)

        # Track expert usage
        if self.debug:
            with torch.no_grad():
                # Count expert activations
                for i in range(self.num_experts):
                    self.expert_counts[i] += (top_k_indices == i).sum().item()
                self.total_tokens += batch_size * seq_len * self.top_k

                # Compute activation percentages
                expert_pcts = self.expert_counts / (self.total_tokens + 1e-8) * 100

                diff_norm = diff.norm().item()
                base_norm = base.norm().item()

                self.metrics = {
                    "base_norm": base_norm,
                    "diff_norm": diff_norm,
                    "delta_base_ratio": diff_norm / (base_norm + 1e-8),
                    "b_norm": self.bias.norm().item(),
                }
                # Add per-expert activation percentages (cumulative over training)
                for i in range(self.num_experts):
                    self.metrics[f"expert_{i}_pct"] = expert_pcts[i].item()

                if not self._debug_logged:
                    print(f"[DEBUG MoeloreftIntervention] First forward:")
                    print(f"  num_experts={self.num_experts}, top_k={self.top_k}")
                    print(f"  base norm: {base_norm:.4f}")
                    print(f"  diff norm: {diff_norm:.4f}")
                    print(f"  delta/base ratio: {self.metrics['delta_base_ratio']:.4f}")
                    self._debug_logged = True

        output = base + delta
        return self.dropout(output.to(base.dtype))

    def reset_expert_counts(self):
        """Reset expert activation counts (call at start of each epoch/eval)."""
        self.expert_counts.zero_()
        self.total_tokens = 0


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
        # Debug logging (off by default)
        self.debug = kwargs.get("debug", False)
        self._debug_logged = False
        self.metrics = {}

    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.learned_source.weight.dtype)
        learned = self.act_fn(self.learned_source(cast_base))
        delta = torch.matmul(learned, self.proj_layer.weight)

        # Store metrics for logging (only if debug=True)
        if self.debug:
            learned_norm = learned.norm().item()
            base_norm = base.norm().item()
            b_norm = self.learned_source.bias.norm().item()
            self.metrics = {
                "base_norm": base_norm,
                "learned_norm": learned_norm,
                "b_norm": b_norm,
                "diff_norm": learned_norm,  # For NodiReFT, diff = learned = W1h + b
                "delta_base_ratio": delta.norm().item() / (base_norm + 1e-8),
            }
            if not self._debug_logged:
                print(f"[DEBUG NodireftIntervention] First forward:")
                print(f"  base norm: {base_norm:.4f}")
                print(f"  W1h+b norm: {learned_norm:.4f}, b norm: {b_norm:.4f}")
                print(f"  delta/base ratio: {self.metrics['delta_base_ratio']:.4f}")
                self._debug_logged = True

        output = base + delta
        return self.dropout(output.to(base.dtype))

