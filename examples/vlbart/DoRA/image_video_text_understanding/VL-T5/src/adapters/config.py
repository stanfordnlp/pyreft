from dataclasses import dataclass


@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = False
    non_linearity: str = "gelu_new"
    reduction_factor: int = 16
    weight_init_range = 1e-2
    # Whether to use conditional layer norms for adapters.
    conditional_layer_norm = False
    hidden_dim = 128
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.
    train_adapters_blocks = True

    task_adapter_layers_encoder = None
    task_adapter_layers_decoder = None
    task_adapter_in_decoder = True
    intrinsic_dim = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"

    # Hypercomplex adapters parameters 
    hypercomplex_adapters = False
    hypercomplex_division = 8
    learn_phm = True
    hypercomplex_nonlinearity="glorot-uniform"
    shared_phm_rule = False 
    factorized_phm = False 
    shared_W_phm = False
    factorized_phm_rule = False 
    phm_c_init = "normal"
    phm_rank = 1
    phm_init_range=0.01

    # prefix-tuning parameters.
    prefix_dim = 100
    init_prefix_from_vocab = False 
    kronecker_prod = False  

    # BitFit configuration.
    bitfit = False

    # Low-rank adapters.
    low_rank_adapters = False
    low_rank_w_init = "glorot-uniform"
    low_rank_rank = 1

    # whether using single adapter for all tasks
    use_single_adapter = False


class MetaAdapterConfig(AdapterConfig):
    """Implements Meta adapter in which a hyper-network generates the parameters of
     adapter layers. In this case we have a task embeddings which is feed to the
     hyper-network to allow it generate the weights for the adapter layers."""
    task_embedding_dim = 512
    task_embedding_dir = None
    hidden_dim = 128
    train_task_embeddings = False
    non_linearity: str = "gelu_new"
    projected_task_embedding_dim = 64
    task_hidden_dim = 128
    parametric_task_embedding = False
    # If Specified, uses one hypernet to generates the adapters weights.
    unique_hyper_net = True
    unique_hyper_net_layer_norm = True
    # We consider only one hyper-net for all the blocks of transformer.
    efficient_unique_hyper_net = False
    task_to_embeddings=None


@dataclass
class CompactorConfig(object):
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = False
    non_linearity: str = "gelu_new"
    reduction_factor: int = 16
    weight_init_range = 1e-2
    # Whether to use conditional layer norms for adapters.
    hidden_dim = 128
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.
    task_adapter_layers_encoder = None
    task_adapter_layers_decoder = None
    task_adapter_in_decoder = True
    intrinsic_dim = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"

    # Hypercomplex adapters parameters 
    hypercomplex_adapters = True
    hypercomplex_division = 4
    train_task_adapters = True
    learn_phm = True
    hypercomplex_nonlinearity="glorot-uniform"
    shared_phm_rule = True 
    factorized_phm = True 
    shared_W_phm = False
    factorized_phm_rule = False 
    phm_c_init = "normal"
    phm_rank = 1
    phm_init_range=0.0001

    # prefix-tuning parameters.
    prefix_dim = 100
    init_prefix_from_vocab = False 
    kronecker_prod = False  

    # BitFit configuration.
    bitfit = False

    # Low-rank adapters.
    low_rank_adapters = False
    low_rank_w_init = "glorot-uniform"
    low_rank_rank = 1

    # whether using single adapter for all tasks
    use_single_adapter = False


@dataclass
class LRAdapterConfig(object):
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = False
    non_linearity: str = "gelu_new"
    reduction_factor: int = 16
    weight_init_range = 1e-2
    # Whether to use conditional layer norms for adapters.
    hidden_dim = 128
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.
    task_adapter_layers_encoder = None
    task_adapter_layers_decoder = None
    task_adapter_in_decoder = True
    intrinsic_dim = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"

    # Hypercomplex adapters parameters 
    hypercomplex_adapters = False
    hypercomplex_division = 4
    train_task_adapters = True
    learn_phm = True
    hypercomplex_nonlinearity="glorot-uniform"
    shared_phm_rule = True 
    factorized_phm = True 
    shared_W_phm = False
    factorized_phm_rule = False 
    phm_c_init = "normal"
    phm_rank = 1
    phm_init_range=0.0001

    # prefix-tuning parameters.
    prefix_dim = 100
    init_prefix_from_vocab = False 
    kronecker_prod = False  

    # BitFit configuration.
    bitfit = False

    # Low-rank adapters.
    low_rank_adapters = True
    low_rank_w_init = "glorot-uniform"
    low_rank_rank = 1

    # whether using single adapter for all tasks
    use_single_adapter = False