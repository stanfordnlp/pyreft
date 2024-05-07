"""Implements an Adapter and Hyper-adapter Layers."""
# Codes from https://github.com/rabeehk/hyperformer/blob/56b0f993eeec62f5a8c962704739912b5f039a9b/hyperformer/adapters/adapter_modeling.py
import torch
import torch.nn as nn

from .adapter_outputs import (SamplerOutput, LayerNormOutput,
                              AdapterT5BlockOutput, AdapterOutput)
from .adapter_utils import Activations, linear_layer, LayerNormHyperNet, TaskHyperNet


class AdapterHyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterHyperNet, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_task_embeddings = config.train_task_embeddings
        self.task_embedding_dim = config.projected_task_embedding_dim if \
            config.train_task_embeddings else config.task_embedding_dim
        # Considers weight and bias parameters for generating adapter weights.
        self.weight_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        weight = self.weight_generator(task_embedding).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(task_embedding).view(-1)
        return weight, bias


class AdapterLayersHyperNet(nn.Module):
    """This module generates the weights for all the meta adapter layers
    given the task embeddings and layer id."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterLayersHyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator = nn.Sequential(
            linear_layer(config.projected_task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(config.projected_task_embedding_dim, self.input_dim))

    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return SamplerOutput(weight=weight, bias=bias)


class AdapterLayersHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config, num_layers=6, include_cross_attention=False):
        super(AdapterLayersHyperNetController, self).__init__()
        self.num_layers = num_layers
        self.layer_norm_epsilon = 1e-6
        self.max_position_embeddings = 2
        self.task_embedding_dim = config.task_embedding_dim
        self.layer_id_embeddings = nn.Embedding(self.num_layers,
                                                self.task_embedding_dim)
        # self.token_type_embeddings = nn.Embedding(self.max_position_embeddings,
        #                                          self.task_embedding_dim).to(self.device)
        self.task_hypernet = TaskHyperNet(config, input_dim=self.task_embedding_dim * 2)
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        # Defines the adapters hyper-nets.
        self.feed_forward_up_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                       self.input_dim, self.down_sample_size)
        self.feed_forward_down_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                         self.down_sample_size, self.input_dim)
        self.self_attention_up_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                         self.input_dim, self.down_sample_size)
        self.self_attention_down_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                           self.down_sample_size, self.input_dim)

        self.include_cross_attention = include_cross_attention

        if include_cross_attention:
            self.cross_attention_up_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                         self.input_dim, self.down_sample_size)
            self.cross_attention_down_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                           self.down_sample_size, self.input_dim)
        # Defines the layer norms' hyper net.
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings
        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.feed_forward_pre_layernorm_hypernet = LayerNormHyperNet(config)
            self.self_attention_pre_layernorm_hypernet = LayerNormHyperNet(config)
            self.cross_attention_pre_layernorm_hypernet = LayerNormHyperNet(config)
        if self.add_layer_norm_after_adapter:
            self.feed_forward_post_layernorm_hypernet = LayerNormHyperNet(config)
            self.self_attention_post_layernorm_hypernet = LayerNormHyperNet(config)
            self.cross_attention_post_layernorm_hypernet = LayerNormHyperNet(config)
        config.train_task_embeddings = self.train_task_embeddings

    def get_embedding(self, task_embedding, layer_id):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device=task_embedding.device)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        embeddings = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1)], axis=0)
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, task_embedding, layer_id):
        embeddings = self.get_embedding(task_embedding, layer_id)
        # Generates the adapters weights in feed-forward and self-attention modules.
        feed_forward_down = self.feed_forward_down_sampler_hyper_net(embeddings)
        feed_forward_up = self.feed_forward_up_sampler_hyper_net(embeddings)
        self_attention_down = self.self_attention_down_sampler_hyper_net(embeddings)
        self_attention_up = self.self_attention_up_sampler_hyper_net(embeddings)
        if self.include_cross_attention:
            cross_attention_down = self.cross_attention_down_sampler_hyper_net(embeddings)
            cross_attention_up = self.cross_attention_up_sampler_hyper_net(embeddings)
            cross_attention_output = AdapterOutput(up=cross_attention_up, down=cross_attention_down)
        else:
            cross_attention_output = None

        feed_forward_output = AdapterOutput(up=feed_forward_up, down=feed_forward_down)
        self_attention_output = AdapterOutput(up=self_attention_up, down=self_attention_down)
        # Generates the weights and baises for pre and post layer norms.
        if self.add_layer_norm_before_adapter:
            weight, bias = self.feed_forward_pre_layernorm_hypernet(embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.self_attention_pre_layernorm_hypernet(embeddings)
            self_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            if self.include_cross_attention:
                weight, bias = self.cross_attention_pre_layernorm_hypernet(embeddings)
                cross_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)

        if self.add_layer_norm_after_adapter:
            weight, bias = self.feed_forward_post_layernorm_hypernet(embeddings)
            feed_forward_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.self_attention_post_layernorm_hypernet(embeddings)
            self_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            if self.include_cross_attention:
                weight, bias = self.cross_attention_post_layernorm_hypernet(embeddings)
                cross_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
        return AdapterT5BlockOutput(feed_forward=feed_forward_output,
                                    self_attention=self_attention_output,
                                    cross_attention=cross_attention_output)


class AdapterLayersOneHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config, num_layers=6, include_cross_attention=False):
        super(AdapterLayersOneHyperNetController, self).__init__()
        self.num_layers = num_layers
        self.layer_norm_epsilon = 1e-6
        self.max_position_embeddings = 2
        self.task_embedding_dim = config.task_embedding_dim
        self.layer_id_embeddings = nn.Embedding(self.num_layers,
                                                self.task_embedding_dim)
        # This is 2 types of adapters for feed-forward, and self-attention.
        self.adapters_block_type = nn.Embedding(3, self.task_embedding_dim)

        self.task_hypernet = TaskHyperNet(config, self.task_embedding_dim * 3)
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor

        self.include_cross_attention = include_cross_attention

        # Defines the adapters hyper-nets.
        self.up_sampler_hyper_net = AdapterLayersHyperNet(config, self.input_dim, self.down_sample_size)
        self.down_sampler_hyper_net = AdapterLayersHyperNet(config, self.down_sample_size, self.input_dim)

        # Defines the layer norms' hyper net.
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings
        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.pre_layernorm_hypernet = LayerNormHyperNet(config)
        if self.add_layer_norm_after_adapter:
            self.post_layernorm_hypernet = LayerNormHyperNet(config)
        config.train_task_embeddings = self.train_task_embeddings

    def get_embedding(self, task_embedding, layer_id, block_type):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device=task_embedding.device)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        type_id_tensor = torch.tensor([block_type], dtype=torch.long, device=task_embedding.device)
        type_embedding = self.adapters_block_type(type_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        type_embedding = type_embedding.view(-1)
        embeddings = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1), type_embedding.view(1, -1)],
                               axis=0)
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, task_embedding, layer_id):
        feed_forward_embeddings = self.get_embedding(task_embedding, layer_id, 0)
        self_attention_embeddings = self.get_embedding(task_embedding, layer_id, 1)

        # Generates the adapters weights in feed-forward.
        feed_forward_down = self.down_sampler_hyper_net(feed_forward_embeddings)
        feed_forward_up = self.up_sampler_hyper_net(feed_forward_embeddings)

        # Generates the adapter weights in self-attention.
        self_attention_down = self.down_sampler_hyper_net(self_attention_embeddings)
        self_attention_up = self.up_sampler_hyper_net(self_attention_embeddings)

        feed_forward_output = AdapterOutput(up=feed_forward_up, down=feed_forward_down)
        self_attention_output = AdapterOutput(up=self_attention_up, down=self_attention_down)

        if self.include_cross_attention:
            cross_attention_embeddings = self.get_embedding(task_embedding, layer_id, 2)

            cross_attention_down = self.down_sampler_hyper_net(cross_attention_embeddings)
            cross_attention_up = self.up_sampler_hyper_net(cross_attention_embeddings)
            cross_attention_output = AdapterOutput(up=cross_attention_up, down=cross_attention_down)
        else:
            cross_attention_output = None

        # Generates the weights and baises for pre and post layer norms.
        if self.add_layer_norm_before_adapter:
            weight, bias = self.pre_layernorm_hypernet(feed_forward_embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.pre_layernorm_hypernet(self_attention_embeddings)
            self_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)

            if self.include_cross_attention:
                weight, bias = self.pre_layernorm_hypernet(cross_attention_embeddings)
                cross_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)

        if self.add_layer_norm_after_adapter:
            weight, bias = self.post_layernorm_hypernet(feed_forward_embeddings)
            feed_forward_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.post_layernorm_hypernet(self_attention_embeddings)
            self_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)

            if self.include_cross_attention:
                weight, bias = self.post_layernorm_hypernet(cross_attention_embeddings)
                cross_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)

        return AdapterT5BlockOutput(feed_forward=feed_forward_output,
                                    self_attention=self_attention_output,
                                    cross_attention=cross_attention_output)