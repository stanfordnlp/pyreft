"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation
from .adapter_modeling import Adapter, HyperComplexAdapter, LowRankAdapter, OutputAdapter


class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        super().__init__()
        # low-rank adapters.
        self.low_rank_adapters = config.low_rank_adapters
        # self.intrinsic_projections_path = os.path.join(config.output_dir, "intrinsic_projections")
        self.config = config
        self.adapters = nn.ModuleDict(dict())
        self.tasks = config.tasks
        # self.device = config.device
        self.shared_phm_rule = config.shared_phm_rule
        self.hypercomplex_adapters = config.hypercomplex_adapters
        self.use_single_adapter = config.use_single_adapter
        self.share_up_sampler = config.share_up_sampler
        self.share_down_sampler = config.share_down_sampler
        self.shared_phm_rule_over_tasks = config.shared_phm_rule_over_tasks
        self.adapters = self.construct_adapters(self.tasks)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.input_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.input_dim)

    def get_task(self, task):
        return task 

    def construct_adapters(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """

        if self.use_single_adapter:
            if self.hypercomplex_adapters:
                adapter = HyperComplexAdapter(self.config)
            elif self.low_rank_adapters:
                adapter = LowRankAdapter(self.config)
            else:
                adapter = Adapter(self.config)

            for task in tasks:
                self.adapters[task] = adapter

        else:
            for task in tasks:
                if self.hypercomplex_adapters:
                    self.adapters[task] = HyperComplexAdapter(self.config)
                elif self.low_rank_adapters:
                    self.adapters[task] = LowRankAdapter(self.config)
                else:
                    self.adapters[task] = Adapter(self.config)

            if self.share_up_sampler:
                layer = self.adapters[tasks[0]].up_sampler # extract the layer of the adapters for first task
                for task in tasks:
                    self.adapters[task].up_sampler = layer

            if self.share_down_sampler:
                layer = self.adapters[tasks[0]].down_sampler # extract the layer of the adapters for first task
                for task in tasks:
                    self.adapters[task].down_sampler = layer

            if self.hypercomplex_adapters and self.shared_phm_rule_over_tasks and not self.shared_phm_rule:
                up_phm_rule = self.adapters[tasks[0]].up_sampler.phm_rule
                down_phm_rule = self.adapters[tasks[0]].down_sampler.phm_rule
                for task in tasks:
                    self.adapters[task].up_sampler.phm_rule = up_phm_rule
                    self.adapters[task].down_sampler.phm_rule = down_phm_rule

        return self.adapters

    def disable_adapters(self, tasks):
        """
        Given a list of tasks, it freezes their corresponding adapter layers'
        parameters.
        Args:
           tasks: List of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for param in adapter.parameters():
                param.requires_grad = False

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def enable_adapters(self, tasks):
        """
        Given a list of tasks, it unfreezes their corresponding adapter layers.
        Args:
            tasks: Given list of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for name, param in adapter.named_parameters():
                if self.config.hypercomplex_adapters and not self.config.learn_phm:
                    if not "phm_rule" in name:
                        param.requires_grad = True
                else:
                    param.requires_grad = True

    def get_adapter(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.adapters[task]

    def forward(self, inputs, task):
        """
        Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer.
        """
        task = self.get_task(task)
        # Enables the adapter layer for the given task.
        # self.enable_adapters(task)
        # # Disable other adapters.
        # other_tasks = [x for x in self.tasks if x != task]
        # if not self.use_single_adapter: # use separate adapters
        #     self.disable_adapters(other_tasks)
        adapter = self.get_adapter(task)
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs


class AdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adapter = Adapter(config)

        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.d_model)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, inputs):
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = self.adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs


class OutputParallelAdapterLayer(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.adapter = OutputAdapter(config, output_dim)

        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.d_model)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(output_dim)

    def forward(self, inputs):
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = self.adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs
        return outputs

    def resize_output_dim(self, resized_size):
        self.adapter.resize_up_sampler(resized_size)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(resized_size)


class MetaLayersAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a unique hyper-network."""

    def __init__(self, config):
        super().__init__()
        self.activation_type = config.non_linearity.lower()
        self.input_dim = config.input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter

        self.track_z = config.track_z

    def apply_layer_norm(self, inputs, layer_norm_weights):
        """Applies layer norm to the inputs."""
        return torch.nn.functional.layer_norm(inputs, (self.input_dim,),
                                              weight=layer_norm_weights.weight,
                                              bias=layer_norm_weights.bias)

    def call_adapter(self, inputs, adapter_weights):
        """Computes the output of the adapter layers."""
        down = F.linear(inputs, weight=adapter_weights.down.weight,
                        bias=adapter_weights.down.bias)
        middle = get_activation(self.activation_type)(down)

        if self.track_z:
            self.z = middle

        output = F.linear(middle, weight=adapter_weights.up.weight,
                          bias=adapter_weights.up.bias)
        return output

    def forward(self, inputs, adapter_weights):
        z = self.apply_layer_norm(inputs, adapter_weights.pre_norm) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, adapter_weights)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_layer_norm(outputs, adapter_weights.post_norm)
        outputs = outputs + inputs
        return outputs
