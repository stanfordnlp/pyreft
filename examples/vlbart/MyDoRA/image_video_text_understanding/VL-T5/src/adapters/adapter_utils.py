"""Implementation of different utility functions for adapter layers."""
import torch
import torch.nn as nn
from transformers.activations import get_activation


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


class TaskHyperNet(nn.Module):
    """This module generates the task-embeddings from the initial feeded task embeddings."""

    def __init__(self, config, input_dim):
        super(TaskHyperNet, self).__init__()
        self.task_hidden_dim = config.task_hidden_dim
        self.projected_task_embedding_dim = config.projected_task_embedding_dim
        self.task_embeding_generator = nn.Sequential(
            linear_layer(input_dim, self.task_hidden_dim),
            nn.ReLU(),
            linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        return self.task_embeding_generator(task_embedding).view(-1)


class LayerNormHyperNet(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config):
        super(LayerNormHyperNet, self).__init__()
        self.task_embedding_dim = config.projected_task_embedding_dim \
            if config.train_task_embeddings else config.task_embedding_dim
        self.weight_generator = linear_layer(self.task_embedding_dim, config.input_dim)
        self.bias_generator = linear_layer(self.task_embedding_dim, config.input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)


class TaskEmbeddingController(nn.Module):
    """Main module controlling task embeddings."""

    def __init__(self, config):
        super(TaskEmbeddingController, self).__init__()
        # self.device = config.device
        self.task_embedding_dim = config.task_embedding_dim
        self.tasks = config.tasks
        self.task_to_task_embeddings = {task: task for task in self.tasks}
        if config.task_to_embeddings is not None:
            self.task_to_task_embeddings = config.task_to_embeddings
            self.tasks = self.task_to_task_embeddings.values()
        self.set_task_embeddings(self.tasks)
        self.train_task_embeddings = config.train_task_embeddings
        if self.train_task_embeddings:
            self.task_hyper_net = TaskHyperNet(config)

    def get_task(self, task):
        return self.task_to_task_embeddings[task]

    def set_task_embeddings(self, tasks):
        self.task_to_embeddings = nn.ParameterDict(dict())
        for task in tasks:
            task_embedding = torch.Tensor(torch.randn(self.task_embedding_dim))
            self.task_to_embeddings[task] = nn.Parameter(task_embedding)

    def forward(self, task):
        task_mapped = self.get_task(task)
        task_embedding = self.task_to_embeddings[task_mapped]
        if self.train_task_embeddings:
            return self.task_hyper_net(task_embedding)
        return task_embedding
