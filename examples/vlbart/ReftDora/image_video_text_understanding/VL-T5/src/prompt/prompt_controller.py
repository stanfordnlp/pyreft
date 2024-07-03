import torch
import torch.nn as nn
from .prompt_modeling import InputPrompts


class PromptController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prompts = nn.ModuleDict(dict())
        self.tasks = config.tasks
        self.use_input_prompt = config.use_input_prompt
        self.use_single_prompt = config.use_single_prompt
        self.prompts = self.construct_prompts(self.tasks)

    def get_task(self, task):
        return task 

    def construct_prompts(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """

        if self.use_single_prompt:
            if self.use_input_prompt:
                prompt = InputPrompts(self.config)

            for task in tasks:
                self.prompts[task] = prompt

        else:
            for task in tasks:
                if self.use_input_prompt:
                    prompt = InputPrompts(self.config)
                    
                    self.prompts[task] = prompt

        return self.prompts

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def get_prompt(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.prompts[task]

    def forward(self, bsz, device, task):
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
        prompt_module = self.get_prompt(task)

        trainable_prompt = prompt_module.get_prompt(bsz, device)

        return trainable_prompt

