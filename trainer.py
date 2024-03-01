import pyvene as pv
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import os

class ReftTrainer(Trainer):
    def get_train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=self._train_batch_size, collate_fn=self.data_collator)
        return train_dataloader

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # run intervened forward pass
        _, cf_outputs = intervenable(
            {"input_ids": inputs["input_ids"]},
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
        )

        # lm loss on counterfactual labels
        lm_logits = cf_outputs.logits
        labels = inputs["labels"]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # return
        return (loss, cf_outputs) if return_outputs else loss

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save(save_directory=f"{output_dir}/intervenable_model")