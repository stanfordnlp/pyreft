import pyvene as pv
import pyreft as pr
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq


def make_dataloader(dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


class ReftAdversarialTrainer(pr.ReftTrainer):
    def __init__(self, adversarial=True, **kwargs):
        super().__init__(**kwargs)
        self.adversarial = adversarial
    
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # invert loss if adversarial
        loss = super().compute_loss(intervenable, inputs, return_outputs)
        if self.adversarial:
            loss = (-loss[0], loss[1]) if return_outputs else -loss
        return loss


class ReftAdversarialTrainerForCausalLM(ReftAdversarialTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)