import pyvene as pv
import pyreft as pr
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq


def make_dataloader(dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


class ReftAdversarialTrainer(pr.ReftTrainer):    
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        """
        Assumes that first half of batch is positive examples and second half is negative examples.
        """
        pos_inputs = {k: v[:len(v)//2] for k, v in inputs.items()}
        neg_inputs = {k: v[len(v)//2:] for k, v in inputs.items()}
        pos_loss = super().compute_loss(intervenable, pos_inputs, return_outputs)
        neg_loss = super().compute_loss(intervenable, neg_inputs, return_outputs)

        if return_outputs:
            return pos_loss[0] - neg_loss[0], pos_loss[1]
        return pos_loss - neg_loss


class ReftAdversarialTrainerForCausalLM(ReftAdversarialTrainer):
    def get_train_dataloader(self) -> DataLoader:
        # for adversarial training code to work, we cannot shuffle our dataset...
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=False)