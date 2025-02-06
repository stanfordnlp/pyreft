import pyvene as pv
import torch
import os


def count_parameters(model):
    """Count parameters of a model that require gradients"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ReftModel(pv.IntervenableModel):
    """
    Base model for Reft methods.
    """
    def __init__(self, config, model, **kwargs):
        super().__init__(config, model, **kwargs)

    @staticmethod
    def _convert_to_reft_model(intervenable_model):
        reft_model = ReftModel(intervenable_model.config, intervenable_model.model)
        # Copy any other necessary attributes
        for attr in vars(intervenable_model):
            setattr(reft_model, attr, getattr(intervenable_model, attr))
        return reft_model

    @staticmethod
    def load(*args, **kwargs):
        try:
            model = pv.IntervenableModel.load(*args, **kwargs)
        except KeyError:
            print("This model is unsupported by pyvene. Set up a reft model and use `load_pv_undefined_model` instead.")
        return ReftModel._convert_to_reft_model(model)

    @staticmethod
    def load_pv_undefined_model(load_directory, reft_model):
        """
        This is a function to load a model which is unsupported by pyvene.
        """
        for key in reft_model.interventions.keys():
            reft_model.interventions[key][0].load_state_dict(torch.load(os.path.join(load_directory, f"intkey_{key}.bin"), weights_only=True))
        return reft_model

    def print_trainable_parameters(self):
        """
        Print trainable parameters.
        """
        _linked_key_set = set([])
        trainable_intervention_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(v, pv.TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        trainable_intervention_parameters += count_parameters(v)
                else:
                    trainable_intervention_parameters += count_parameters(v)

        trainable_model_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

        all_model_parameters = sum(
            p.numel() for p in self.model.parameters())

        total_trainable_parameters = trainable_intervention_parameters + trainable_model_parameters
        
        print(
            f"trainable intervention params: {trainable_intervention_parameters:,d} || trainable model params: {trainable_model_parameters:,d}\n"
            f"model params: {all_model_parameters:,d} || trainable%: {100 * total_trainable_parameters / all_model_parameters}"
        )

