import pyvene as pv


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
        model = pv.IntervenableModel.load(*args, **kwargs)
        return ReftModel._convert_to_reft_model(model)

    def print_trainable_parameters(self):
        """
        Print trainable parameters.
        """
        _linked_key_set = set([])
        trainable_intervention_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(v[0], pv.TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        trainable_intervention_parameters += count_parameters(v[0])
                else:
                    trainable_intervention_parameters += count_parameters(v[0])

        trainable_model_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

        all_model_parameters = sum(
            p.numel() for p in self.model.parameters())

        total_trainable_parameters = trainable_intervention_parameters + trainable_model_parameters
        
        print(
            f"trainable intervention params: {trainable_intervention_parameters:,d} || trainable model params: {trainable_model_parameters:,d}\n"
            f"model params: {all_model_parameters:,d} || trainable%: {100 * total_trainable_parameters / all_model_parameters}"
        )

    def unfreeze_intervention_parameters(self):
        """
        Unfreeze intervention parameters.
        """
        _linked_key_set = set([])
        trainable_intervention_parameters = {}
        for k, v in self.interventions.items():
            if isinstance(v[0], pv.TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        for n, p in v[0].named_parameters():
                            p.requires_grad = True
                            trainable_intervention_parameters[k+"#"+n] = p
                else:
                    for n, p in v[0].named_parameters():
                        p.requires_grad = True
                        trainable_intervention_parameters[k+"#"+n] = p
        #for n, p in trainable_intervention_parameters.items():
        #    print("Grad of " +n + " is ", p.grad, " value is ", p.data.norm())
        return trainable_intervention_parameters

