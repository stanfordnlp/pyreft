import pyvene as pv


class ReftConfig(pv.IntervenableConfig):
    """
    Reft config for Reft methods.
    """
    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)


    def to_dict(self):
        """
        Overwrite to bypass trainer initial config checking.
        
        If don't overwrite, it may throw json dump error based
        on your python version.
        """
        output = super().to_dict()
    
        if not isinstance(output["intervention_types"], list):
            output["intervention_types"] = [output["intervention_types"]]
        output["intervention_types"] = [
            str(t) for t in output["intervention_types"]]
        
        output["representations"] = [
            str(r) for r in output["representations"]]
    
        return output