import pyvene as pv


class ReftConfig(pv.IntervenableConfig):
    """
    Reft config for Reft methods.
    """
    def __init__(
        self,
        representations=[pv.RepresentationConfig()],
        intervention_types=pv.VanillaIntervention,
        mode="parallel",
        **kwargs,
    ):
        super().__init__(
            representations=representations,
            intervention_types=intervention_types,
            mode=mode,
            **kwargs,
        )