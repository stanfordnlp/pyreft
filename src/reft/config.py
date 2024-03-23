import pyvene as pv


class ReftConfig(pv.IntervenableConfig):
    """
    Reft config for Reft methods.
    """
    def __init__(
        self,
        task_type: str,
        representations,
        **kwargs,
    ):
        super().__init__(
            representations=representations,
            **kwargs,
        )
        self.task_type = task_type