from dataclasses import dataclass


@dataclass
class EncoderPromptConfig(object):
    seq_len = 0
    input_dim = 768
    mid_dim = 768
    use_input_prompt = True
    use_single_prompt = False

@dataclass
class DecoderPromptConfig(object):
    seq_len = 0
    input_dim = 768
    mid_dim = 768
    use_input_prompt = True
    use_single_prompt = False