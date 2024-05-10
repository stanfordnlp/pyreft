IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"

gsm8k_template = """%s\nAnswer the above question. First think step by step and then answer the final number.\n"""

no_header_prompt_template = """\
### Instruction:
%s

### Response:
"""

commonsense_prompt_template = """Please choose the correct answer to the question. Begin your response with "the correct answer is".

### Example
Please choose the correct answer to the question: Which piece of safety equipment is used to keep mold spores from entering the respiratory system?

Answer1: safety goggles Answer2: breathing mask Answer3: rubber gloves Answer4: lead apron

Answer format: answer1/answer2/answer3/answer4

Your response: the correct answer is answer2

### Instruction
%s

Your response:
"""

alpaca_prompt_template = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

alpaca_prompt_no_input_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""