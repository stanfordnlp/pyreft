# Chat-model built with ReFT
Based on the notebook [`chat_model.ipynb`](https://github.com/stanfordnlp/pyreft/blob/main/examples/chat/chat_model.ipynb).

The goal is to show how this library integrates with HuggingFace, loading chat-models from HuggingFace.

## Loading artifacts from HuggingFace

pyReFT artifacts are minimum. For our chat-model, it can go as low as **1MB on disk**. Take a look at [our files](https://huggingface.co/zhengxuanzenwu/Loreft1k-Llama-2-7b-hf). You can follow the notebook to see how you can load ReFT-trained models from HuggingFace.

Note that pyReFT currently is not optimized for inference speed. If you are interested, feel free to open PR and work on it!
