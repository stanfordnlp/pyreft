from concurrent.futures import ProcessPoolExecutor
import queue
import subprocess
def evaluate(dataset, gpu):
    print('*******dataset:', dataset)

    command = f"CUDA_VISIBLE_DEVICES={gpu} python evaluate.py \
               --model LLaMA-7B \
               --adapter LoRA \
               --dataset {dataset} \
               --base_model '../LLM/models/llama-7b-hf' \
               --lora_weights './trained_models/llama-lora'"

    result = subprocess.run(command, shell=True, text=True, capture_output=False)
    print(f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}")
    return gpu


datasets = ['AQuA', 'AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'SVAMP']
gpus = [1, 2, 3]
tasks_queue = queue.Queue()
gpu_queue = queue.Queue()

for gpu in gpus:
    gpu_queue.put(gpu)
for task in datasets:
    tasks_queue.put(task)

num_processes = min(len(datasets), len(gpus))  # number of processes to run in parallel

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = [executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()) for i in range(num_processes)]
    for future in futures:
        gpu_id = future.result()
        gpu_queue.put(gpu_id)
        if tasks_queue.qsize() > 0:
            futures.append(executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()))






