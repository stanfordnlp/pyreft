# Paper's old source code folder

This folder contains all the code which later turns into the `pyreft` library. `pyreft` is more flexible and generic. However, all of our results are obtained using these scripts. We are happy to share these more raw code to provide additional insights in terms of replicating our results.


## Example: how we run the training code for GLUE tasks

```bash
python task_steer.py \
-task glue -train_dataset cola -model FacebookAI/roberta-base \
-seed 45 \
-l all -r 1 -p f3 -e 60 -lr 4e-4 \
-type ConditionedSourceLowRankRotatedSpaceIntervention \
-gradient_accumulation_steps 1 \
-batch_size 32 -eval_batch_size 32 \
-test_split validation -max_length 256 \
--metric_for_best_model matthews_correlation \
--dropout 0.2 --weight_decay 0.00000 \
--warmup_ratio 0.005 --logging_steps 20 \
--allow_cls_grad
```

Note that the major change here is the main running script and the intervention naming. You need to modify the command above for other tasks accordingly in the same fashion.