# from transformers.deepspeed import HfDeepSpeedConfig
import json

def deepspeed_init(trainer, resume_from_checkpoint=None):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.
    If ``resume_from_checkpoint`` was passed then an attempt to resume from a previously saved checkpoint will be made.
    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
    Returns: model, optimizer, lr_scheduler
    """
    import deepspeed
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    optimizer = trainer.optim
    lr_scheduler = trainer.lr_scheduler

    with open(args.deepspeed, "r") as f:
        ds_config = json.load(f)

    if args.fp16:
        ds_config["fp16"] = {"enabled": True, "loss_scale": 0}

    ds_config["gradient_clipping"] = args.clip_grad_norm
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["zero_allow_untested_optimizer"] = True

    # hf_deepspeed_config = args.hf_deepspeed_config
    # hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    # config = HfDeepSpeedConfig(args.deepspeed)
    config = ds_config

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    # set the Deepspeed log level consistent with the trainer
    # ds_logger.setLevel(args.get_process_log_level())

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config_params=config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    if resume_from_checkpoint is not None:

        # it's possible that the user is trying to resume from model_path, which doesn't necessarily
        # contain a deepspeed checkpoint. e.g. examples just check if the dir exists and assume it's
        # a resume from a checkpoint and not just a local pretrained weight. So we check here if the
        # path contains what looks like a deepspeed checkpoint
        import glob

        deepspeed_checkpoint_dirs = sorted(glob.glob(f"{resume_from_checkpoint}/global_step*"))

        if len(deepspeed_checkpoint_dirs) > 0:
            logger.info(f"Attempting to resume from {resume_from_checkpoint}")
            # this magically updates self.optimizer and self.lr_scheduler
            load_path, _ = model.load_checkpoint(
                resume_from_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
            )
            if load_path is None:
                raise ValueError(f"[deepspeed] failed to resume from checkpoint {resume_from_checkpoint}")
        else:
            logger.info(f"{resume_from_checkpoint} doesn't have deepspeed checkpoints, doing nothing")

    return model, optimizer, lr_scheduler
