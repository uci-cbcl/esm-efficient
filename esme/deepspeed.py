def import_deepseed():
    try:
        import deepspeed
        return deepspeed
    except ImportError as e:
        raise ImportError(
            'deepspeed not installed. Please install deepspeed to use deepspeed features.'
            'Install deepspeed using `pip install deepspeed` or conda.'
        ) from e


DEEPSPEED_STAGE2_CONFIG = {
    "steps_per_print": 1,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 1,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
    },
    'optimizer': {
        "type": "AdamW",
    }
}
