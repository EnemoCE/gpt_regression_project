from quinine import (
    tstring,
    tlist,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge

model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "ready_data": merge(tboolean, default(False)),
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian"])),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "max_iters": merge(tinteger, default(1000)),
    "eval_interval": merge(tinteger, default(100)),
    "eval_iters": merge(tinteger, default(50)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

transform_schema = {
    "switch_params": merge(tlist, nullable),
    "duplicate_params": merge(tlist, nullable),
    "slice_params": merge(tlist, nullable),
    "average_params": merge(tlist, nullable),
    "full_backbone_rnn_iters":  merge(tinteger, default(2)),
    "no_layernorm_full_backbone_copy": merge(tboolean, default(False)),
    "first_n_layers": merge(tinteger, nullable),
    "new_backbone_training": merge(tboolean, default(True)),
    "diverge_new_backbone_training": merge(tboolean, default(True)),
    "clear_readout2": merge(tboolean, default(True)),
    "readout2_training": merge(tboolean, default(True)),
    "post_eval": merge(tboolean, default(True)),
    "cfm_loss": merge(tlist, nullable),
    "retrain_readout2_iters":  merge(tinteger, default(1800)),
    "model_variants": merge(tlist, default([])),
    "transform_choice": merge(tinteger, default(1)),
    "transform_variants": merge(tlist, default([])),
}

auto_transform_schema = {
    "permute_bounds_params": merge(tlist, nullable),
    "permute_interval": merge(tinteger, default(10)),
    "permute_model": merge(tboolean, default(True)),
    "use_custom_permute": merge(tboolean, default(False)),
    "auto_transform_choice": merge(tinteger, default(1)),
    "auto_transform_variants": merge(tlist, default([])),
}

experiment_schema = {
    "log_model_weights": merge(tboolean, default(True)),
    "show_normality": merge(tboolean, default(True)),
    "show_layer_errors": merge(tboolean, default(True)),
    "show_embedding_confusion": merge(tboolean, default(True)),
    "logarithmic_scale": merge(tboolean, default(True)),
    "transform_conf": stdict(transform_schema),
    "auto_transform_conf": stdict(auto_transform_schema),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "experiment_conf": stdict(experiment_schema),
    "test_run": merge(tboolean, default(False)),
}
