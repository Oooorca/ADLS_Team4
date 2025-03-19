import torch
import optuna
import matplotlib.pyplot as plt
from transformers import AutoModel
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from copy import deepcopy
from pathlib import Path
import dill
import os
import numpy as np
from optuna.samplers import RandomSampler
from chop.nn.quantized.modules.linear import (
    LinearInteger, LinearMinifloatDenorm, 
    LinearMinifloatIEEE, LinearLog,
    LinearBlockFP, LinearBlockLog, LinearBinary,
    LinearBinaryScaling,
)


from chop.nn.quantizers import binary_quantizer as orig_binary_quantizer

def safe_binary_quantizer(x, stochastic, bipolar):
    original_shape = x.shape  # 记录原始形状
    if x.dim() < 4:
        # 扩展到 4D
        while x.dim() < 4:
            x = x.unsqueeze(-1)
        result = orig_binary_quantizer(x, stochastic=stochastic, bipolar=bipolar)
        # 将结果 reshape 回原始形状
        result = result.view(original_shape)
        return result
    else:
        return orig_binary_quantizer(x, stochastic=stochastic, bipolar=bipolar)

from chop.nn.quantizers import binary_quantizer
binary_quantizer = safe_binary_quantizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
with open(f"{Path.home()}/tutorial_5_best_model.pkl", "rb") as f:
    base_model = dill.load(f).to(device).eval()



search_space = {
    "data_in_width": [8, 16, 32],
    "data_in_frac_width": [2, 4, 8],
    "bias": [8, 16],
    "exponent_width": [4, 5, 6],
    "exponent_bias": [3, 7, 15],
    "skip_first_dim": [True, False],
    "block_size": [8, 16],
    "exponent_bias_width": [2, 3, 4],
    "data_in_stochastic": [True, False],
    "bias_stochastic": [True, False],
    "binary_training": [True, False],
    "residual_bipolar": [True, False],
    "linear_layer_choices": [
        torch.nn.Linear, 
        # LinearInteger,
        # LinearMinifloatDenorm, 
        # LinearMinifloatIEEE,
        # LinearLog, 
        # LinearBlockFP, 
        # LinearBlockLog,
        # LinearBinary, 
        LinearBinaryScaling,
        # LinearBinaryResidualSign
    ],
}


def construct_model(trial,base_model):
    trial_model = deepcopy(base_model)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            chosen_linear_idx = trial.suggest_int("linear_layer_choices",0,len(search_space["linear_layer_choices"]) - 1)
            new_layer_cls = search_space["linear_layer_choices"][chosen_linear_idx]
            
            width = trial.suggest_categorical(f"{name}_width", search_space["data_in_width"])
            frac_width= trial.suggest_categorical(f"{name}_frac_width", search_space["data_in_frac_width"])
            
            bias = trial.suggest_categorical(f"{name}_bias", search_space["bias"])
            exponent_width = trial.suggest_categorical(f"{name}_exponent_width", search_space["exponent_width"])
            exponent_bias = trial.suggest_categorical(f"{name}_exponent_bias", search_space["exponent_bias"])
            skip_first_dim = trial.suggest_categorical(f"{name}_skip_first_dim", search_space["skip_first_dim"])
            block_size = trial.suggest_categorical(f"{name}_block_size", search_space["block_size"])
            exponent_bias_width = trial.suggest_categorical(f"{name}_exponent_bias_width", search_space["exponent_bias_width"])
            stochastic = trial.suggest_categorical(f"{name}_stochastic", [True, False])
            bipolar = trial.suggest_categorical(f"{name}_bipolar", [True])
            binary_training = trial.suggest_categorical(f"{name}_binary_training", [True, False])
            scaling_factor = trial.suggest_float(f"{name}_scaling_factor",0.1,1.0)
            residual_bipolar = trial.suggest_categorical(f"{name}_residual_bipolar", search_space["residual_bipolar"])
            if new_layer_cls == torch.nn.Linear:
                continue
            
            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }
            print(f"Replacing {name}: in_features={layer.in_features}, out_features={layer.out_features}")
            if new_layer_cls == LinearInteger:
                kwargs["config"] = {
                    "data_in_width": width,
                    "data_in_frac_width": frac_width,
                    "weight_width": width,
                    "weight_frac_width": frac_width,
                    "bias_width": width,
                    "bias_frac_width": frac_width,
                }
            elif new_layer_cls == LinearLog:
                kwargs["config"] = {
                    "data_in_width": width,
                    "data_in_exponent_bias": bias,
                    "weight_width": width,
                    "weight_exponent_bias": bias,
                    "bias_width": width,
                    "bias_exponent_bias": bias,
                }
            elif new_layer_cls in [LinearMinifloatDenorm, LinearMinifloatIEEE]:
                kwargs["config"] = {
                    "weight_width": width,
                    "weight_exponent_width": exponent_width,
                    "weight_exponent_bias": exponent_bias,
                    "data_in_width": width,
                    "data_in_exponent_width": exponent_width,
                    "data_in_exponent_bias": exponent_bias,
                    "bias_width": width,
                    "bias_exponent_width": exponent_width,
                    "bias_exponent_bias": exponent_bias,
                }

            elif new_layer_cls == LinearBlockFP:
                kwargs["config"] = {
                    "weight_width": width,
                    "weight_exponent_width": exponent_width,
                    "weight_exponent_bias": exponent_bias,
                    "data_in_width": width,
                    "data_in_exponent_width": exponent_width,
                    "data_in_exponent_bias": exponent_bias,

                    
                    "bias_width": width,
                    "bias_exponent_width": exponent_width,
                    "bias_exponent_bias": exponent_bias,
                    
                    "weight_block_size": [block_size],
                    "data_in_block_size": [block_size],
                    "bias_block_size": [block_size],
                }
            elif new_layer_cls == LinearBlockLog:
                kwargs["config"] = {
                    "weight_width": width,
                    "weight_exponent_bias_width": exponent_bias_width,
                    "weight_block_size": [block_size],

                    "data_in_width": width,
                    "data_in_exponent_bias_width": exponent_bias_width,
                    "data_in_block_size": [block_size],


                    "bias_width": width,
                    "bias_exponent_bias_width": exponent_bias_width,
                    "bias_block_size": [block_size],
                }

            elif new_layer_cls == LinearBinary:
                kwargs["config"] = {
                    "scaling_factor": scaling_factor,
                    "weight_stochastic": stochastic,
                    "weight_bipolar": bipolar,
                }
            
            elif new_layer_cls == LinearBinaryScaling:
                kwargs["config"] = {
                    "data_in_stochastic": stochastic,
                    "bias_stochastic": stochastic,
                    "weight_stochastic": stochastic,
                    "data_in_bipolar": bipolar,
                    "bias_bipolar": bipolar,
                    "weight_bipolar": bipolar,
                    "binary_training": binary_training,
                }

            else:
                raise ValueError(f"Unknown layer type: {new_layer_cls}")
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data
            if new_layer.weight.shape == layer.weight.shape:
                new_layer.weight.data.copy_(layer.weight.data)
            else:
                print(f"Warning: Shape mismatch for {name}, skipping weight copy.")

            deepsetattr(trial_model, name, new_layer)

    return trial_model.to(device)

def objective(trial):
    model = construct_model(trial,base_model)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=0.1,
    )
    trainer.train()
    eval_results = trainer.evaluate()
    trial.set_user_attr("layer_search", eval_results["eval_accuracy"])

    return eval_results["eval_accuracy"]

tpe_LinearBinaryScaling_study = optuna.create_study(direction="maximize", sampler=RandomSampler(), study_name="TPE_LAB3_2_LinearBinaryScaling")
tpe_LinearBinaryScaling_study.optimize(objective, n_trials=10)