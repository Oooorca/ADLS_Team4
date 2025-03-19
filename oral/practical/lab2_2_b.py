import torch
import json
import optuna
import matplotlib.pyplot as plt
import os
import copy  # 用于深拷贝传入的字典

from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoConfig
from chop.pipelines import CompressionPipeline
from chop import MaseGraph
from chop.nn.modules import Identity
from chop.tools.utils import deepsetattr
from chop.tools import get_tokenized_dataset, get_trainer
from optuna.samplers import TPESampler

import torch.nn as nn
import inspect

class MyIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

# -----------------------------
# 1. 加载 Task1 结果（未压缩 NAS 结果）
# -----------------------------
with open("nas_results.json", "r") as f:
    nas_results = json.load(f)

# 使用 TPESampler（假设 Task1 中 TPESampler 的效果最好）
best_sampler = TPESampler()

# -----------------------------
# 2. 检查 GPU
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 3. 数据集 & Tokenizer
# -----------------------------
checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# -----------------------------
# 4. 定义搜索空间（确保超参数值均合理）
# -----------------------------
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choice": [
        "linear",    # 保持 nn.Linear
        "identity",  # 替换为自定义 MyIdentity
    ],
}

# -----------------------------
# 5. CompressionPipeline 配置
# -----------------------------
compression_pipeline = CompressionPipeline()

quantization_config = {
    "by": "type",
    "default": {
        "config": {
            "name": None,
        }
    },
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 32,
            "data_in_frac_width": 16,
            # weight
            "weight_width": 32,
            "weight_frac_width": 16,
            # bias
            "bias_width": 32,
            "bias_frac_width": 16,
        }
    },
}

pruning_config = {
    "weight": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
    "activation": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
}

# -----------------------------
# 6. 定义两个目标函数
# -----------------------------

def objective_with_post(trial):
    """
    压缩感知搜索：压缩后进行后续训练（1 epoch）
    """
    # (a) 根据 trial 选择网络超参
    config = AutoConfig.from_pretrained(checkpoint)
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][idx])
    # 检查：确保 hidden_size 能被 num_heads 整除
    if config.hidden_size % config.num_heads != 0:
        raise optuna.exceptions.TrialPruned("hidden_size must be divisible by num_heads")
    linear_choice_idx = trial.suggest_int("linear_layer_choice", 0, len(search_space["linear_layer_choice"]) - 1)
    linear_choice_str = search_space["linear_layer_choice"][linear_choice_idx]

    # (b) 构建模型
    model = AutoModelForSequenceClassification.from_config(config)
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear) and (layer.in_features == layer.out_features):
            if linear_choice_str == "linear":
                pass
            elif linear_choice_str == "identity":
                deepsetattr(model, name, MyIdentity())
            else:
                raise ValueError(f"Unknown linear layer choice: {linear_choice_str}")
    model.to(device)

    # (c) 常规训练 1 epoch
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1
    )
    trainer.train()

    # (d) 压缩：量化 + 剪枝
    model.cpu()
    mg = MaseGraph(model, hf_input_names=["input_ids", "attention_mask", "labels", "token_type_ids"])
    mg, _ = compression_pipeline(
        mg,
        pass_args={
            "quantize_transform_pass": copy.deepcopy(quantization_config),
            "prune_transform_pass": copy.deepcopy(pruning_config),
        },
    )

    # (e) 压缩后进行再训练 1 epoch
    compressed_model = mg.model
    compressed_model.to(device)
    trainer = get_trainer(
        model=compressed_model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1
    )
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_accuracy"]

def objective_without_post(trial):
    """
    压缩感知搜索：压缩后不进行后续训练，直接评估
    """
    # (a) 根据 trial 选择网络超参
    config = AutoConfig.from_pretrained(checkpoint)
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][idx])
    if config.hidden_size % config.num_heads != 0:
        raise optuna.exceptions.TrialPruned("hidden_size must be divisible by num_heads")
    linear_choice_idx = trial.suggest_int("linear_layer_choice", 0, len(search_space["linear_layer_choice"]) - 1)
    linear_choice_str = search_space["linear_layer_choice"][linear_choice_idx]

    # (b) 构建模型
    model = AutoModelForSequenceClassification.from_config(config)
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear) and (layer.in_features == layer.out_features):
            if linear_choice_str == "linear":
                pass
            elif linear_choice_str == "identity":
                deepsetattr(model, name, MyIdentity())
            else:
                raise ValueError(f"Unknown linear layer choice: {linear_choice_str}")
    model.to(device)

    # (c) 常规训练 1 epoch
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1
    )
    trainer.train()

    # (d) 压缩：量化 + 剪枝
    model.cpu()
    mg = MaseGraph(model, hf_input_names=["input_ids", "attention_mask", "labels", "token_type_ids"])
    mg, _ = compression_pipeline(
        mg,
        pass_args={
            "quantize_transform_pass": copy.deepcopy(quantization_config),
            "prune_transform_pass": copy.deepcopy(pruning_config),
        },
    )

    # (e) 不进行压缩后再训练，直接评估
    compressed_model = mg.model
    compressed_model.to(device)
    # 这里直接设置 num_train_epochs=0，即不再进行训练
    trainer = get_trainer(
        model=compressed_model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=0
    )
    eval_results = trainer.evaluate()
    return eval_results["eval_accuracy"]

# -----------------------------
# 7. 分别运行搜索，保存两套结果
# -----------------------------
# 先运行压缩感知搜索（带后续训练）的目标函数（假设你之前已生成该结果并保存在 compression_results_with_post.json）
# 如果尚未生成，可以取消下面的注释运行：
# study_with_post = optuna.create_study(direction="maximize", sampler=best_sampler)
# study_with_post.optimize(objective_with_post, n_trials=20)
# compression_results_with_post = [trial.value for trial in study_with_post.trials]
# with open("compression_results_with_post.json", "w") as f:
#     json.dump(compression_results_with_post, f, indent=2)

# 现在运行压缩感知搜索（不带后续训练）的目标函数
study_without_post = optuna.create_study(direction="maximize", sampler=best_sampler)
study_without_post.optimize(objective_without_post, n_trials=20)
compression_results_without_post = [trial.value for trial in study_without_post.trials]
with open("compression_results_without_post.json", "w") as f:
    json.dump(compression_results_without_post, f, indent=2)

# -----------------------------
# 8. 绘图对比：NAS（未压缩）、压缩感知搜索（带后续训练）和（不带后续训练）
# -----------------------------
# 读取 NAS 结果
with open("nas_results.json", "r") as f:
    nas_results = json.load(f)
nas_tpe_vals = nas_results["TPESampler"]

# 读取压缩感知搜索（带后续训练）的结果
with open("compression_results_with_post.json", "r") as f:
    comp_results_with_post = json.load(f)

# 读取压缩感知搜索（不带后续训练）的结果
with open("compression_results_without_post.json", "r") as f:
    comp_results_without_post = json.load(f)

def cumulative_best(results):
    cum_best = []
    best = -1e9
    for v in results:
        if v is not None:
            best = max(best, v)
        cum_best.append(best)
    return cum_best

max_acc_nas = cumulative_best(nas_tpe_vals)
max_acc_with_post = cumulative_best(comp_results_with_post)
max_acc_without_post = cumulative_best(comp_results_without_post)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(max_acc_nas)+1), max_acc_nas, label="NAS (No Compression)")
plt.plot(range(1, len(max_acc_with_post)+1), max_acc_with_post, label="Compression-Aware NAS (with post-training)")
plt.plot(range(1, len(max_acc_without_post)+1), max_acc_without_post, label="Compression-Aware NAS (without post-training)")
plt.xlabel("Number of Trials")
plt.ylabel("Max Achieved Accuracy")
plt.title("Comparison: NAS vs Compression-Aware NAS")
plt.legend()
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/new_compression_nas_comparison.png")
plt.show()
