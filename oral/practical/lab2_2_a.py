import torch
import json
import optuna
import matplotlib.pyplot as plt
import os
import copy  # 添加此行

from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoConfig
from chop.pipelines import CompressionPipeline
from chop import MaseGraph
from chop.nn.modules import Identity
from chop.tools.utils import deepsetattr
from chop.tools import get_tokenized_dataset, get_trainer
from optuna.samplers import TPESampler  # 这里直接导入 TPESampler

import torch.nn as nn
import inspect

class MyIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

# -----------------------------
# 1. 加载 Task1 结果
# -----------------------------
with open("nas_results.json", "r") as f:
    nas_results = json.load(f)

# 根据 Task1 结果选择最佳采样器（这里假设使用 TPESampler）
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
# 6. Compression-Aware NAS 的目标函数
# -----------------------------
def objective(trial):
    # (a) 根据 trial 选择网络超参
    config = AutoConfig.from_pretrained(checkpoint)
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][idx])
    # 简单检查：确保 hidden_size 能被 num_heads 整除
    if config.hidden_size % config.num_heads != 0:
        raise optuna.exceptions.TrialPruned("hidden_size must be divisible by num_heads")

    # 线性层选项
    linear_choice_idx = trial.suggest_int("linear_layer_choice", 0, len(search_space["linear_layer_choice"]) - 1)
    linear_choice_str = search_space["linear_layer_choice"][linear_choice_idx]

    # (b) 构建模型
    model = AutoModelForSequenceClassification.from_config(config)
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear) and (layer.in_features == layer.out_features):
            if linear_choice_str == "linear":
                pass  # 保持 nn.Linear
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
    # 将模型转回 CPU 并创建 MaseGraph，同时传入 hf_input_names 参数确保数据格式正确
    model.cpu()
    mg = MaseGraph(model, 
                   hf_input_names=[
                       "input_ids", 
                       "attention_mask", 
                       "labels", 
                       "token_type_ids"
                       ])
    mg, _ = compression_pipeline(
        mg,
        pass_args={
            "quantize_transform_pass": copy.deepcopy(quantization_config),
            "prune_transform_pass": copy.deepcopy(pruning_config),
        },
    )

    # (e) 再训练压缩后模型（将其移回 GPU）
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

# -----------------------------
# 7. 运行搜索
# -----------------------------
study = optuna.create_study(direction="maximize", sampler=best_sampler)
study.optimize(objective, n_trials=20)  # 示例跑20次

# -----------------------------
# 8. 保存 Task2 结果
# -----------------------------
compression_results = [trial.value for trial in study.trials]
with open("compression_results.json", "w") as f:
    json.dump(compression_results, f, indent=2)

# -----------------------------
# 9. 读取并与 Task1 对比绘图
# -----------------------------
with open("nas_results.json", "r") as f:
    nas_results = json.load(f)
tpe_vals = nas_results["TPESampler"]

plt.figure(figsize=(10, 6))
max_acc_tpe = []
best_so_far = -1e9
for v in tpe_vals:
    if v is not None:
        best_so_far = max(best_so_far, v)
    max_acc_tpe.append(best_so_far)
plt.plot(range(1, len(max_acc_tpe) + 1), max_acc_tpe, label="NAS (No Compression)")

max_acc_compressed = []
best_so_far = -1e9
for v in compression_results:
    if v is not None:
        best_so_far = max(best_so_far, v)
    max_acc_compressed.append(best_so_far)
plt.plot(range(1, len(max_acc_compressed) + 1), max_acc_compressed, label="Compression-Aware NAS")

plt.xlabel("Number of Trials")
plt.ylabel("Max Achieved Accuracy")
plt.title("Comparison: NAS vs Compression-Aware NAS")
plt.legend()
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/compression_nas_comparison.png")
plt.show()
