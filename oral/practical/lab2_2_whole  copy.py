import torch
import json
import optuna
import matplotlib.pyplot as plt
import os
import copy  # 用于深拷贝

from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoConfig
from chop.pipelines import CompressionPipeline
from chop import MaseGraph
from chop.nn.modules import Identity
from chop.tools.utils import deepsetattr
from chop.tools import get_tokenized_dataset, get_trainer
from optuna.samplers import TPESampler
import torch.nn as nn

# 定义一个简单的 Identity 模块（用于替换全连接层）
class MyIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

# -----------------------------
# 1. 加载 NAS 结果（未压缩 NAS 结果），假设文件 nas_results.json 已存在
# -----------------------------
with open("nas_results.json", "r") as f:
    nas_results = json.load(f)

# 使用 TPESampler（假设 Task1 中 TPESampler 效果最好）
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
# 4. 定义搜索空间（超参数与线性层替换策略）
# -----------------------------
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choice": [
        "linear",    # 使用 nn.Linear
        "identity",  # 替换为自定义 MyIdentity
    ],
}

# -----------------------------
# 5. CompressionPipeline 配置（量化与剪枝配置）
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
            "data_in_width": 16,
            "data_in_frac_width": 8,
            # weight
            "weight_width": 16,
            "weight_frac_width": 8,
            # bias
            "bias_width": 16,
            "bias_frac_width": 8,
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
# 6. 定义目标函数（合并共同训练、压缩、两种评估）
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
    
    # 选择线性层替换策略
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
    
    # (c) 共同训练 1 epoch
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1
    )
    trainer.train()
    
    # (d) 压缩：量化 + 剪枝
    model.cpu()  # 切回 CPU 进行图构建
    mg = MaseGraph(model, hf_input_names=["input_ids", "attention_mask", "labels", "token_type_ids"])
    mg, _ = compression_pipeline(
        mg,
        pass_args={
            "quantize_transform_pass": copy.deepcopy(quantization_config),
            "prune_transform_pass": copy.deepcopy(pruning_config),
        },
    )
    # 得到压缩后的模型
    compressed_model = mg.model

    # (e) 压缩后直接评估（不进行后续训练）
    compressed_model.to(device)
    trainer_no_post = get_trainer(
        model=compressed_model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=0  # 不进行额外训练
    )
    score_no_post = trainer_no_post.evaluate()["eval_accuracy"]

    # (f) 对压缩模型进行后续训练（post-training）后评估
    # 为保证两次评估在相同基础上进行，使用深拷贝压缩后的模型
    model_for_post = copy.deepcopy(compressed_model)
    model_for_post.to(device)
    trainer_with_post = get_trainer(
        model=model_for_post,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1
    )
    trainer_with_post.train()
    score_with_post = trainer_with_post.evaluate()["eval_accuracy"]

    # 保存两个评估结果到 trial 的属性中，便于后续对比绘图
    trial.set_user_attr("score_no_post", score_no_post)
    trial.set_user_attr("score_with_post", score_with_post)
    
    # 目标函数返回带后续训练后的结果作为优化目标
    return score_with_post

# -----------------------------
# 7. 运行搜索并保存结果
# -----------------------------
study = optuna.create_study(direction="maximize", sampler=best_sampler)
n_trials = 30
study.optimize(objective, n_trials=n_trials)

# 提取搜索结果：带后续训练和不带后续训练的评估
results_with_post = [trial.user_attrs["score_with_post"] for trial in study.trials]
results_no_post   = [trial.user_attrs["score_no_post"] for trial in study.trials]

with open("compression_results_with_post_2.json", "w") as f:
    json.dump(results_with_post, f, indent=2)
with open("compression_results_no_post_2.json", "w") as f:
    json.dump(results_no_post, f, indent=2)

# -----------------------------
# 8. 绘图对比
# -----------------------------
# (A) NAS（未压缩 NAS）的结果，从 nas_results.json 中读取（这里假设存储的是 TPESampler 的 NAS 结果列表）
tpe_vals = nas_results["TPESampler"]
max_acc_nas = []
best_so_far = -1e9
for v in tpe_vals:
    if v is not None:
        best_so_far = max(best_so_far, v)
    max_acc_nas.append(best_so_far)

# (B) Compression-Aware NAS (with post-training)
max_acc_with_post = []
best_so_far = -1e9
for v in results_with_post:
    if v is not None:
        best_so_far = max(best_so_far, v)
    max_acc_with_post.append(best_so_far)

# (C) Compression-Aware NAS (no post-training)
max_acc_no_post = []
best_so_far = -1e9
for v in results_no_post:
    if v is not None:
        best_so_far = max(best_so_far, v)
    max_acc_no_post.append(best_so_far)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(max_acc_nas) + 1), max_acc_nas, label="NAS (No Compression)", marker='o')
plt.plot(range(1, len(max_acc_with_post) + 1), max_acc_with_post, label="Compression-Aware NAS (with post-training)", marker='o')
plt.plot(range(1, len(max_acc_no_post) + 1), max_acc_no_post, label="Compression-Aware NAS (no post-training)", marker='o')

plt.xlabel("Number of Trials")
plt.ylabel("Max Achieved Accuracy")
plt.title("Comparison: NAS vs Compression-Aware NAS")
plt.legend()
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/lab2_t2_compression_nas_comparison.png")
plt.show()
