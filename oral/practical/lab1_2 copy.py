import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer
import os
from pathlib import Path

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 稀疏率范围
sparsity_levels = [0.1,0.2, 0.3, 0.4,0.5,0.6, 0.7, 0.8,0.9]

# 数据集与分词器
checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"

mg = MaseGraph(
    model,
    hf_input_names=[
        "input_ids",
        "attention_mask",
        "labels",
    ],
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)
mg = MaseGraph.from_checkpoint(f"models/qat_model_width_16")

# 剪枝配置函数
def get_pruning_config(sparsity, method="l1-norm", scope="local"):
    return {
        "weight": {
            "sparsity": sparsity,
            "method": method,
            "scope": scope,
        },
        "activation": {
            "sparsity": sparsity,
            "method": method,
            "scope": scope,
        },
    }

# 精度评估函数
def evaluate_model(model):
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=5,
    )
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_accuracy"]

# 初始化结果存储
random_accuracies = []
l1_norm_accuracies = []

# 剪枝与评估循环
for sparsity in sparsity_levels:
    # Random Pruning
    random_pruning_config = get_pruning_config(sparsity, method="random")
    mg_random, _ = passes.prune_transform_pass(mg, pass_args=random_pruning_config)
    random_accuracy = evaluate_model(mg_random.model)
    random_accuracies.append(random_accuracy)

    # L1-Norm Pruning
    l1_pruning_config = get_pruning_config(sparsity, method="l1-norm")
    mg_l1, _ = passes.prune_transform_pass(mg, pass_args=l1_pruning_config)
    l1_accuracy = evaluate_model(mg_l1.model)
    l1_norm_accuracies.append(l1_accuracy)

    # 保存剪枝后的模型
    os.makedirs("models_lab1_2", exist_ok=True)
    # 使用 state_dict() 方式保存
    torch.save(mg_random.model.state_dict(), f"models_lab1_2/random_pruned_sparsity_{sparsity}.pt")
    torch.save(mg_l1.model.state_dict(), f"models_lab1_2/l1_pruned_sparsity_{sparsity}.pt")

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(sparsity_levels, random_accuracies, label="Random Pruning", marker="o")
plt.plot(sparsity_levels, l1_norm_accuracies, label="L1-Norm Pruning", marker="o")
plt.xlabel("Sparsity")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Sparsity for Pruning Strategies")
plt.legend()
plt.grid(True)
plt.savefig("plots/lab1_t2_pruning_accuracy_vs_sparsity.png")
plt.show()
