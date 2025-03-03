import torch
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import os

from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
from chop.nn.modules import Identity
from chop.tools import get_tokenized_dataset, get_trainer

# 为了演示：用到 GridSampler 和 TPESampler
from optuna.samplers import GridSampler, TPESampler

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 1. 准备数据集
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
# 2. 定义搜索空间
# -----------------------------
# 注意，这里将 linear_layer_choices 合并成一个顶层参数 "linear_layer_choice"
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    # 只定义一个，用于控制所有线性层的替换方式
    "linear_layer_choice": [
        nn.Linear,
        Identity,
    ],
}

# -----------------------------
# 3. 构建模型的函数
# -----------------------------
def construct_model(trial):
    # 从预训练配置里加载
    config = AutoConfig.from_pretrained(checkpoint)

    # 1) 先为基本超参选择索引
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][chosen_idx])

    # 2) 为“线性层的替换方式”选择索引
    chosen_linear_idx = trial.suggest_int("linear_layer_choice", 
                                          0, 
                                          len(search_space["linear_layer_choice"]) - 1)
    chosen_linear_cls = search_space["linear_layer_choice"][chosen_linear_idx]

    # 根据 config 初始化模型
    trial_model = AutoModelForSequenceClassification.from_config(config)

    # 遍历所有 nn.Linear，并统一替换成 chosen_linear_cls
    # （只替换 in_features == out_features 的线性层）
    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            if chosen_linear_cls == nn.Linear:
                # 保持原状
                continue
            elif chosen_linear_cls == Identity:
                new_layer = Identity()
                deepsetattr(trial_model, name, new_layer)
            else:
                raise ValueError(f"Unknown layer type: {chosen_linear_cls}")

    return trial_model.to(device)

# -----------------------------
# 4. 定义 objective
# -----------------------------
def objective(trial):
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,  # 只训练 1 轮
    )

    trainer.train()
    eval_results = trainer.evaluate()
    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]

# -----------------------------
# 5. 比较 GridSampler 和 TPESampler
# -----------------------------

# (A) 运行 GridSampler
print("Running NAS with GridSampler")

# 1) 根据 search_space 构造网格
#    注意：'num_layers', 'num_heads', 'hidden_size', 'intermediate_size', 'linear_layer_choice'
#    每个都对应一组值的索引 [0,1,2,...]
grid_sampler = GridSampler({
    param: list(range(len(vals))) for param, vals in search_space.items()
})

study_grid = optuna.create_study(direction="maximize", sampler=grid_sampler)
# 注意：如果你希望覆盖完整网格，需要让 n_trials = 所有组合数量
# 比如 3(层数) * 4(头数) * 5(隐藏维度) * 5(中间层维度) * 2(线性层选择) = 3 * 4 * 5 * 5 * 2 = 600
# 这里为了演示写 20，可根据作业需求自行决定
study_grid.optimize(objective, n_trials=30)

# (B) 运行 TPESampler
print("Running NAS with TPESampler")
tpe_sampler = TPESampler()
study_tpe = optuna.create_study(direction="maximize", sampler=tpe_sampler)
study_tpe.optimize(objective, n_trials=30)

# -----------------------------
# 6. 保存实验结果
# -----------------------------
os.makedirs("results", exist_ok=True)
torch.save(
    {"grid_trials": study_grid.trials, "tpe_trials": study_tpe.trials},
    "results/new_lab2_1_results.pt",
)

# -----------------------------
# 7. 绘制对比曲线
# -----------------------------
def plot_results(study, label):
    trials = study.trials
    max_acc = []
    best_so_far = 0
    for t in trials:
        if t.value is not None:
            best_so_far = max(best_so_far, t.value)
        max_acc.append(best_so_far)

    plt.plot(range(len(max_acc)), max_acc, label=label, marker='o')

plt.figure(figsize=(10, 6))

plot_results(study_grid, "GridSampler")
plot_results(study_tpe, "TPESampler")

plt.xlabel("Number of Trials")
plt.ylabel("Max Achieved Accuracy")
plt.title("Comparison of NAS Strategies: Grid vs TPE")
plt.legend()
plt.grid(True)

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/lab2_t1_nas_comparison.png")
plt.show()
