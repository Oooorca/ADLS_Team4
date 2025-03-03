import torch
import optuna
import matplotlib.pyplot as plt
from transformers import AutoModel
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from chop.nn.quantized.modules.linear import LinearInteger
from copy import deepcopy
from pathlib import Path
import dill

#让不同层使用不同的整数和小数位宽

# 在 Tutorial 6 代码中，所有 LinearInteger 层都使用相同的 width 和 fractional width。
# 不同层对量化的敏感度不同，使用相同的量化位宽可能不是最佳方案。
# 目标是让每一层独立选择适合的 width 和 fractional width。

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# IMDb 数据集
dataset_name = "imdb"
tokenizer_checkpoint = "bert-base-uncased"
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# 预训练 BERT
checkpoint = "prajjwal1/bert-tiny"
# model = AutoModel.from_pretrained(checkpoint)

with open(f"{Path.home()}/tutorial_5_best_model.pkl", "rb") as f:
    model = dill.load(f)

# 定义搜索空间
search_space = {
    "linear_layer_choices": [torch.nn.Linear, LinearInteger],
    "width_choices": [8, 16, 32],
    "fractional_width_choices": [2, 4, 8],
}

# 训练函数
def construct_model(trial):
    trial_model = deepcopy(model)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            new_layer_cls = trial.suggest_categorical(f"{name}_type", search_space["linear_layer_choices"])
            if new_layer_cls == torch.nn.Linear:
                continue
            
            # 采样量化精度
            width = trial.suggest_categorical(f"{name}_width", search_space["width_choices"])
            frac_width = trial.suggest_categorical(f"{name}_frac_width", search_space["fractional_width_choices"])

            kwargs = {"in_features": layer.in_features, "out_features": layer.out_features}
            kwargs["config"] = {
                "data_in_width": width,
                "data_in_frac_width": frac_width,
                "weight_width": width,
                "weight_frac_width": frac_width,
                "bias_width": width,
                "bias_frac_width": frac_width,
            }

            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data
            deepsetattr(trial_model, name, new_layer)

    return trial_model

# 目标函数
def objective(trial):
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer.train()
    eval_results = trainer.evaluate()

    trial.set_user_attr("model", model)
    return eval_results["eval_accuracy"]

# 运行 Optuna 搜索
sampler = optuna.samplers.RandomSampler()
study = optuna.create_study(direction="maximize", study_name="bert-int-quantization", sampler=sampler)

n_trials = 20
study.optimize(objective, n_trials=n_trials)

# 结果绘图
trials = range(1, n_trials + 1)
best_acc = [max([t.value for t in study.trials[:i]]) for i in trials]

plt.figure(figsize=(10, 6))
plt.plot(trials, best_acc, label="Best Accuracy", marker="o")
plt.xlabel("Number of Trials")
plt.ylabel("Max Achieved Accuracy")
plt.title("Integer Quantization Search: Accuracy vs Trials")
plt.legend()
plt.grid(True)
plt.savefig("plots/lab3_t1_accuracy_vs_trials.png")
plt.show()
