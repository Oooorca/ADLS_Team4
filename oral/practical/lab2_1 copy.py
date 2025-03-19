import torch
import json

# 1. 读取 .pt 文件
data = torch.load("results/new_lab2_1_results.pt")
# data 是一个 dict:
# {
#   "grid_trials": study_grid.trials,  # <class 'list'>, 每个元素是 optuna.trial.FrozenTrial
#   "tpe_trials": study_tpe.trials     # 同上
# }

grid_accuracies = []
for trial in data["grid_trials"]:
    # trial.value 就是 objective 返回的 eval_accuracy
    grid_accuracies.append(trial.value)

tpe_accuracies = []
for trial in data["tpe_trials"]:
    tpe_accuracies.append(trial.value)

nas_results = {
    "GridSampler": grid_accuracies,
    "TPESampler": tpe_accuracies,
}

# 2. 保存为 JSON
with open("new_nas_results.json", "w") as f:
    json.dump(nas_results, f, indent=2)

print("new_nas_results.json 导出完成！")
