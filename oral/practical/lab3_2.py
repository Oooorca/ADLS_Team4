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

# -- 针对二值化量化层的维度问题，添加安全版 binary_quantizer --
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
# -- 安全版补丁结束 --

# -------------------------
# 1. 环境与数据准备
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_name = "imdb"
tokenizer_checkpoint = "bert-base-uncased"
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# 读取预训练BERT模型（假设已存储在 tutorial_5_best_model.pkl）
checkpoint = "prajjwal1/bert-tiny"
with open(f"{Path.home()}/tutorial_5_best_model.pkl", "rb") as f:
    base_model = dill.load(f).to(device).eval()

# 将可选精度用字符串表示，并建立映射
layer_types = {
    "torch.nn.Linear": torch.nn.Linear,
    "LinearInteger": __import__("chop.nn.quantized.modules.linear", fromlist=["LinearInteger"]).LinearInteger,
    "LinearMinifloatDenorm": __import__("chop.nn.quantized.modules.linear", fromlist=["LinearMinifloatDenorm"]).LinearMinifloatDenorm,
    "LinearMinifloatIEEE": __import__("chop.nn.quantized.modules.linear", fromlist=["LinearMinifloatIEEE"]).LinearMinifloatIEEE,
    "LinearLog": __import__("chop.nn.quantized.modules.linear", fromlist=["LinearLog"]).LinearLog,
    "LinearBlockFP": __import__("chop.nn.quantized.modules.linear", fromlist=["LinearBlockFP"]).LinearBlockFP,
    "LinearBlockLog": __import__("chop.nn.quantized.modules.linear", fromlist=["LinearBlockLog"]).LinearBlockLog,
    "LinearBinary": __import__("chop.nn.quantized.modules.linear", fromlist=["LinearBinary"]).LinearBinary,
    "LinearBinaryScaling": __import__("chop.nn.quantized.modules.linear", fromlist=["LinearBinaryScaling"]).LinearBinaryScaling,
    "LinearBinaryResidualSign": __import__("chop.nn.quantized.modules.linear", fromlist=["LinearBinaryResidualSign"]).LinearBinaryResidualSign,
}


def construct_model_for_precision(trial, precision_str, base_model):
    """
    将 base_model 中所有 Linear 层替换为 precision_str 对应的量化层，
    并对该量化层的超参数做随机搜索（若精度是全精度 torch.nn.Linear，则不改动）。
    """
    trial_model = deepcopy(base_model)
    new_layer_cls = layer_types[precision_str]

    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            if new_layer_cls == torch.nn.Linear:
                # 保持原始全精度Linear，不做任何替换
                continue

            # 在终端打印
            print(f"[Trial {trial.number}] Setting layer '{name}' to {precision_str}")

            kwargs = {"in_features": layer.in_features, "out_features": layer.out_features}
            # 根据具体类型搜索不同的超参数
            if precision_str == "LinearInteger":
                width = trial.suggest_categorical(f"{name}_width", [8, 16, 32])
                frac_width = trial.suggest_categorical(f"{name}_frac_width", [2, 4, 8])
                kwargs["config"] = {
                    "data_in_width": width,
                    "data_in_frac_width": frac_width,
                    "weight_width": width,
                    "weight_frac_width": frac_width,
                    "bias_width": width,
                    "bias_frac_width": frac_width,
                }
            elif precision_str in ("LinearMinifloatDenorm", "LinearMinifloatIEEE"):
                data_in_width = trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32])
                data_in_exponent_width = trial.suggest_categorical(f"{name}_data_in_exponent_width", [3, 4, 5])
                data_in_exponent_bias = trial.suggest_categorical(f"{name}_data_in_exponent_bias", [0, 1, 2])
                weight_width = trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32])
                weight_exponent_width = trial.suggest_categorical(f"{name}_weight_exponent_width", [3, 4, 5])
                weight_exponent_bias = trial.suggest_categorical(f"{name}_weight_exponent_bias", [0, 1, 2])
                bias_width = trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32])
                bias_exponent_width = trial.suggest_categorical(f"{name}_bias_exponent_width", [3, 4, 5])
                bias_exponent_bias = trial.suggest_categorical(f"{name}_bias_exponent_bias", [0, 1, 2])
                kwargs["config"] = {
                    "data_in_width": data_in_width,
                    "data_in_exponent_width": data_in_exponent_width,
                    "data_in_exponent_bias": data_in_exponent_bias,
                    "weight_width": weight_width,
                    "weight_exponent_width": weight_exponent_width,
                    "weight_exponent_bias": weight_exponent_bias,
                    "bias_width": bias_width,
                    "bias_exponent_width": bias_exponent_width,
                    "bias_exponent_bias": bias_exponent_bias,
                }
            elif precision_str == "LinearLog":
                weight_width = trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32])
                weight_exponent_bias = trial.suggest_categorical(f"{name}_weight_exponent_bias", [0, 1, 2])
                data_in_width = trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32])
                data_in_exponent_bias = trial.suggest_categorical(f"{name}_data_in_exponent_bias", [0, 1, 2])
                bias_width = trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32])
                bias_exponent_bias = trial.suggest_categorical(f"{name}_bias_exponent_bias", [0, 1, 2])
                kwargs["config"] = {
                    "weight_width": weight_width,
                    "weight_exponent_bias": weight_exponent_bias,
                    "data_in_width": data_in_width,
                    "data_in_exponent_bias": data_in_exponent_bias,
                    "bias_width": bias_width,
                    "bias_exponent_bias": bias_exponent_bias,
                }
            elif precision_str == "LinearBlockFP":
                weight_block_size = trial.suggest_categorical(f"{name}_weight_block_size", [8, 16])
                weight_width = trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32])
                weight_exponent_width = trial.suggest_categorical(f"{name}_weight_exponent_width", [3, 4, 5])
                weight_exponent_bias = trial.suggest_categorical(f"{name}_weight_exponent_bias", [0, 1, 2])
                data_in_width = trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32])
                data_in_exponent_width = trial.suggest_categorical(f"{name}_data_in_exponent_width", [3, 4, 5])
                data_in_exponent_bias = trial.suggest_categorical(f"{name}_data_in_exponent_bias", [0, 1, 2])
                data_in_block_size = trial.suggest_categorical(f"{name}_data_in_block_size", [8, 16])
                bias_width = trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32])
                bias_exponent_width = trial.suggest_categorical(f"{name}_bias_exponent_width", [3, 4, 5])
                bias_exponent_bias = trial.suggest_categorical(f"{name}_bias_exponent_bias", [0, 1, 2])
                bias_block_size = trial.suggest_categorical(f"{name}_bias_block_size", [8, 16])
                kwargs["config"] = {
                    "weight_block_size": weight_block_size,
                    "weight_width": weight_width,
                    "weight_exponent_width": weight_exponent_width,
                    "weight_exponent_bias": weight_exponent_bias,
                    "data_in_width": data_in_width,
                    "data_in_exponent_width": data_in_exponent_width,
                    "data_in_exponent_bias": data_in_exponent_bias,
                    "data_in_block_size": data_in_block_size,
                    "data_in_skip_first_dim": True,
                    "bias_width": bias_width,
                    "bias_exponent_width": bias_exponent_width,
                    "bias_exponent_bias": bias_exponent_bias,
                    "bias_block_size": bias_block_size,
                }
            elif precision_str == "LinearBlockLog":
                weight_block_size = trial.suggest_categorical(f"{name}_weight_block_size", [8, 16])
                weight_width = trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32])
                weight_exponent_bias_width = trial.suggest_categorical(f"{name}_weight_exponent_bias_width", [4, 5, 6])
                data_in_width = trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32])
                data_in_exponent_bias_width = trial.suggest_categorical(f"{name}_data_in_exponent_bias_width", [4, 5, 6])
                data_in_block_size = trial.suggest_categorical(f"{name}_data_in_block_size", [8, 16])
                bias_width = trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32])
                bias_exponent_bias_width = trial.suggest_categorical(f"{name}_bias_exponent_bias_width", [4, 5, 6])
                bias_block_size = trial.suggest_categorical(f"{name}_bias_block_size", [8, 16])
                kwargs["config"] = {
                    "weight_block_size": [weight_block_size],
                    "weight_width": weight_width,
                    "weight_exponent_bias_width": weight_exponent_bias_width,
                    "data_in_width": data_in_width,
                    "data_in_exponent_bias_width": data_in_exponent_bias_width,
                    "data_in_block_size": [data_in_block_size],
                    "data_in_skip_first_dim": True,
                    "bias_width": bias_width,
                    "bias_exponent_bias_width": bias_exponent_bias_width,
                    "bias_block_size": [bias_block_size],
                }
            elif precision_str == "LinearBinary":
                kwargs["config"] = {
                    "scaling_factor": trial.suggest_float(f"{name}_scaling_factor", 0.1, 1.0),
                    "weight_stochastic": trial.suggest_categorical(f"{name}_weight_stochastic", [True, False]),
                    "weight_bipolar": True,
                }
            elif precision_str == "LinearBinaryScaling":
                #scaling_factor = trial.suggest_float(f"{name}_scaling_factor", 0.1, 1.0)
                weight_stochastic = trial.suggest_categorical(f"{name}_weight_stochastic", [True, False])
                weight_bipolar = trial.suggest_categorical(f"{name}_weight_bipolar", [True])
                data_in_stochastic = trial.suggest_categorical(f"{name}_data_in_stochastic", [True, False])
                data_in_bipolar = trial.suggest_categorical(f"{name}_data_in_bipolar", [True])
                bias_stochastic = trial.suggest_categorical(f"{name}_bias_stochastic", [True, False])
                bias_bipolar = trial.suggest_categorical(f"{name}_bias_bipolar", [True])
                binary_training = trial.suggest_categorical(f"{name}_binary_training", [True, False])
                kwargs["config"] = {
                    "scaling_factor": scaling_factor,
                    "weight_stochastic": weight_stochastic,
                    "weight_bipolar": weight_bipolar,
                    "data_in_stochastic": data_in_stochastic,
                    "data_in_bipolar": data_in_bipolar,
                    "bias_stochastic": bias_stochastic,
                    "bias_bipolar": bias_bipolar,
                    "binary_training": binary_training,
                }
            elif precision_str == "LinearBinaryResidualSign":
                #scaling_factor = trial.suggest_float(f"{name}_scaling_factor", 0.1, 1.0)
                weight_stochastic = trial.suggest_categorical(f"{name}_weight_stochastic", [True, False])
                weight_bipolar = trial.suggest_categorical(f"{name}_weight_bipolar", [True])
                data_in_stochastic = trial.suggest_categorical(f"{name}_data_in_stochastic", [True, False])
                data_in_bipolar = trial.suggest_categorical(f"{name}_data_in_bipolar", [True])
                bias_stochastic = trial.suggest_categorical(f"{name}_bias_stochastic", [True, False])
                bias_bipolar = trial.suggest_categorical(f"{name}_bias_bipolar", [True])
                binary_training = trial.suggest_categorical(f"{name}_binary_training", [True, False])
                kwargs["config"] = {
                    "scaling_factor": scaling_factor,
                    "weight_stochastic": weight_stochastic,
                    "weight_bipolar": weight_bipolar,
                    "data_in_stochastic": data_in_stochastic,
                    "data_in_bipolar": data_in_bipolar,
                    "bias_stochastic": bias_stochastic,
                    "bias_bipolar": bias_bipolar,
                    "binary_training": binary_training,
                }
            
            new_layer = new_layer_cls(**kwargs)
            # 拷贝原权重
            new_layer.weight.data = layer.weight.data
            deepsetattr(trial_model, name, new_layer)
    return trial_model


def run_study_for_precision(precision_str, n_trials=5, storage=None):
    """
    针对给定 precision_str，运行 n_trials 次搜索（主要搜索其量化层对应的超参数）。
    返回该 study 对象。
    """
    def objective(trial):
        trial_model = construct_model_for_precision(trial, precision_str, base_model)
        trainer = get_trainer(
            model=trial_model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=1
        )
        trainer.train()
        eval_results = trainer.evaluate()
        return eval_results["eval_accuracy"]
    
    if storage is None:
        # 可以使用内存Storage，或者使用 sqlite
        # 这里用内存模式：只保留当前进程的 study
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    else:
        # 如果要持久化，传入 sqlite url
        study = optuna.create_study(direction="maximize", storage=storage, load_if_exists=True)
    
    study.optimize(objective, n_trials=n_trials)
    return study


if __name__ == "__main__":
    # 你想要对比的多种精度（可以自行增删）
    precision_candidates = [
        "LinearInteger",
        "LinearMinifloatDenorm",
        "LinearMinifloatIEEE",
        "LinearLog",
        "LinearBlockFP",
        "LinearBlockLog",
        "LinearBinary",
        "LinearBinaryScaling",
        "LinearBinaryResidualSign",
    ]
    
    # 每种精度搜索的 trial 数
    N_TRIALS = 10
    
    # 用于保存每种精度在每个 trial 后的“最好准确率”
    results_dict = {}
    
    for prec in precision_candidates:
        print(f"\n=== Running study for precision: {prec} ===")
        study = run_study_for_precision(precision_str=prec, n_trials=N_TRIALS)
        
        # 取出该 study 在 [1..N_TRIALS] 的“best so far”
        best_acc_list = []
        for i in range(1, N_TRIALS + 1):
            # 截取前 i 个 trial
            partial_trials = study.trials[:i]
            # 获取其中最高的准确率
            best_val = max(t.value for t in partial_trials if t.value is not None)
            best_acc_list.append(best_val)
        
        results_dict[prec] = best_acc_list
    
    # -------------------------
    # 绘图：每种精度一条曲线
    # -------------------------
    print("[MAIN] All studies done. Now plotting...")
    plt.figure(figsize=(10, 6))
    for prec, acc_list in results_dict.items():
        plt.plot(range(1, N_TRIALS + 1), acc_list, marker="o", label=prec)
    
    plt.xlabel("Number of Trials")
    plt.ylabel("Max Achieved Accuracy")
    plt.title("Precision Search (Single-Precision) Comparison")
    plt.grid(True)
    plt.legend()
    out_fig = "plots/lab3_t2_accuracy_vs_trials.png"
    plt.savefig(out_fig)
    print(f"[MAIN] Figure saved to {out_fig}. Exiting now.")
