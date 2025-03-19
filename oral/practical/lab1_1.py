import torch
import matplotlib.pyplot as plt
from chop.tools import get_tokenized_dataset, get_trainer
import chop.passes as passes
from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import os
from pathlib import Path

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义固定点宽度范围
fixed_point_widths = [4,8,16,24,32]  # 修改为固定宽度值

# 初始化精度存储
ptq_accuracies = []
qat_accuracies = []

# 创建保存路径
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 加载 IMDb 数据集和分词器
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
mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_2_lora")

# 将模型移动到 GPU
mg.model.to(device)

# 评估模型精度函数
def evaluate_model(model):
    trainer = get_trainer(
        model=model,            # 训练器会内部处理数据和设备放置
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
    )
    eval_results = trainer.evaluate()
    return eval_results['eval_accuracy']

# 循环遍历固定点宽度
for width in fixed_point_widths:
    print(f"\nProcessing fixed-point width: {width}")


    # 5. 定义量化配置
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
                "data_in_width": width,
                "data_in_frac_width": width // 2,
                "weight_width": width,
                "weight_frac_width": width // 2,
                "bias_width": width,
                "bias_frac_width": width // 2,
            }
        },
    }

    # 6. 应用量化
    mg, _ = passes.quantize_transform_pass(
        mg,
        pass_args=quantization_config,
    )
    mg.model.to(device)  # 确保量化后的模型在 GPU 上

    # 7. 评估 PTQ 精度
    ptq_accuracy = evaluate_model(mg.model)
    ptq_accuracies.append(ptq_accuracy)

    # 使用 Tutorial 3 方法保存 PTQ 模型
    mg.export(f"models/ptq_model_width_{width}")

    # 8. 运行 QAT
    trainer = get_trainer(
        model=mg.model,  # 模型已在 GPU 上
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
    )
    trainer.train()

    # 9. 评估 QAT 精度
    qat_accuracy = evaluate_model(mg.model)
    qat_accuracies.append(qat_accuracy)

    # 使用 Tutorial 3 方法保存 QAT 模型
    mg.export(f"models/qat_model_width_{width}")

# 10. 绘制结果并保存图片
plt.figure(figsize=(10, 6))
plt.plot(fixed_point_widths, ptq_accuracies, label='PTQ', marker='o')
plt.plot(fixed_point_widths, qat_accuracies, label='QAT', marker='o')
plt.xlabel('Fixed Point Width')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Fixed Point Width for PTQ and QAT')
plt.legend()
plt.grid(True)
plt.savefig("plots/accuracy_vs_fixed_point_width.png")  # 保存图片
plt.close()
