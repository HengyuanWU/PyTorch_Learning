# utils/visualization.py
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates


def plot_training_metrics(metrics, save_path=None):
    """
    根据训练得到的 (loss, accuracy) 列表绘制验证损失和准确率曲线。

    :param metrics: list of tuple, 每个元素为 (validation_loss, validation_accuracy)
    :param save_path: str or None, 如果指定则保存图像到 "{save_path}_loss.png" 和 "{save_path}_acc.png"
    """
    epochs = list(range(1, len(metrics) + 1))
    val_losses = [m[0] for m in metrics]
    val_accs = [m[1] for m in metrics]

    # 验证损失
    plt.figure()
    plt.plot(epochs, val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch")
    plt.xticks(epochs)
    if save_path:
        plt.savefig(f"{save_path}_loss.png")
    plt.show()

    # 验证准确率
    plt.figure()
    plt.plot(epochs, val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy vs Epoch")
    plt.xticks(epochs)
    if save_path:
        plt.savefig(f"{save_path}_acc.png")
    plt.show()


def plot_experiment_comparison(results: List[Dict], metric: str = 'accuracy'):
    """
    绘制不同超参数组合在最后一个 epoch 的指定指标对比。

    :param results: 从 JSON 加载的实验结果
    :param metric: 'accuracy' 或 'loss'
    """
    labels = []
    values = []
    for exp in results:
        cfg = exp['config']
        metrics = exp['metrics']
        # 取最后一个 epoch 的指定指标
        if metric == 'accuracy':
            val = metrics[-1][1]
        else:
            val = metrics[-1][0]
        # 构造标签，仅展示关键超参
        label = f"lr={cfg['lr']}, mom={cfg['momentum']}, bs={cfg['batch_size']}, wd={cfg.get('weight_decay', 0)}, pat={cfg.get('patience', None)}"
        labels.append(label)
        values.append(val)
    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha='right')
    plt.ylabel(metric.title())
    plt.title(f'Final {metric.title()} Comparison')
    plt.tight_layout()
    plt.show()

def plot_hyperparam_parallel(results: list[dict]):
    """
    用平行坐标图展示各超参组合与最终 accuracy 的关系。

    :param results: load_experiment_results() 返回的列表，每项包含
                    {'config': {...}, 'metrics': [(loss, acc), ...]}
    """
    # 构造 DataFrame
    rows = []
    for exp in results:
        cfg = exp['config'].copy()
        final_acc = exp['metrics'][-1][1]
        cfg['accuracy'] = final_acc
        rows.append(cfg)
    df = pd.DataFrame(rows)

    # 平行坐标图
    plt.figure()
    parallel_coordinates(df, class_column='accuracy', cols=list(df.columns), alpha=0.7)
    plt.title('Hyperparameter vs Accuracy (Parallel Coordinates)')
    plt.ylabel('Value / Accuracy')
    plt.xticks(rotation=20)
    plt.legend([],[])  # 去掉图例过多的标签
    plt.tight_layout()
    plt.show()