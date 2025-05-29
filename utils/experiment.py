import json
from typing import List, Dict
import matplotlib.pyplot as plt


def save_experiment_results(results: List[Dict], filepath: str):
    """
    保存实验结果到 JSON 文件。

    :param results: 实验结果列表，每个元素为 {"config": dict, "metrics": list[tuple(float, float)]}
    :param filepath: JSON 文件保存路径，推荐以 .json 结尾
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_experiment_results(filepath: str) -> List[Dict]:
    """
    从 JSON 文件加载实验结果。

    :param filepath: JSON 文件路径
    :return: 实验结果列表
    """
    with open(filepath, 'r') as f:
        return json.load(f)
