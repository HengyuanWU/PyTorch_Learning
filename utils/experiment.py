import json
from typing import List, Dict, Any, TypeVar, cast
import matplotlib.pyplot as plt

T = TypeVar('T')

def save_experiment_results(results: List[Dict], filepath: str):
    """
    保存实验结果到 JSON 文件。

    :param results: 实验结果列表，每个元素为 {"config": dict, "metrics": list[tuple(float, float)]}
    :param filepath: JSON 文件保存路径，推荐以 .json 结尾
    """
    # 将结果中的元组转换为列表，以便JSON序列化
    def convert_tuples(obj):
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_tuples(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tuples(i) for i in obj]
        return obj
    
    serializable_results = convert_tuples(results)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"实验结果已保存到: {filepath}")
    except Exception as e:
        print(f"保存实验结果时出错: {e}")


def load_experiment_results(filepath: str) -> List[Dict]:
    """
    从 JSON 文件加载实验结果。

    :param filepath: JSON 文件路径
    :return: 实验结果列表
    """
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # 将metrics中的列表转回元组
        def convert_metrics_to_tuples(obj: Any) -> Any:
            if isinstance(obj, dict):
                if 'metrics' in obj and isinstance(obj['metrics'], list):
                    obj['metrics'] = [tuple(item) if isinstance(item, list) else item 
                                     for item in obj['metrics']]
                return {k: convert_metrics_to_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_metrics_to_tuples(i) for i in obj]
            return obj
        
        converted_results = convert_metrics_to_tuples(results)
        # 确保返回类型是 List[Dict]
        return cast(List[Dict], converted_results)
    except FileNotFoundError:
        print(f"文件不存在: {filepath}")
        raise
    except json.JSONDecodeError:
        print(f"JSON解析错误: {filepath} 不是有效的JSON文件")
        raise
    except Exception as e:
        print(f"加载实验结果时出错: {e}")
        raise
