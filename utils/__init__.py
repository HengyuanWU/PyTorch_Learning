# utils/__init__.py
"""
工具模块包含数据加载、可视化、实验管理等功能。
"""

# 避免在初始化时导入所有模块，防止循环导入问题
# 使用时请显式导入所需功能

# 例如：
# from .dataloader import get_dataloader, find_project_root
# from .text_dataloader import get_ag_news_dataloader
# from .visualization import plot_experiment_comparison
# from .trainer import fit
# from .experiment import run_experiment, load_experiment_results

# 导出PROJECT_ROOT变量
from .dataloader import find_project_root, get_dataloader

PROJECT_ROOT = find_project_root()