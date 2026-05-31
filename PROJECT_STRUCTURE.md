# DeepView — 项目结构与函数索引

> 生成日期：2026-05-29  
> 用途：为重构提供代码地图，包含关键函数与可复用组件索引

---

## 项目概述

DeepView 是一个基于 PySide6 的时序数据标注 GUI，支持 **Human-in-the-Loop** 工作流：

1. 导入原始传感器 CSV 数据
2. 无监督训练（AutoEncoder / SimCLR）提取特征
3. t-SNE 降维 + 交互式散点图辅助人工标注
4. 有监督训练（DeepConvLSTM 分类器）
5. 评估与迭代

---

## 目录结构

```
DeepView/
├── main.py                                     # 入口：启动 GUI（launch_dview）
├── environment.yaml
├── deepview/
│   ├── __init__.py                             # 包导出（顶层 API）
│   ├── model_cfg.yaml                          # 默认模型超参数
│   │
│   ├── gui/                                    # GUI 层（PySide6）
│   │   ├── window.py                           # MainWindow 主窗口
│   │   ├── components.py                       # DefaultTab / DefaultWebTab 基类
│   │   ├── widgets.py                          # 通用 Widget 组件
│   │   ├── dview_params.py                     # 参数配置 UI
│   │   ├── assets/                             # 图标、样式表
│   │   ├── static/                             # Web 资源（HTML/CSS/JS）
│   │   │   ├── html/                           # 交互图表 HTML 模板
│   │   │   ├── css/
│   │   │   └── js/                             # 图表交互逻辑（ECharts 等）
│   │   │
│   │   ├── tabs/                               # 各功能 Tab
│   │   │   ├── create_project.py               # ProjectCreator 对话框
│   │   │   ├── open_project.py                 # OpenProject 对话框
│   │   │   ├── create_training_dataset.py      # 构建训练集
│   │   │   ├── train_network.py                # 无监督训练 Tab
│   │   │   ├── label_with_interactive_plot.py  # 交互标注 Tab（入口）
│   │   │   ├── supervised_learning_new_labels.py # 有监督训练 Tab
│   │   │   ├── supervised_cl.py                # 监督对比学习 Tab
│   │   │   └── evaluate_network.py             # 评估 Tab
│   │   │
│   │   ├── label_with_interactive_plot/        # 交互标注核心模块
│   │   │   ├── __init__.py                     # LabelWithInteractivePlot 主类
│   │   │   ├── utils.py                        # 特征提取、数据加载工具
│   │   │   └── styles.py                       # 组件样式
│   │   │
│   │   ├── supervised_cl/                      # 监督对比学习 UI 模块
│   │   │   ├── __init__.py                     # SupervisedClWidget
│   │   │   ├── ui/                             # UI 子组件
│   │   │   ├── train/                          # 训练辅助
│   │   │   └── utils.py
│   │   │
│   │   └── plot/                               # 绘图工具
│   │
│   ├── utils/                                  # 通用工具层
│   │   ├── auxiliaryfunctions.py               # 配置读写、路径管理（最核心工具文件）
│   │   ├── auxfun_files.py                     # 文件 I/O（fileReader 类）
│   │   ├── conversioncode.py                   # 数据格式转换
│   │   └── __init__.py
│   │
│   ├── create_project/                         # 项目初始化
│   │   ├── new.py                              # create_new_project()
│   │   ├── load_videos.py                      # 数据文件加载
│   │   └── __init__.py
│   │
│   ├── generate_training_dataset/              # 训练集构建
│   │   ├── trainingsetmanipulation.py          # merge_annotateddatasets、read_process_csv 等
│   │   ├── utils.py                            # 数据分割、重采样工具
│   │   └── __init__.py
│   │
│   ├── clustering_pytorch/                     # 无监督学习（聚类/对比学习）
│   │   ├── training.py                         # train_network() 主函数
│   │   ├── config.py                           # 配置类
│   │   ├── default_config.py
│   │   ├── visualizemaps.py
│   │   ├── core/
│   │   │   ├── train.py                        # 训练循环
│   │   │   └── evaluate.py                     # 评估
│   │   ├── nnet/                               # 神经网络组件
│   │   │   ├── models.py                       # AutoEncoder、SimCLR 架构
│   │   │   ├── train_utils.py                  # AE_eval_time_series、simclr_eval_time_series
│   │   │   ├── losses.py                       # 对比损失函数
│   │   │   ├── base.py
│   │   │   ├── factory.py                      # get_model() 工厂函数
│   │   │   └── augment.py                      # 数据增强
│   │   └── datasets/
│   │       ├── factory.py                      # prepare_unsup_dataset()
│   │       ├── utils.py
│   │       └── __init__.py
│   │
│   ├── supv_learning_pytorch/                  # 有监督学习（分类）
│   │   ├── sup_training.py                     # train_sup_network() 主函数
│   │   ├── core/
│   │   │   ├── trainer.py                      # 通用 Trainer
│   │   │   └── supv_trainer.py                 # 有监督 Trainer
│   │   ├── models/
│   │   │   └── dcl.py                          # DeepConvLstmV3 模型
│   │   ├── preprocess/
│   │   │   ├── preprocess_logbot.py
│   │   │   └── process_utils.py
│   │   ├── utils/
│   │   ├── config/
│   │   │   ├── config_dl.yaml
│   │   │   └── path.yaml
│   │   └── __init__.py
│   │
│   ├── mad_labeling/                           # MAD 框架接口（占位/扩展点）
│   │   ├── myalgorithm.py                      # BaseAlgorithm 实现
│   │   ├── myimporter.py                       # BaseImporter 实现
│   │   ├── myexporter.py                       # BaseExporter 实现
│   │   └── __init__.py
│   │
│   └── calculate_results/                      # 研究结果分析（实验脚本）
│       ├── models/
│       ├── GUIcode/
│       └── data/                               # 各数据集实验代码
│
├── test-data/                                  # 测试数据集
├── conda-env/                                  # Conda 环境配置
│   ├── deepview.yaml
│   └── deepview_m1.yaml
└── .vscode/
    └── launch.json
```

---

## 数据流

```
raw-data/*.csv
    ↓  [CreateTrainingDataset]  分段 + 重采样
unsupervised-datasets/*.pkl
    ↓  [TrainNetwork]  AutoEncoder / SimCLR 训练
unsupervised-model/  (模型权重)
    ↓  [LabelWithInteractivePlot]  特征提取 + t-SNE + 人工标注
db/database.db  (SQLite 标注)
    ↓  [SupervisedLearning]  DeepConvLSTM 训练
supervised-model/  (分类器权重)
    ↓  [EvaluateNetwork]  F1 / Accuracy
```

---

## 关键函数索引

### 入口与主窗口

| 函数 / 方法 | 文件 | 说明 |
|---|---|---|
| `launch_dview()` | [main.py](main.py) | 启动 GUI，创建 QApplication 和 MainWindow |
| `MainWindow.__init__()` | [deepview/gui/window.py](deepview/gui/window.py) | 主窗口初始化，注册所有 Tab |
| `MainWindow.add_tabs()` | [deepview/gui/window.py](deepview/gui/window.py) | 动态添加工作流 Tab |
| `MainWindow._update_project_state()` | [deepview/gui/window.py](deepview/gui/window.py) | 项目加载后刷新 UI 状态 |
| `MainWindow.load_settings()` | [deepview/gui/window.py](deepview/gui/window.py) | 加载最近使用的项目 |

### 项目管理

| 函数 / 方法 | 文件 | 说明 |
|---|---|---|
| `create_new_project()` | [deepview/create_project/new.py](deepview/create_project/new.py) | 创建项目目录结构和 config.yaml |
| `read_config(path)` | [deepview/utils/auxiliaryfunctions.py](deepview/utils/auxiliaryfunctions.py) | 读取 YAML 配置文件 |
| `write_config(path, cfg)` | [deepview/utils/auxiliaryfunctions.py](deepview/utils/auxiliaryfunctions.py) | 写入 YAML 配置文件 |
| `create_config_template()` | [deepview/utils/auxiliaryfunctions.py](deepview/utils/auxiliaryfunctions.py) | 生成默认配置模板 |

### 路径管理（全部在 auxiliaryfunctions.py）

| 函数 | 返回路径 |
|---|---|
| `get_unsup_model_folder(cfg)` | `project/unsupervised-model/…` |
| `get_sup_model_folder(cfg)` | `project/supervised-model/…` |
| `get_unsupervised_set_folder()` | `project/unsupervised-datasets/` |
| `get_raw_data_folder()` | `project/raw-data/` |
| `get_labeled_data_folder(cfg)` | `project/labeled-data/` |
| `get_db_folder()` | `project/db/` |
| `grab_files_in_folder(folder, ext)` | 列出文件夹内指定扩展名文件 |
| `grab_files_in_folder_deep(folder, ext)` | 递归列出文件 |
| `attempt_to_make_folder(name)` | 递归创建文件夹 |

### 无监督训练

| 函数 / 方法 | 文件 | 说明 |
|---|---|---|
| `train_network(config)` | [deepview/clustering_pytorch/training.py](deepview/clustering_pytorch/training.py) | 无监督训练主函数（AE / SimCLR） |
| `get_model(p_backbone, p_setup, num_channel)` | [deepview/clustering_pytorch/nnet/factory.py](deepview/clustering_pytorch/nnet/factory.py) | 模型工厂：实例化 AutoEncoder 或 SimCLR |
| `prepare_unsup_dataset(cfg, files)` | [deepview/clustering_pytorch/datasets/factory.py](deepview/clustering_pytorch/datasets/factory.py) | 构建无监督 DataLoader |
| `AE_eval_time_series(dataloader, model, device)` | [deepview/clustering_pytorch/nnet/train_utils.py](deepview/clustering_pytorch/nnet/train_utils.py) | AutoEncoder 推理，返回特征向量列表 |
| `simclr_eval_time_series(dataloader, model, device)` | [deepview/clustering_pytorch/nnet/train_utils.py](deepview/clustering_pytorch/nnet/train_utils.py) | SimCLR 推理，返回特征向量列表 |

### 交互标注（Human-in-the-Loop 核心）

| 函数 / 方法 | 文件 | 说明 |
|---|---|---|
| `LabelWithInteractivePlot.__init__()` | [deepview/gui/label_with_interactive_plot/__init__.py](deepview/gui/label_with_interactive_plot/__init__.py) | 标注组件：Qt + WebEngine + WebChannel |
| `featureExtraction(root, data, data_length, column_names, model_path, model_name)` | [deepview/gui/label_with_interactive_plot/utils.py](deepview/gui/label_with_interactive_plot/utils.py) | 加载预训练模型，提取特征，t-SNE 降维 |
| `get_data_from_pkl(filename, cfg)` | [deepview/gui/label_with_interactive_plot/utils.py](deepview/gui/label_with_interactive_plot/utils.py) | 从 pkl 文件加载预处理传感器数据 |
| `split_dataframe(data, data_length)` | [deepview/gui/label_with_interactive_plot/utils.py](deepview/gui/label_with_interactive_plot/utils.py) | 将 DataFrame 切分为固定长度窗口 |
| `find_data_columns(sensor_dict, column_names)` | [deepview/gui/label_with_interactive_plot/utils.py](deepview/gui/label_with_interactive_plot/utils.py) | 传感器名称映射到 DataFrame 列名 |

### 有监督训练

| 函数 / 方法 | 文件 | 说明 |
|---|---|---|
| `train_sup_network(config)` | [deepview/supv_learning_pytorch/sup_training.py](deepview/supv_learning_pytorch/sup_training.py) | 有监督训练主函数（DeepConvLSTM） |
| `DeepConvLstmV3` | [deepview/supv_learning_pytorch/models/dcl.py](deepview/supv_learning_pytorch/models/dcl.py) | 主分类模型（CNN + LSTM） |

### 训练集管理

| 函数 / 方法 | 文件 | 说明 |
|---|---|---|
| `merge_annotateddatasets()` | [deepview/generate_training_dataset/trainingsetmanipulation.py](deepview/generate_training_dataset/trainingsetmanipulation.py) | 合并多个标注数据集 |
| `read_process_csv()` | [deepview/generate_training_dataset/trainingsetmanipulation.py](deepview/generate_training_dataset/trainingsetmanipulation.py) | 读取并预处理原始 CSV |
| `divide_df_if_timestamp_gap_detected_2(df, gap_min_limit)` | [deepview/generate_training_dataset/utils.py](deepview/generate_training_dataset/utils.py) | 检测时间戳跳变并分割 DataFrame |
| `run_resampling_and_concat_df(df_list, resample_rate)` | [deepview/generate_training_dataset/utils.py](deepview/generate_training_dataset/utils.py) | 重采样并拼接多个 DataFrame |

---

## 可复用组件（重构优先关注）

### GUI 通用组件 — [deepview/gui/widgets.py](deepview/gui/widgets.py)

| 组件 | 说明 | 复用场景 |
|---|---|---|
| `StreamWriter` / `StreamReceiver` | 线程安全 stdout 重定向到 QTextEdit | 所有有后台任务的 Tab |
| `ClickableLabel` | 可点击的带 hover 效果文字标签 | 任意需要链接样式的标签 |
| `DragDropListView` | 带复选框的拖放列表 | 文件选择、通道选择 |
| `ConfigEditor` | YAML 配置文件编辑器组件 | 任意配置编辑界面 |

### Tab 基类 — [deepview/gui/components.py](deepview/gui/components.py)

| 类 | 关键机制 | 重构建议 |
|---|---|---|
| `DefaultTab` | 懒初始化（`firstShowEvent`）、自动布局 | 所有 Tab 必须继承此类 |
| `DefaultWebTab` | QWebEngineView + QWebChannel 双向通信 | 凡需要交互式图表的 Tab |

### 数据处理工具 — [deepview/generate_training_dataset/utils.py](deepview/generate_training_dataset/utils.py)

```python
GRAVITATIONAL_ACCELERATION = 9.80665   # IMU 数据归一化常量
GYROSCOPE_SCALE = 10                   # 陀螺仪缩放因子
label_str2num: dict                    # 活动名称 → 类别 ID 映射字典
```

### 模型工厂 — [deepview/clustering_pytorch/nnet/factory.py](deepview/clustering_pytorch/nnet/factory.py)

`get_model(p_backbone, p_setup, num_channel)` 是唯一需要调用的模型实例化入口，支持：
- `p_setup = "autoencoder"` → AE 系列
- `p_setup = "simclr"` → SimCLR 系列

### 文件读取 — [deepview/utils/auxfun_files.py](deepview/utils/auxfun_files.py)

`fileReader` 类：封装多格式传感器数据读取（CSV、日志格式）

### 扩展点（插件接口） — [deepview/mad_labeling/](deepview/mad_labeling/)

```python
BaseAlgorithm   # myalgorithm.py：实现自定义标注算法
BaseImporter    # myimporter.py：实现自定义数据导入格式
BaseExporter    # myexporter.py：实现自定义数据导出格式
```

---

## 关键依赖关系

```
main.py
  └─ gui/window.py (MainWindow)
       ├─ gui/components.py (DefaultTab, DefaultWebTab)
       │    └─ gui/tabs/* (各功能 Tab)
       ├─ gui/widgets.py (通用 Widget)
       └─ utils/auxiliaryfunctions.py (配置 I/O)

gui/tabs/train_network.py
  └─ clustering_pytorch/training.py
       ├─ nnet/factory.py (get_model)
       ├─ datasets/factory.py (prepare_unsup_dataset)
       └─ generate_training_dataset/trainingsetmanipulation.py

gui/tabs/label_with_interactive_plot.py
  └─ gui/label_with_interactive_plot/__init__.py
       ├─ label_with_interactive_plot/utils.py
       │    └─ clustering_pytorch/nnet/train_utils.py
       └─ utils/auxiliaryfunctions.py

supv_learning_pytorch/sup_training.py
  ├─ supv_learning_pytorch/models/dcl.py
  ├─ supv_learning_pytorch/core/trainer.py
  └─ generate_training_dataset/trainingsetmanipulation.py
```

---

## 项目数据目录结构（运行时生成）

```
{project_name}-{date}/
├── config.yaml                         # 项目配置
├── raw-data/                           # 原始传感器 CSV
├── labeled-data/                       # 人工标注 pkl
├── unsupervised-datasets/              # 分段特征 pkl
├── supervised-datasets/                # 训练/测试集
├── unsupervised-model/                 # 无监督模型权重
│   ├── model_cfg.yaml
│   ├── train/
│   └── test/
├── supervised-model/                   # 有监督模型权重
│   ├── config.yaml
│   ├── train/
│   └── test/
└── db/
    └── database.db                     # SQLite 标注数据库
```

---

## 外部依赖

| 类别 | 库 |
|---|---|
| GUI | PySide6, qdarkstyle, pyqtgraph |
| 深度学习 | PyTorch, timm |
| 数据处理 | pandas, numpy, scipy, scikit-learn |
| 可视化 | matplotlib |
| 配置 | ruamel.yaml |
| 数据库 | SQLite3（标准库） |
