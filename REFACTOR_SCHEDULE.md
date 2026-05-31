# DeepView 重构日程表

> 制定日期：2026-05-31
> 范围：全仓（GUI + 训练/数据层），**排除 `deepview/calculate_results/`（一次性实验脚本，不重构）**
> 节奏：全职冲刺，连续工作日，起始 2026-06-01（周一）
> 配套地图：见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 0. 重构原则（贯穿始终）

1. **行为零变更**：重构只移动/拆分/重命名，不改逻辑、不改 GUI 行为。任何"顺手优化"另开任务。
2. **小步提交**：每个子任务一个 commit，可单独回滚。禁止"一次拆完再提交"。
3. **先建安全网，再动刀**：没有可重复的回归验证手段之前，不碰 god-file。
4. **自底向上拆 god-file**：先抽离不依赖主类的独立小类，最后才动主类本体。
5. **Skill 护栏强制收口**：每个子任务结束必须跑完下方"Skill 约束循环"才算完成。

---

## 1. 代码现状诊断（实测）

| 指标 | 数值 | 说明 |
|---|---|---|
| 头号 god-file | `gui/label_with_interactive_plot/__init__.py` **3514 行 / 13 个类** | 主类 `LabelWithInteractivePlot` 一个类 ~70 方法、~2200 行 |
| GUI 其他大文件 | `window.py` 586 / `tabs/train_network.py` 538 / `supervised_cl/ui/new_scatter_map.py` 505 | |
| 训练/数据层大文件 | `supv_learning_pytorch/preprocess/process_utils.py` 716 / `generate_training_dataset/trainingsetmanipulation.py` 598 / `supv_learning_pytorch/utils/utils.py` 566 | |
| `print()` 调试残留 | **385 处**（非 calculate_results） | 应迁移到 logging |
| TODO/FIXME/HACK | **411 处** | 分类处理：删除/转 issue/就地修 |

> `calculate_results/` 是研究实验脚本，**不在本次重构范围**（一次性脚本，重构收益低、风险高）。

---

## 2. Skill 约束循环（每个子任务的强制收口）

每个子任务按此顺序执行，缺一不可：

```
① 改动        — 仅移动/抽取/重命名，保持行为不变
② /run        — 启动 GUI，手动走一遍受影响的工作流，确认无回归
③ /simplify   — 对本次 diff 做复用/简化/效率/层级清理并应用
④ /code-review — 对本次 diff 扫正确性 bug（拆分时易漏导入、漏 self 绑定、循环引用）
⑤ git commit  — 单任务单提交，commit message 写清"抽取了什么/从哪到哪"
```

约束要点：
- **②必须先于③④**：先证明没改坏，再谈清理和评审。
- **③④只看本次 diff**，不扩大战场。
- 拆分阶段最常见 bug：移动类后忘了搬 import、跨模块循环引用、`self.xxx` 引用断裂、信号/槽连接丢失 —— `/code-review` 重点盯这些。

---

## 3. 任务分解 + 日程表（全职 · 共 15 个工作日）

### Phase 0 — 安全网（Day 1，06-01）

| 子任务 | 产出 |
|---|---|
| 0.1 建重构分支 `refactor/decompose`，从当前分支切出 | 干净分支 |
| 0.2 编写"烟雾回归清单"：启动 GUI→建/开项目→无监督训练→交互标注→有监督训练→评估，逐步截图记录基线行为 | `docs/refactor-smoke-test.md` |
| 0.3 跑通 `/run` 一次，确认能启动、记录启动命令与依赖 | 基线确认 |

> 安全网建不起来就不进 Phase 1。Day 1 只做这件事。

---

### Phase 1 — 拆解头号 god-file（Day 2–7，06-02 ~ 06-07）

目标包结构（自底向上抽取，每步独立提交 + 走 Skill 循环）：

```
label_with_interactive_plot/
├── __init__.py          # 仅导出 LabelWithInteractivePlot
├── main_widget.py       # 主类本体
├── widgets/
│   ├── time_selector.py # ClickableLabel, TimeSelectorWidget, DateTimeSelector
│   └── combo.py         # ReComboBox, LabelOption
├── video.py             # VideoProcessor, VideoEditor
├── workers.py           # TaskSignals, SaveCsvTask, HandleComputeWorker, find_nearest_index
├── backend.py           # Backend, BackendMap（WebChannel 桥接）
├── utils.py             # 既有
└── styles.py            # 既有
```

| Day | 子任务 | 抽取内容 | 风险 |
|---|---|---|---|
| Day 2 (06-02) | 1.1 抽离纯展示小组件 | `ClickableLabel` / `TimeSelectorWidget` / `DateTimeSelector` → `widgets/time_selector.py` | 低（基本不依赖主类） |
| Day 2 (06-02) | 1.2 抽离下拉/标签组件 | `ReComboBox` / `LabelOption` → `widgets/combo.py` | 低 |
| Day 3 (06-03) | 1.3 抽离视频处理 | `VideoProcessor`(QThread) / `VideoEditor`(QDialog) → `video.py` | 中（线程信号） |
| Day 4 (06-04) | 1.4 抽离后台任务 | `TaskSignals` / `SaveCsvTask` / `HandleComputeWorker` / `find_nearest_index` → `workers.py` | 中（QRunnable/信号槽） |
| Day 5 (06-05) | 1.5 抽离 WebChannel 后端 | `Backend` / `BackendMap` → `backend.py` | 高（与 JS 双向通信，回归要重点测散点图/地图联动） |
| Day 6–7 (06-06~07) | 1.6 主类瘦身 | `LabelWithInteractivePlot` → `main_widget.py`，再按职责切分助手模块：`_layout`（init/create* 布局方法）、`_plotting`（左/中/右图与 region）、`_compute`（handleCompute*）、`_labeling`（save/label）、`_video`（播放控制） | 高（70 方法相互引用，建议用 mixin 或委托类，逐组迁移 + 每组跑 Skill 循环） |

> Phase 1 是整个重构的核心，占 6 天。1.6 若发现耦合过深，宁可保留主类但拆出无状态助手函数，不强求一次到位。

---

### Phase 2 — GUI 其他大文件（Day 8–10，06-08 ~ 06-10）

| Day | 子任务 | 文件 | 方向 |
|---|---|---|---|
| Day 8 (06-08) | 2.1 拆 `window.py`(586) | `gui/window.py` | 分离 Tab 注册逻辑、设置加载、菜单/状态栏 |
| Day 9 (06-09) | 2.2 拆 `train_network.py`(538) | `gui/tabs/train_network.py` | 分离参数 UI 构建 与 训练任务调度 |
| Day 10 (06-10) | 2.3 拆 `new_scatter_map.py`(505) + `supervised_cl/__init__.py`(339) | `gui/supervised_cl/*` | 散点图组件与数据装配分离 |

---

### Phase 3 — 训练/数据层（Day 11–12，06-11 ~ 06-12）

| Day | 子任务 | 文件 | 方向 |
|---|---|---|---|
| Day 11 (06-11) | 3.1 拆 `process_utils.py`(716) | `supv_learning_pytorch/preprocess/process_utils.py` | 按预处理步骤分函数文件 |
| Day 11 (06-11) | 3.2 拆 `supv_learning_pytorch/utils/utils.py`(566) | 同上 | 按职责（度量/IO/张量工具）分组 |
| Day 12 (06-12) | 3.3 拆 `trainingsetmanipulation.py`(598) | `generate_training_dataset/` | `merge_annotateddatasets` / `read_process_csv` 等拆为独立模块 |

> 训练层重构后必须用一个小数据集**实跑一轮无监督+有监督训练**确认数值无变化（不只是 `/run` 启动）。

---

### Phase 4 — 全仓清理（Day 13–14，06-13 ~ 06-14）

| Day | 子任务 | 内容 |
|---|---|---|
| Day 13 (06-13) | 4.1 `print()` → `logging` | 385 处分批迁移；建立统一 logger 配置；调试用的临时 print 直接删 |
| Day 14 (06-14) | 4.2 TODO/FIXME 分类清理 | 411 处过一遍：能立即修的修、过时的删、需跟进的转成 issue/任务清单 |
| Day 14 (06-14) | 4.3 复用组件落位 | 把 `widgets.py` 的 `StreamWriter`/`ConfigEditor`/`DragDropListView` 等确认被各 Tab 复用，消除重复实现 |

---

### Phase 5 — 收尾验收（Day 15，06-15）

| 子任务 | 内容 |
|---|---|
| 5.1 全量烟雾回归 | 按 0.2 清单逐步对照基线截图，确认行为一致 |
| 5.2 `/code-review` 全分支扫一遍 | 对整个 `refactor/decompose` vs `master` 的 diff |
| 5.3 更新 `PROJECT_STRUCTURE.md` | 反映新的包结构与文件索引 |
| 5.4 整理 commit、准备合并 PR | PR 描述列出拆分前后对照 |

---

## 4. 里程碑视图

```
Day 1        Phase 0  安全网 ███
Day 2–7      Phase 1  god-file 拆解（核心）██████████████████
Day 8–10     Phase 2  GUI 其他大文件 █████████
Day 11–12    Phase 3  训练/数据层 ██████
Day 13–14    Phase 4  print→logging / TODO / 复用清理 ██████
Day 15       Phase 5  收尾验收 ███
```

预计 **3 周（15 个工作日）** 完成全仓重构。Phase 1 是重心与最大风险点。

---

## 5. 风险与回滚

| 风险 | 缓解 |
|---|---|
| WebChannel(Backend) 拆分后 JS↔Py 联动失效 | 1.5 单列一天，回归重点测散点图点击高亮、地图联动 |
| 主类 mixin 拆分引入隐式状态依赖 | 1.6 逐组迁移，每组跑完 Skill 循环再继续；耦合过深则降级为助手函数 |
| 训练层重构改变数值结果 | Phase 3 强制小数据集实跑对比，不只启动 |
| 小步提交被打断丢上下文 | 每个子任务独立 commit，分支可任意回退到上一个绿色提交 |

---

## 6. 每日开工/收工固定动作

- **开工**：`git status` 确认在 `refactor/decompose`；明确今天的子任务编号。
- **每个子任务收工**：跑完 §2 的 Skill 约束循环五步，绿了才 commit。
- **每日收工**：push 分支；在本文件对应 Day 行打勾标记完成。