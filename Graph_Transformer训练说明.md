# Graph Transformer DTI 训练操作说明

本说明对应脚本：`train_graph_dti.py`  
数据文件：`bindingdb_processed/bindingdb_20k_with_images.csv`（直接复用已有 CSV，使用其中的 SMILES 列）

---

## 1. 架构变更说明

| 维度 | 旧方案（Image CNN/ViT） | 新方案（Graph Transformer） |
|---|---|---|
| 分子输入 | 2D 图片 (PNG) | SMILES → 分子图（原子特征矩阵 + 邻接矩阵） |
| 分子编码器 | CNN / Image Transformer | **Graph Transformer**（Graphormer 风格，含边类型注意力偏置） |
| 蛋白编码器 | Transformer / BiGRU | 不变 |
| 融合方式 | 门控 + 差分 + 乘性交互 | 不变 |
| 损失函数 | SmoothL1 + Ranking | 不变 |

分子图结构包含：
- **原子特征矩阵** `[N, 38]`：原子类型(13) + 度(6) + 电荷(5) + 氢数(5) + 芳香性(1) + 环内(1) + 杂化(7)
- **边类型矩阵** `[N, N]`：0=无键, 1=单键, 2=双键, 3=三键, 4=芳香键, 5=自环
- **度编码**：从邻接矩阵计算，作为位置编码加入 Transformer

---

## 2. 环境准备

```bash
conda activate molca      # 或你的训练环境
pip install rdkit          # 如果尚未安装
```

---

## 3. 训练命令

### 主实验（多 seed）

```bash
python train_graph_dti.py --data_csv bindingdb_processed/bindingdb_20k_with_images.csv --output_dir runs/graph_dti_seed42 --epochs 30 --batch_size 64 --num_workers 12 --prefetch_factor 4 --require_cuda --seed 42

python train_graph_dti.py --data_csv bindingdb_processed/bindingdb_20k_with_images.csv --output_dir runs/graph_dti_seed123 --epochs 30 --batch_size 64 --num_workers 12 --prefetch_factor 4 --require_cuda --seed 123

python train_graph_dti.py --data_csv bindingdb_processed/bindingdb_20k_with_images.csv --output_dir runs/graph_dti_seed3407 --epochs 30 --batch_size 64 --num_workers 12 --prefetch_factor 4 --require_cuda --seed 3407
```

### 消融实验：蛋白 BiGRU

```bash
python train_graph_dti.py --data_csv bindingdb_processed/bindingdb_20k_with_images.csv --output_dir runs/graph_dti_protgru_seed42 --epochs 30 --batch_size 64 --num_workers 12 --prefetch_factor 4 --require_cuda --seed 42 --disable_protein_transformer
```

### 与 DeepPurpose 对比

```bash
python compare_with_deeppurpose.py --graph_ckpt runs/graph_dti_seed42/best_model.pt --data_csv bindingdb_processed/bindingdb_20k_with_images.csv
```

---

## 4. 输出文件

每个 `runs/<name>/` 目录包含：

| 文件 | 说明 |
|---|---|
| `best_model.pt` | 最优 checkpoint（含模型权重 + 训练参数） |
| `train_history.csv` | 逐 epoch 训练/验证指标 |
| `test_metrics.json` | 测试集最终指标（含参数量） |
| `param_count.json` | 模型参数量统计 |
| `split.csv` | 数据划分记录 |
| `split_stats.json` | 划分统计 |

---

## 5. 常见问题

| 问题 | 解决 |
|---|---|
| `ModuleNotFoundError: No module named 'rdkit'` | `pip install rdkit` |
| Windows `WinError 1455` | 添加 `--windows_safe_loader` |
| GPU 内存不足 | 降低 `--batch_size` 或 `--max_atoms` |
| SMILES 解析失败被跳过 | 正常现象，日志会打印跳过数量 |

---

## 6. 部署到 Web 系统

训练完成后，将 `best_model.pt` 路径填入 `main_app.py` 中的 `DEFAULT_GRAPH_DTI_CKPT`，或在网页界面中手动指定权重路径，即可在线预测。
