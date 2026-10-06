# CSMap：小论文代码与结果复现

本仓库对应小论文最终图1—5，使用论文实际采用的原始稠密全局分配版 CSMap。方法将每个空间点位的容量展开为带细胞类型标签的槽位，通过表达成本与有限的类型不匹配惩罚共同决定分配。

完整安装、输入格式、数据来源、指标定义和复现边界见 [英文 README](README.md)。

## 快速运行

在仓库根目录执行，推荐 Python 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/demo.py
python analysis/check_benchmarks.py
python analysis/test_original_quota_fix.py
```

示例生成 24 个细胞、8 个点位的输入，并检查每个点位恰好接收 3 个细胞。基准检查从保存的映射文件重新计算三种设置、四种方法的 60 项指标。

## 重绘最终结果图

```bash
python analysis/plot_figure2.py
python analysis/plot_figure3.py
python analysis/plot_figure4.py
python analysis/plot_figure5.py
```

结果保存在 `results/figures/`，包括 PDF、PNG 和 SVG。绘图所需的精简结果保存在 `data/figure2/` 至 `data/figure5/`。图一为手工编辑的最终 PPT 示意图，没有对应的最终 Python 绘图入口，因此未放入旧版示意图代码。

重绘使用保存结果，不等于重新运行完整生物数据实验。图2的 SMLE 保留原始历史误差表；从现存映射计算的点位中心距离不能直接替换该口径。图5展示 D02T1、D18T2，完整表保留含 D24T2 在内的 33 项细胞类型与切片组合结果。

## 完整分析入口

- `run_mapping.py`：在新数据上运行 CSMap；输入范例由 `demo.py` 生成。
- `run_figure4_improved.py`：惩罚强度和先验扰动。
- `run_figure5_robustness.py`：沿用历史文件名，实际对应图4F—H的容量、抽样种子与基因池实验；资源测量使用 macOS 的 `/usr/bin/time -l`。
- `download_gastric_validation_data.py`、`prepare_gastric_biology.py`、`run_gastric.py`：胃癌公开数据下载、预处理和成对映射。
- `paired_analysis.py`：MOB 成对效应及胃癌留出标记相关性评价。

大规模表达矩阵需要单独准备或下载，具体路径见英文 README。完整生物数据流程未在本次打包过程中全部重新运行；本次验证包括最小映射、指标核对、配额测试及最终图2—5重绘。

仓库未包含毕业论文、投稿稿件、专利材料、旧版图稿、无关组织实验和稀疏求解器试验代码。保留了与最终分析有关的上游代码归属和许可文件；具体使用条件见 [第三方说明](THIRD_PARTY_NOTICES.md)。

通讯作者：艾冬梅，aidongmei@ustb.edu.cn。
