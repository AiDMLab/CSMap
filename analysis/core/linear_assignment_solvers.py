import numpy as np
import pandas as pd
import random
import time
from ortools.graph import pywrapgraph
from common.common import normalize_data, matrix_correlation_pearson, matrix_correlation_spearman
from scipy.spatial import distance


def import_solver(solver_method):
    try:
        if solver_method == "lapjv_compat":
            from lap import lapjv
            solver = lapjv
        elif solver_method == "lapjv":
            from lapjv import lapjv
            solver = lapjv
        else:
            raise NotImplementedError(f"The solver {solver_method} is not a supported solver "
                                      "for the shortest augmenting path method, choose between "
                                      "'lapjv' and 'lapjv_compat'.")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The Python package containing the solver_method option "
                                  f"you have chosen {solver_method} was not found. If you "
                                  "selected 'lapjv_compat' solver, install package 'lap'"
                                  "by running 'pip install lap==0.4.0'. If you selected 'lapjv'"
                                  "solver, install package 'lapjv' by running `pip install lapjv==1.3.14'"
                                  "or check the package home page for further instructions.")

    return solver


def call_solver(solver, solver_method, cost_scaled):
    if solver_method == "lapjv_compat":
        _, _, y = solver(cost_scaled)
    elif solver_method == "lapjv":
        _, y, _ = solver(cost_scaled)

    return y

def calculate_cost(expressions_tpm_scRNA_log, expressions_tpm_st_log, cell_number_to_node_assignment,
                    solver_method, distance_metric, sc_cell_types=None, st_spot_cell_type_counts=None, lambda_1=None):
    """
    新增参数：
    - sc_cell_types: 单细胞的细胞类型列表（1D np.array），长度=scRNA细胞数
    - st_spot_cell_types: ST spot的细胞类型列表（1D np.array），长度=ST spot数
    - lambda_1: 细胞类型不匹配的惩罚系数
    """
    print("Building cost matrix ...")
    t0 = time.perf_counter()
    # 原成本矩阵计算逻辑

    # ===================== 1. 原始成本矩阵计算（保持不变） =====================
    if solver_method == "lap_CSPR":
        if distance_metric == "Pearson_correlation":
            cost = -np.transpose(matrix_correlation_pearson(expressions_tpm_st_log, expressions_tpm_scRNA_log))
        elif distance_metric == "Spearman_correlation":
            cost = -np.transpose(matrix_correlation_spearman(expressions_tpm_st_log, expressions_tpm_scRNA_log))
        elif distance_metric == "Euclidean":
            cost = np.transpose(distance.cdist(np.transpose(expressions_tpm_scRNA_log), 
                                               np.transpose(expressions_tpm_st_log), 'euclidean'))
    else:
        if distance_metric == "Pearson_correlation":
            cost = -matrix_correlation_pearson(expressions_tpm_scRNA_log, expressions_tpm_st_log)
        elif distance_metric == "Spearman_correlation":
            cost = -matrix_correlation_spearman(expressions_tpm_scRNA_log, expressions_tpm_st_log)
        elif distance_metric == "Euclidean":
            cost = np.transpose(distance.cdist(np.transpose(expressions_tpm_scRNA_log), 
                                               np.transpose(expressions_tpm_st_log), 'euclidean'))
    # ===================== 新增：打印原始成本矩阵关键信息 =====================
    print("="*50 + " 原成本矩阵信息 " + "="*50)
    print(f"原成本矩阵维度 (Spot数 × scRNA细胞数): {cost.shape}")
    print(f"原成本矩阵最小值: {np.min(cost):.8f}")
    print(f"原成本矩阵最大值: {np.max(cost):.8f}")
    print(f"原成本矩阵平均值: {np.mean(cost):.8f}")
    print(f"原成本矩阵总和: {int(np.sum(cost))}")
    # 可选：打印前5行×前5列的具体值（直观查看）
    print("原成本矩阵前5×5具体值：")
    print(cost[:5, :5])
    cost_a = np.abs(cost)
    print(f"原成本矩阵最小值: {np.min(cost_a):.8f}")
    print(f"原成本矩阵最大值: {np.max(cost_a):.8f}")
    print(f"原成本矩阵平均值: {np.mean(cost_a):.8f}")
    print(f"原成本矩阵总和: {int(np.sum(cost_a))}")
    print("="*110)
    # === 自适应λ：None=均值数量级×0.1，否则用手动值 ===
    mean_cost = np.mean(cost)
    # 计算均值的数量级
    abs_mean = abs(mean_cost)
    if abs_mean == 0:
        magnitude = 1.0
    else:
        # 取数量级：10 的整数次幂
        magnitude = 10 ** np.floor(np.log10(abs_mean))

    if lambda_1 is None:
        # λ = 均值绝对值的数量级 * 0.1
        lambda_1 = magnitude * 0.1
        print(f"【自适应λ】成本均值={mean_cost:.8f}，均值绝对值数量级={magnitude:.8f} → λ={lambda_1:.8f}")
    else:
        print(f"【手动λ】使用传入值 λ={lambda_1:.8f}")
    # ===================== 2. 重构核心：拆分Spot为亚位点（按细胞类型数量） =====================
    # 初始化输出变量
    cost_expanded = None
    location_repeat = None
    P = None  # 惩罚矩阵
    
    # 检查输入有效性
    if sc_cell_types is not None and st_spot_cell_type_counts is not None:
        # 2.1 获取细胞类型标签（从DataFrame列名）
        cell_type_labels = st_spot_cell_type_counts.columns.tolist()
        
        # 2.2 对每个Spot，按类型数量生成亚位点类型列表
        sub_spot_types = []  # 所有亚位点的类型
        sub_spot_spot_ids = []  # 亚位点对应的原始Spot索引（数字）
        # 遍历SpotID（而非数字索引）
        for spot_idx, (spot_id, row) in enumerate(st_spot_cell_type_counts.iterrows()):
            # 获取当前Spot的各类型数量
            type_counts = row.values
            # 对每个类型，按数量重复生成亚位点
            for type_idx, type_label in enumerate(cell_type_labels):
                count = int(type_counts[type_idx])
                if count > 0:
                    # 生成count个该类型的亚位点
                    sub_spot_types.extend([f"TYPE_{type_label}"] * count)
                    # 记录亚位点对应的原始Spot索引（数字）
                    sub_spot_spot_ids.extend([spot_idx] * count)
        
        # 转换为numpy数组（格式与sc_cell_types统一）
        sub_spot_types = np.array(sub_spot_types)
        sub_spot_spot_ids = np.array(sub_spot_spot_ids, dtype=int)
        
        # 2.3 验证亚位点总数与总细胞数一致
        total_sub_spots = len(sub_spot_types)
        total_cells = np.sum(cell_number_to_node_assignment)
        if total_sub_spots != total_cells:
            raise ValueError(f"亚位点总数({total_sub_spots})与总细胞数({total_cells})不匹配！请检查st_spot_cell_type_counts的数值")
        
        # 2.4 扩展原始成本矩阵到亚位点级（每个亚位点复用对应Spot的成本）
        # cost维度：[Spot数, 单细胞数] → 扩展为 [亚位点数, 单细胞数]
        cost_expanded = cost[sub_spot_spot_ids, :]
        # ===================== 新增：打印扩展后（未加惩罚）的成本矩阵 =====================
        print("="*50 + " 扩展后（未加惩罚）成本矩阵信息 " + "="*50)
        print(f"扩展后成本矩阵维度 (亚位点数 × scRNA细胞数): {cost_expanded.shape}")
        print(f"扩展后成本矩阵最小值: {np.min(cost_expanded):.8f}")
        print(f"扩展后成本矩阵最大值: {np.max(cost_expanded):.8f}")
        print(f"扩展后成本矩阵平均值: {np.mean(cost_expanded):.8f}")
        print(f"扩展后成本矩阵总和: {int(np.sum(cost_expanded))}")
        cost_abs = np.abs(cost_expanded)  # 先计算所有元素的绝对值
        print(f"扩展后成本矩阵绝对值最小值: {np.min(cost_abs):.8f}")
        print(f"扩展后成本矩阵绝对值最大值: {np.max(cost_abs):.8f}")
        print(f"扩展后成本矩阵绝对值之和: {int(np.sum(cost_abs))}")  # 转int避免小数过长
        print(f"扩展后成本矩阵绝对值平均值: {np.mean(cost_abs):.8f}")
        print("="*110)
        # 2.5 构建细胞类型惩罚矩阵（向量化计算，无循环）
        # P维度：[亚位点数, 单细胞数] → P[i,j]=1（亚位点i与单细胞j类型不匹配），=0（匹配）
        P = (sub_spot_types[:, np.newaxis] != sc_cell_types[np.newaxis, :]).astype(int)
        cost_expanded = cost_expanded.astype(np.float32)# 转 float32（确保维度/类型匹配）
        P = P.astype(np.float32)
        # 2.6 融合惩罚项到成本矩阵
        cost_expanded = cost_expanded + lambda_1 * P
        # ===================== 新增：打印融合惩罚后的成本矩阵 =====================
        print("="*50 + " 融合惩罚后成本矩阵信息 " + "="*50)
        print(f"融合惩罚后成本矩阵维度: {cost_expanded.shape}")
        print(f"融合惩罚后成本矩阵最小值: {np.min(cost_expanded):.4f}")
        print(f"融合惩罚后成本矩阵最大值: {np.max(cost_expanded):.4f}")
        print(f"融合惩罚后成本矩阵平均值: {np.mean(cost_expanded):.4f}")
        print(f"融合惩罚后成本矩阵总和: {int(np.sum(cost_expanded))}")
        print("="*110)
        # 2.7 生成location_repeat（亚位点对应的原始Spot ID，兼容原有逻辑）
        location_repeat = sub_spot_spot_ids
        
        print(f"成功拆分Spot为亚位点：总亚位点数={total_sub_spots}")
        print(f"亚位点类型分布：{np.unique(sub_spot_types, return_counts=True)}")
    else:
        # 无细胞类型数据时，使用原始扩展逻辑（兼容旧版本）
        print("Warning: sc_cell_types or st_spot_cell_type_counts is None，使用原始Spot扩展逻辑")
        location_repeat = np.repeat(np.arange(len(cell_number_to_node_assignment)), cell_number_to_node_assignment)
        location_repeat = location_repeat.astype(int)
        cost_expanded = cost[location_repeat, :]
    
    # ===================== 3. 输出日志 =====================
    print(f"Time to build cost matrix: {round(time.perf_counter() - t0, 2)} seconds")
    total_penalty = lambda_1 * np.sum(P) if P is not None else 0
    print(f"Lambda_1 = {lambda_1}, 惩罚矩阵总惩罚值: {int(total_penalty)}")
    print(f"扩展后成本矩阵维度: {cost_expanded.shape}")
    
    return cost_expanded, location_repeat


def match_solution(cost):
    rows = len(cost)
    cols = len(cost[0])
    assignment_mat = np.zeros((rows, 2))
    assignment = pywrapgraph.LinearSumAssignment()
    for worker in range(rows):
        for task in range(cols):
            if cost[worker][task]:
                assignment.AddArcWithCost(worker, task, cost[worker][task])

    solve_status = assignment.Solve()
    if solve_status == assignment.OPTIMAL:
        print('Total cost = ', assignment.OptimalCost())
        print()
        for i in range(0, assignment.NumNodes()):
            assignment_mat[i, 0] = assignment.RightMate(i)
            assignment_mat[i, 1] = assignment.AssignmentCost(i)
    elif solve_status == assignment.INFEASIBLE:
        print('No assignment is possible.')
    elif solve_status == assignment.POSSIBLE_OVERFLOW:
        print('Some input costs are too large and may cause an integer overflow.')
    else:
        raise ValueError("The assignment failed")

    return assignment_mat
