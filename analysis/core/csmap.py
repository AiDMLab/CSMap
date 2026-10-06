
import pandas as pd
import numpy as np
from pandas.core.common import flatten

import random
import time

import concurrent.futures
import os

from common.common import read_file, read_visium, normalize_data, downsample, check_paths, argument_parser, estimate_cell_type_fractions
from post_processing import save_results, plot_results
from linear_assignment_solvers import (calculate_cost, match_solution, import_solver,
                                                 call_solver)

def read_data(scRNA_path, cell_type_path, cell_type_fraction_estimation_path, n_cells_per_spot_path, 
                st_cell_type_path, output_path, output_prefix, spaceranger_path=None, st_path=None, coordinates_path=None):
    if spaceranger_path is not None:
        st_data, coordinates_data = read_visium(spaceranger_path,output_path)
    elif (st_path is None) and (coordinates_path is None):
        raise ValueError("For ST data, you must provide either a tar.gz file or paths for expression and coordinates.")
    else:
        # Read data
        st_data = read_file(st_path)
        coordinates_data = read_file(coordinates_path)

    st_data = st_data[~st_data.index.duplicated(keep=False)]
    if (st_cell_type_path is None) and (cell_type_fraction_estimation_path is None):
        print('Estimating cell type fractions')
        if (spaceranger_path is not None) or (st_path.endswith('.mtx')):
            st_data_outpath = os.path.join(output_path, f"{output_prefix}ST_expression.txt")
            st_data.to_csv(st_data_outpath, sep='\t')
            cell_type_fraction_estimation_path = estimate_cell_type_fractions(scRNA_path, cell_type_path, st_data_outpath, output_path, output_prefix)
        else:
            cell_type_fraction_estimation_path = estimate_cell_type_fractions(scRNA_path, cell_type_path, st_path, output_path, output_prefix)

    st_data.columns = ['SPOT_'+str(col) for col in st_data.columns]
    st_data.index = ['GENE_'+str(idx) for idx in st_data.index]

    coordinates_data.index = ['SPOT_'+str(idx) for idx in coordinates_data.index]

    scRNA_data = read_file(scRNA_path)
    scRNA_data.columns = ['CELL_'+str(col) for col in scRNA_data.columns]
    scRNA_data.index = ['GENE_'+str(idx) for idx in scRNA_data.index]

    scRNA_data = scRNA_data[~scRNA_data.index.duplicated(keep=False)]

    cell_type_data = read_file(cell_type_path)
    cell_type_data.index = ['CELL_'+str(idx) for idx in cell_type_data.index]
    cell_type_data.iloc[:,0] = ['TYPE_'+str(cell) for cell in list(cell_type_data.iloc[:,0])]

    #st_data = st_data.loc[(st_data!=0).any(1), (st_data!=0).any(0)]
    #scRNA_data = scRNA_data.loc[(scRNA_data!=0).any(1), (scRNA_data!=0).any(0)]

    if st_cell_type_path is not None:
        st_cell_type_data = read_file(st_cell_type_path)
        st_cell_type_data.index = ['SPOT_'+str(idx) for idx in st_cell_type_data.index]
        st_cell_type_data.iloc[:,0] = ['TYPE_'+str(cell) for cell in list(st_cell_type_data.iloc[:,0])]
    else:
        st_cell_type_data = None
    if cell_type_fraction_estimation_path is not None:
        cell_type_fraction_data = read_file(cell_type_fraction_estimation_path)
        cell_type_fraction_data.columns = ['TYPE_'+str(col) for col in cell_type_fraction_data.columns]
    else:
        cell_type_fraction_data = None
    if n_cells_per_spot_path is not None:
        n_cells_per_spot_data = read_file(n_cells_per_spot_path)
        n_cells_per_spot_data.index = ['SPOT_'+str(idx) for idx in n_cells_per_spot_data.index]
    else:
        n_cells_per_spot_data = None

    # Order data to match
    try:
        st_data = st_data[coordinates_data.index]
        scRNA_data = scRNA_data[cell_type_data.index]
        if st_cell_type_data is not None:
            st_cell_type_data = st_cell_type_data.loc[coordinates_data.index, :]
        if n_cells_per_spot_data is not None:
            n_cells_per_spot_data = n_cells_per_spot_data.transpose(copy=False)
            n_cells_per_spot_data = n_cells_per_spot_data[coordinates_data.index]
            n_cells_per_spot_data = n_cells_per_spot_data.transpose(copy=False)
    except Exception as e:
        raise IndexError(f"The ST data: {st_path} and coordinates data: {coordinates_path} have to "
                         "have the same spot IDs for columns and rows, respectively, "
                         f"and scRNA data: {scRNA_path} and cell type data: {cell_type_path} have"
                         " to have the same cell IDs for columns and rows, respectively.")

    # Validate input
    if (st_data.columns != coordinates_data.index).any():
        raise IndexError(f"The ST data: {st_path} and coordinates data: {coordinates_path} have to "
                         "have the same spot IDs for columns and rows, respectively.")

    if (scRNA_data.columns != cell_type_data.index).any():
        raise IndexError(f"The scRNA data: {scRNA_path} and cell type data: {cell_type_path} have"
                         " to have the same cell IDs for columns and rows, respectively.")
    
    if (st_cell_type_data is not None) and (st_cell_type_data.index != coordinates_data.index).any():
        raise IndexError(f"The ST cell type data: {st_cell_type_path} and coordinates data: {coordinates_path} have to "
                         "have the same spot IDs for rows.")

    if (st_cell_type_data is None) and (cell_type_fraction_data is None):
        raise ValueError("At least one of st_cell_type_path and cell_type_fraction_estimation_path should be specified."
                         "For --single-cell, st_cell_type_path is recommended; if not --single-cell, cell_type_fraction_estimation_path is required.")
    
    if (st_cell_type_data is not None) and (cell_type_fraction_data is not None):
        print("Warning: st_cell_type_path and cell_type_fraction_estimation_path are both specified.")
        print("If --single-cell, cell_type_fraction_estimation_path will be ignored in this case.")

    return scRNA_data, cell_type_data, st_data, coordinates_data, cell_type_fraction_data, n_cells_per_spot_data, st_cell_type_data


def estimate_cell_number_RNA_reads(st_data, mean_cell_numbers):
    # Read data
    expressions = st_data.values.astype(float)

    # Data normalization
    expressions_tpm_log = normalize_data(expressions)

    # Set up fitting problem
    RNA_reads = np.sum(expressions_tpm_log, axis=0, dtype=float)
    mean_RNA_reads = np.mean(RNA_reads)
    min_RNA_reads = np.min(RNA_reads)

    min_cell_numbers = 1 if min_RNA_reads > 0 else 0

    fit_parameters = np.polyfit(np.array([min_RNA_reads, mean_RNA_reads]),
                                np.array([min_cell_numbers, mean_cell_numbers]), 1)
    polynomial = np.poly1d(fit_parameters)
    cell_number_to_node_assignment = polynomial(RNA_reads).astype(int)

    return cell_number_to_node_assignment


def get_cell_type_fraction(number_of_cells, cell_type_fraction_data):
    cell_type_numbers = cell_type_fraction_data.transpose()
    fractions_to_numbers = cell_type_numbers.values * number_of_cells
    fractions_to_numbers = fractions_to_numbers.astype(int).flatten()  # 展平为一维
    # 关键修复1：强制列类型为int，避免浮点型
    cell_type_numbers.iloc[:,0] = fractions_to_numbers
    cell_type_numbers = cell_type_numbers.astype({cell_type_numbers.columns[0]: int})  # 显式指定列类型
    
    # 补充：修正细胞数求和偏差
    total = sum(cell_type_numbers.iloc[:,0])
    if total != number_of_cells:
        max_ct = cell_type_numbers.idxmax().values[0]
        # 关键修复2：确保增量是整数
        cell_type_numbers.loc[max_ct, cell_type_numbers.columns[0]] += int(number_of_cells - total)
    
    # 最终确保返回值全为整数
    cell_type_numbers = cell_type_numbers.astype(int)
    return cell_type_numbers

def partition_indices(indices, split_by_category_list=None, split_by_interval_int=None, shuffle=True):
    """
    Splits the provided indices into list of smaller index sets based on other parameters.
    indices is originally a single 1D numpy array, which is then split and returned as a list of smaller 1D numpy arrays.
    e.g., indices = np.arange(0, 5000) can be split into [np.arange(0, 2000), np.arange(2000, 5000)].

    Parameters :
        indices                (1D np.array(int)) : indices to be split.

        split_by_category_list (1D np.array(int)) : number of indices for each category that cannot be mixed together.
            split_by_category_list should sum to len(indices)
            e.g., if split_by_category_list is [3000, 5000], with indices == 0:8000,
                    then the first 3000 will be partitioned separately from the latter 5000.
                    i.e., [0:2000, 2000:3000, 3000:7000, 7000:8000] is possible, but [0:2000, 2000:7000, 7000:8000] is not.
        
        split_by_interval_int               (int) : max length of each partition.
            if split_by_category_list is None, then indices are split into partitions of this size.
            e.g., if split_by_interval_int is 1000, with indices == 0:2500, then [0:1000, 1000:2000, 2000:2500] is returned.
            if split_by_category_list is specified, then any category that exceeds this size will be further partitioned.
            e.g., if split_by_category_list is [500, 1000, 300] and split_by_interval_int is 400, with indices == 0:1800,
                    then [0:400, 400:500, 500:900, 900:1300, 1300:1500, 1500:1800] is returned.
                    - split_by_category_list sets breakpoints at 500 and 1500
                    - split_by_interval_int sets breakpoints at every 400 inside each of the three groups

        shuffle                            (bool) : whether indices should be shuffled before being split.
    
    Returns :
        List[1D np.array(int)] : Partitioned indices.
    """
    num_indices = len(indices)

    if shuffle:
        np.random.shuffle(indices)

    # initialize breakpoints, as start and end
    breakpoints = [0, num_indices]

    # add breakpoints set by split_by_category_list
    if split_by_category_list is not None:
        if np.sum(split_by_category_list) != num_indices:
            print('Warning: sum of counts in each category does not match the full length')
        breakpoints.extend(np.cumsum(split_by_category_list))
        breakpoints = sorted(np.unique(breakpoints))
    
    # add breakpoints set by split_by_interval_int
    if split_by_interval_int is not None:
        breakpoints_new = breakpoints.copy()
        for idx in range(len(breakpoints)-1):
            if breakpoints[idx+1] - breakpoints[idx] <= split_by_interval_int:
                continue
            breakpoints_new.extend(np.arange(breakpoints[idx], breakpoints[idx+1], split_by_interval_int))
        breakpoints = breakpoints_new
    
    # take unique values, sort in ascending order, and remove start and end indices
    breakpoints = sorted(np.unique(breakpoints))[1:-1]

    # partition the indices based on breakpoints
    split_indices_list = np.array_split(indices, breakpoints)

    return split_indices_list


def sample_single_cells(scRNA_data, cell_type_data, cell_type_numbers_int, sampling_method, seed):
    """
    仅按ST总细胞数量随机采样单细胞，确保采样数 = ST总需求数：
    1. 计算ST所有spot的总细胞需求数
    2. 无放回/有放回采样至总需求数
    3. 保证输出细胞数与ST需求严格一致
    """    
    np.random.seed(seed)
    random.seed(seed)

    # 核心：计算ST总细胞需求数（所有类型求和）
    total_desired_cells = cell_type_numbers_int.iloc[:, 0].sum()
    # 获取所有scRNA细胞的索引（列是细胞）
    all_sc_cell_indices = list(range(scRNA_data.shape[1]))
    total_available_cells = len(all_sc_cell_indices)

    if total_available_cells == 0:
        raise ValueError("No single-cell data available in the scRNA-seq dataset.")

    # 严格采样至总需求数
    if total_available_cells >= total_desired_cells:
        # 数量充足：无放回随机抽取（保证无重复）
        selected_index = random.sample(all_sc_cell_indices, total_desired_cells)
    else:
        # 数量不足：先全选，再重复抽样补充（保证总数匹配）
        selected_index = all_sc_cell_indices.copy()
        remaining = total_desired_cells - total_available_cells
        supplement_index = np.random.choice(all_sc_cell_indices, remaining, replace=True)
        selected_index.extend(supplement_index.tolist())

# ========== 新增核心逻辑：为重复列名添加唯一后缀 ==========
    # 3. 提取采样的表达数据（按索引取）
    sampled_expression = scRNA_data.iloc[:, selected_index].to_numpy()
    # 4. 生成纯数字唯一列名（避免重复）
    unique_cell_names = [f"CELL_{i}" for i in range(total_desired_cells)]
    # =========================================================

    # 提取采样结果（使用唯一列名，确保列名无重复）
    # 关键：记录映射关系（数字列名→原始细胞ID、原始细胞类型）
    original_cell_ids = scRNA_data.columns[selected_index].tolist()  # 原始细胞ID
    original_cell_types = cell_type_data.iloc[selected_index, 0].tolist()  # 原始细胞类型
    # 构建映射字典
    cell_mapping = {
        "numeric_name": unique_cell_names,
        "original_cid": original_cell_ids,
        "original_type": original_cell_types
    }
    all_cells_save = pd.DataFrame(
        sampled_expression,
        index=scRNA_data.index,
        columns=unique_cell_names  # 替换为唯一列名
    )

    # 校验：采样数必须等于总需求数
    if all_cells_save.shape[1] != total_desired_cells:
        raise ValueError(f"采样细胞数({all_cells_save.shape[1]})与ST总需求数({total_desired_cells})不匹配！")

    return all_cells_save, selected_index, cell_mapping



def solve_linear_assignment_problem(scRNA_norm_data, st_norm_data, cell_number_to_node_assignment,
                                    solver_method, solver, seed, distance_metric, 
                                    sc_cell_types=None, cell_type_spot_counts_df=None, lambda_1=None,  # 新增参数
                                    process_idx=None):
    """新增参数：
    - sc_cell_types: 单细胞类型列表
    - st_spot_cell_types: ST spot类型列表
    - lambda_1: 惩罚系数
    """
    # 调用calculate_cost时传递新增参数
    distance_repeat, location_repeat =\
        calculate_cost(scRNA_norm_data, st_norm_data, cell_number_to_node_assignment,
                       solver_method, distance_metric,
                       sc_cell_types=sc_cell_types, st_spot_cell_type_counts=cell_type_spot_counts_df, lambda_1=lambda_1)  # 新增

    # 原求解逻辑不变
    if solver_method == 'lapjv' or solver_method == 'lapjv_compat':
        print('Solving linear assignment problem ...')
        np.random.seed(seed)
        cost_scaled = distance_repeat + 1e-16 * np.random.rand(distance_repeat.shape[0],
                                                               distance_repeat.shape[1])
        t0 = time.perf_counter()
        assignment = call_solver(solver, solver_method, cost_scaled)
        print(f"Time to solve linear assignment problem: {round(time.perf_counter() - t0, 2)} seconds")
        assigned_nodes = location_repeat[assignment]
        mapped_st_index = np.transpose(assigned_nodes).tolist()
        # 新增：校验映射结果长度 == scRNA_norm_data的列数（细胞数）
        if len(mapped_st_index) != scRNA_norm_data.shape[1]:
            raise ValueError(
                f"映射结果长度({len(mapped_st_index)})与scRNA细胞数({scRNA_norm_data.shape[1]})不匹配！"
            )
        return mapped_st_index, process_idx
    elif solver_method == 'lap_CSPR':
        print('Solving linear assignment problem ...')
        np.random.seed(seed)
        cost_scaled = 10**6 * distance_repeat + 10 * np.random.rand(distance_repeat.shape[0],
                                                                    distance_repeat.shape[1]) + 1
        cost_scaled = np.transpose(cost_scaled)
        cost_scaled_int = cost_scaled.astype(int)
        cost_scaled_int_list = cost_scaled_int.tolist()
        t0 = time.perf_counter()
        assignment = match_solution(cost_scaled_int_list)

        print(f"Time to solve linear assignment problem: {round(time.perf_counter() - t0, 2)} seconds")
        assigned_nodes = location_repeat[assignment[:, 0].astype(int)]
        mapped_st_index = assigned_nodes.tolist()
    else:
        raise ValueError("Invalid solver_method provided")

    return mapped_st_index, process_idx


def apply_linear_assignment(scRNA_data, st_data, coordinates_data, cell_number_to_node_assignment,
                                solver_method, solver, seed, distance_metric, number_of_processors,
                                index_sc_list, index_st_list=None, subsampled_cell_number_to_node_assignment_list=None,
                                sc_cell_types=None, cell_type_spot_counts_df=None, lambda_1=None):  # 新增参数
    """新增参数传递细胞类型和惩罚系数"""
    if (index_st_list is not None) and (subsampled_cell_number_to_node_assignment_list is not None):
        raise ValueError("index_st_list and subsampled_cell_number_to_node_assignment_list cannot both be specified")
    
    # 原归一化逻辑不变
    scRNA_norm_np = normalize_data(scRNA_data.to_numpy())
    st_norm_np = normalize_data(st_data.to_numpy())
    scRNA_norm_data = pd.DataFrame(scRNA_norm_np, index=scRNA_data.index, columns=scRNA_data.columns)
    st_norm_data = pd.DataFrame(st_norm_np, index=st_data.index, columns=st_data.columns)

    if (index_st_list is None) and (subsampled_cell_number_to_node_assignment_list is None):
        # 传递新增参数
        mapped_st_index, _ =\
            solve_linear_assignment_problem(
                scRNA_norm_data.iloc[:, index_sc_list[0]].to_numpy(), st_norm_data.to_numpy(), cell_number_to_node_assignment,
                solver_method, solver, seed, distance_metric,
                sc_cell_types=sc_cell_types[index_sc_list[0]] if sc_cell_types is not None else None,  # 新增
                cell_type_spot_counts_df=cell_type_spot_counts_df, lambda_1=lambda_1,  # 新增
                process_idx=None)
        assigned_locations = coordinates_data.iloc[mapped_st_index]
        cell_ids_selected = scRNA_norm_data.columns.values[index_sc_list[0]]
        return assigned_locations, cell_ids_selected

    results = []
    cell_ids_selected_list = []
    assigned_locations_list = []

    if index_st_list is not None:
        num_iters = len(index_st_list)
    elif subsampled_cell_number_to_node_assignment_list is not None:
        num_iters = len(subsampled_cell_number_to_node_assignment_list)
    else:
        raise ValueError("Invalid point")
    print(f"Number of required processors: {num_iters}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_iters, number_of_processors)) as executor:
        for idx in range(num_iters):
            if index_st_list is not None:
                st_norm_data_selected = st_norm_data.iloc[:, index_st_list[idx]]
                cell_number_to_node_assignment_selected = cell_number_to_node_assignment[index_st_list[idx]]
                # 截取对应ST spot的类型
                if cell_type_spot_counts_df is not None:
                    # 截取对应ST spot的类型（按index_st_list[idx]）
                    st_spot_cell_types_selected = cell_type_spot_counts_df.iloc[index_st_list[idx]]
                else:
                    st_spot_cell_types_selected = None
            elif subsampled_cell_number_to_node_assignment_list is not None:
                st_norm_data_selected = st_norm_data
                cell_number_to_node_assignment_selected = subsampled_cell_number_to_node_assignment_list[idx]
                st_spot_cell_types_selected = cell_type_spot_counts_df
            else:
                raise ValueError("Invalid point")

            scRNA_norm_data_selected = scRNA_norm_data.iloc[:, index_sc_list[idx]]
            # 截取对应scRNA细胞的类型
            sc_cell_types_selected = sc_cell_types[index_sc_list[idx]] if sc_cell_types is not None else None

            # 提交任务时传递新增参数
            results.append(executor.submit(
                solve_linear_assignment_problem,
                scRNA_norm_data_selected.to_numpy(), st_norm_data_selected.to_numpy(), cell_number_to_node_assignment_selected,
                solver_method, solver, seed, distance_metric,
                sc_cell_types=sc_cell_types_selected, cell_type_spot_counts_df=cell_type_spot_counts_df, lambda_1=lambda_1,  # 新增
                process_idx=idx
            ))
        
        for f in concurrent.futures.as_completed(results):
            mapped_st_index, process_idx = f.result()
            assigned_locations = coordinates_data.iloc[index_st_list[process_idx]].iloc[mapped_st_index]\
                                    if index_st_list is not None \
                                    else coordinates_data.iloc[mapped_st_index]
            assigned_locations_list.append(assigned_locations)
            cell_ids_selected = scRNA_norm_data.columns.values[index_sc_list[process_idx]]
            cell_ids_selected_list.append(cell_ids_selected)
        
        cell_ids_selected = np.concatenate(cell_ids_selected_list, axis=0)
        assigned_locations = pd.concat(assigned_locations_list)

    return assigned_locations, cell_ids_selected

import numpy as np
import pandas as pd

def filter_genes_by_expression_and_variance(expression_matrix, min_expr=0.1, var_quantile=0.2):
    """
    过滤低表达、低变异基因（返回矩阵+过滤掩码，支持索引对齐）
    :param expression_matrix: 输入表达矩阵 (pd.DataFrame，行=基因，列=细胞/Spot)
    :param min_expr: 基因在至少1%细胞中的表达值阈值
    :param var_quantile: 保留变异度前 (1-var_quantile) 的基因（如0.2=保留前80%）
    :return: 
        filtered_matrix: 过滤后的表达矩阵 (pd.DataFrame)
        gene_mask: 基因过滤掩码（bool数组，True=保留的基因）
    """
    # 转为numpy数组计算（保留原DataFrame索引）
    expr_np = expression_matrix.to_numpy().astype(np.float32)
    cell_count = expr_np.shape[1]
    
    # 步骤1：过滤低表达基因（至少1%细胞有表达）
    expr_mask = (expr_np > min_expr).sum(axis=1) > 0.01 * cell_count  # 行掩码（基因）
    
    # 应用低表达过滤
    expr_filtered_np = expr_np[expr_mask, :]
    expr_filtered_genes = expression_matrix.index[expr_mask]
    
    # 步骤2：过滤低变异基因（保留变异度前80%）
    var = np.var(expr_filtered_np, axis=1)
    var_threshold = np.quantile(var, var_quantile)
    var_mask = var > var_threshold  # 低表达过滤后的基因掩码
    
    # 应用变异度过滤
    final_filtered_np = expr_filtered_np[var_mask, :]
    final_filtered_genes = expr_filtered_genes[var_mask]
    
    # 构建过滤后的DataFrame（保留原索引/列名）
    filtered_matrix = pd.DataFrame(
        final_filtered_np,
        index=final_filtered_genes,
        columns=expression_matrix.columns
    )
    
    # 构建完整的基因掩码（匹配原矩阵行数）
    # 先初始化全False，再将保留的基因设为True
    gene_mask = np.zeros(expression_matrix.shape[0], dtype=bool)
    # 找到保留基因在原索引中的位置
    keep_idx = expression_matrix.index.isin(final_filtered_genes)
    gene_mask[keep_idx] = True
    
    print(f"基因过滤：{expression_matrix.shape[0]} → {filtered_matrix.shape[0]}")
    return filtered_matrix, gene_mask

def main_csmap(scRNA_path, cell_type_path,
                   n_cells_per_spot_path, st_cell_type_path, cell_type_fraction_estimation_path=None,
                   spaceranger_path=None, st_path=None, coordinates_path=None,
                   output_folder="csmap_results", output_prefix="",
                   mean_cell_numbers=15, downsample_off=False, scRNA_max_transcripts_per_cell=1500,
                   solver_method="lapjv", distance_metric="Pearson_correlation", sampling_method="duplicates",
                   single_cell=False, number_of_selected_spots=10000,
                   sampling_sub_spots=False, number_of_selected_sub_spots=10000,
                   number_of_processors=1, seed=1,
                   plot_off=False, geometry="honeycomb", max_num_cells_plot=50000, num_column=3,
                   lambda_1=None): # 新增惩罚参数
    # For timing execution
    start_time = time.perf_counter()

    # Check paths
    output_path = check_paths(output_folder, output_prefix)

    # Record log
    fout_log = os.path.join(output_path, f"{output_prefix}log.txt")
    with open(fout_log, "w") as f:
        f.write("CSMap log file \n\nStart time: "+str(time.asctime( time.localtime(time.time()) ))+"\n")
        f.write("\nINPUT ARGUMENTS\n")
        f.write("scRNA_path: "+str(scRNA_path)+"\n")
        f.write("cell_type_path: "+str(cell_type_path)+"\n")
        f.write("st_path: "+str(st_path)+"\n")
        f.write("coordinates_path: "+str(coordinates_path)+"\n")
        f.write("n_cells_per_spot_path: "+str(n_cells_per_spot_path)+"\n")
        f.write("cell_type_fraction_estimation_path: "+str(cell_type_fraction_estimation_path)+"\n")
        f.write("st_cell_type_path: "+str(st_cell_type_path)+"\n")
        f.write("output_folder: "+str(output_folder)+"\n")

        f.write("mean_cell_numbers: "+str(mean_cell_numbers)+"\n")
        f.write("downsample_off: "+str(downsample_off)+"\n")
        f.write("scRNA_max_transcripts_per_cell: "+str(scRNA_max_transcripts_per_cell)+"\n")
        f.write("plot_off: "+str(plot_off)+"\n")
        f.write("geometry: "+str(geometry)+"\n")
        f.write("output_prefix: "+str(output_prefix)+"\n")
        f.write("seed: "+str(seed)+"\n")
        f.write("solver_method: "+str(solver_method)+"\n")
        f.write("sampling_method: "+str(sampling_method)+"\n")
        f.write("distance_metric: "+str(distance_metric)+"\n")
        f.write("single_cell: "+str(single_cell)+"\n\n")
        f.write("sampling_sub_spots: "+str(sampling_sub_spots)+"\n\n")

            
    if solver_method == "lapjv" or solver_method == "lapjv_compat":
        solver = import_solver(solver_method)
    else:
        solver = None
    
    # Read data
    print("Read and validate data ...")

    t0 = time.perf_counter()
    scRNA_data, cell_type_data, st_data, coordinates_data, cell_type_fractions_data, n_cells_per_spot_data, st_cell_type_data =\
        read_data(scRNA_path, cell_type_path, 
                  cell_type_fraction_estimation_path, n_cells_per_spot_path, st_cell_type_path,
                  output_path, output_prefix, spaceranger_path, st_path, coordinates_path)

    print(f"Time to read and validate data: {round(time.perf_counter() - t0, 2)} seconds")
    with open(fout_log,"a") as f:
        f.write(f"Time to read and validate data: {round(time.perf_counter() - t0, 2)} seconds\n")
 
    # Set seed
    np.random.seed(seed)
    random.seed(seed)

    t0_core = time.perf_counter()

    ### Parse/compute required inputs
    # cell_number_to_node_assignment
    if single_cell:
        cell_number_to_node_assignment = np.ones(st_data.shape[1]).astype(int)
    else:
        if n_cells_per_spot_data is None:
            print('Estimating number of cells in each spot ...')
            cell_number_to_node_assignment = estimate_cell_number_RNA_reads(st_data, mean_cell_numbers)
            print(f"Time to estimate number of cells per spot: {round(time.perf_counter() - t0_core, 2)} seconds")
        else:
            cell_number_to_node_assignment = n_cells_per_spot_data.values[:, 0].astype(int)

    # cell_type_numbers_int
    if single_cell and (st_cell_type_data is not None):
        cell_types = sorted(st_cell_type_data.iloc[:, 0].unique(), key=str.lower)
        cell_counts = st_cell_type_data.iloc[:, 0].value_counts().reindex(cell_types)
        cell_type_numbers_int = pd.DataFrame(cell_counts)
        cell_type_numbers_int.columns = ['Fraction']
    else:
        if cell_type_fractions_data is None:
            raise ValueError("cell_type_fraction_estimation_path must be specified.")
        # Compute number of cells
        number_of_cells = np.sum(cell_number_to_node_assignment)
        cell_type_numbers_int = get_cell_type_fraction(number_of_cells, cell_type_fractions_data)


    ### Sample scRNA_data
    print("Down/up sample of scRNA-seq data according to estimated cell type fractions")
    t0 = time.perf_counter()

    # subset datasets to intersect_genes
    intersect_genes = st_data.index.intersection(scRNA_data.index)
    scRNA_data = scRNA_data.loc[intersect_genes, :]
    st_data = st_data.loc[intersect_genes, :]

    # downsample scRNA_data to equal transcript counts per cell
    # so that the assignment is not dependent on expression level
    if not downsample_off:
        scRNA_data = downsample(scRNA_data, scRNA_max_transcripts_per_cell)

    with open(fout_log, "a") as f:
        f.write("Number of genes used for mapping: "+str(len(intersect_genes))+"\n")
        f.write("Number of spots satisfying input for mapping: "+str(st_data.shape[1])+"\n")
        f.write("Number of cells satisfying input for mapping: "+str(scRNA_data.shape[1])+"\n")

    # sample scRNA_data based on cell type composition
    # cell count in scRNA_data_sampled will be equal to cell count (not spot count) in ST data
    # additionally, the cells in scRNA_data_sampled will be in the order of cell types in cell_type_numbers_int

    scRNA_data_sampled, selected_index, cell_mapping = sample_single_cells(scRNA_data, cell_type_data, cell_type_numbers_int, sampling_method, seed)
    scRNA_data_sampled, gene_mask = filter_genes_by_expression_and_variance(
        scRNA_data_sampled,
        min_expr=0.1,
        var_quantile=0.2
    )

    print(f"Time to down/up sample scRNA-seq data: {round(time.perf_counter() - t0, 2)} seconds")
    ### 新增：提取细胞类型映射（用于成本函数）
    # 1. 提取scRNA细胞类型（采样后）
    sc_cell_types = cell_type_data.iloc[selected_index, :].values.flatten()
    # 2. 提取ST spot细胞类型
    if st_cell_type_data is not None:
        st_spot_cell_types = st_cell_type_data.values.flatten()
    else:
        # 若无ST spot类型，使用细胞类型比例推导（或设为None，关闭软约束）
        st_spot_cell_types = None

    # 1. 定义文件路径（替换为你的实际路径）
    output_folder = "testresult"  # 你的输出目录
    cell_frac_file = os.path.join(output_folder, "Seurat_cellfracs.txt")
    if os.path.exists(cell_frac_file):
        print(f"读取Spot级占比文件：{cell_frac_file}")
        cell_type_fractions_data = pd.read_csv(
            cell_frac_file,
            sep="\t",
            index_col=0  # 第一列作为SpotID索引
        )
    # ========== 核心修复：给占比文件的SpotID添加SPOT_前缀 ==========
        # 提取占比文件的原始SpotID（无SPOT_）
        original_spot_ids = cell_type_fractions_data.index.tolist()
        # 添加SPOT_前缀，和ST数据完全匹配
        new_spot_ids = [f"SPOT_{sid}" for sid in original_spot_ids]
        # 替换占比文件的索引
        cell_type_fractions_data.index = new_spot_ids
    else:
        raise FileNotFoundError(f"未找到占比文件：{cell_frac_file}")
    

 
    ### ========== 新增：提前计算每个Spot的细胞类型数量（内存中） ==========
    print("Calculating cell type counts per Spot (in memory) ...")
    # 1. 获取细胞类型列表（去重排序）
    unique_type_labels = cell_type_fractions_data.columns.tolist()
    n_cell_types = len(unique_type_labels)
    
    # 2. ST数据基本信息
    st_data = st_data.loc[gene_mask, :]
    n_spots = st_data.shape[1]  # ST Spot总数
    st_spot_ids_raw = st_data.columns.tolist()  # ST原始SpotID（带SPOT_）
    # 3. 初始化数量矩阵
    cell_type_spot_counts = np.zeros((n_spots, n_cell_types), dtype=int)

    # 5. 计算每个Spot的细胞类型数量（核心逻辑）
    if single_cell:
        # 单细胞模式：每个Spot对应1个细胞类型
        for spot_idx in range(n_spots):
            spot_type = st_spot_cell_types[spot_idx] if st_spot_cell_types is not None else "Unknown"
            if spot_type in unique_cell_types:
                type_idx = unique_cell_types.index(spot_type)
                cell_type_spot_counts[spot_idx, type_idx] = 1  # 单细胞Spot数量=1
    else:
        # 多细胞模式：按cell_number_to_node_assignment和cell_type_fractions_data计算
        if cell_type_fractions_data is not None:
            # 按比例分配每个Spot的细胞数到各类型
            for spot_idx in range(n_spots):
                # ========== 修复：改用loc访问（基于SpotID）而非iloc ==========
                # 获取当前Spot的ID（从st_data列名中取）
                spot_id = st_spot_ids_raw[spot_idx]  # ST的SpotID（带SPOT_）
                # 基于SpotID获取占比（避免位置索引越界）
                spot_fractions = cell_type_fractions_data.loc[spot_id].values
            
                # 当前Spot的总细胞数
                spot_total_cells = cell_number_to_node_assignment[spot_idx]
                # 按比例计算各类型数量（取整）
                spot_type_counts = np.round(spot_fractions * spot_total_cells).astype(int)
            
                # 修正：确保总数匹配（避免取整误差）
                total = spot_type_counts.sum()
                if total != spot_total_cells:
                    diff = spot_total_cells - total
                    max_idx = np.argmax(spot_type_counts)
                    spot_type_counts[max_idx] += diff
            
                # 赋值到矩阵
                cell_type_spot_counts[spot_idx] = spot_type_counts

    # 6. 转换为DataFrame（便于后续保存和使用）
    cell_type_spot_counts_df = pd.DataFrame(
        cell_type_spot_counts,
        index=st_spot_ids_raw,
        columns=unique_type_labels
    )
    print(f"内存中Spot类型数量矩阵维度：{cell_type_spot_counts_df.shape}")
    print("前5个Spot的细胞类型数量：")
    print(cell_type_spot_counts_df.head())
    ### ========== 新增结束 ==========
    ### Case-specific parts
    if single_cell:
        with open(fout_log, "a") as f:
            f.write("Number of selected spots: "+str(number_of_selected_spots)+"\n\n")
            f.write("Number of processors: "+str(number_of_processors)+"\n\n")
        
        if st_cell_type_data is not None:
            index_sc_list = partition_indices(np.arange(scRNA_data_sampled.shape[1]),
                                                split_by_category_list=cell_type_numbers_int.values.flatten(),
                                                split_by_interval_int=number_of_selected_spots,
                                                shuffle=False)
            index_st_celltype_order = []
            for cell_type in cell_type_numbers_int.index:
                index_st_celltype_order.append(np.where(st_cell_type_data.values[:, 0] == cell_type)[0])
            index_st_celltype_order = np.concatenate(index_st_celltype_order, axis=0)
            index_st_list = partition_indices(index_st_celltype_order,
                                                split_by_category_list=cell_type_numbers_int.values.flatten(),
                                                split_by_interval_int=number_of_selected_spots,
                                                shuffle=False)

            # 调用apply_linear_assignment时传递细胞类型和lambda_1
            assigned_locations, cell_ids_selected =\
                apply_linear_assignment(scRNA_data_sampled, st_data, coordinates_data, cell_number_to_node_assignment,
                                        solver_method, solver, seed, distance_metric, number_of_processors,
                                        index_sc_list, index_st_list=index_st_list,
                                        sc_cell_types=sc_cell_types, cell_type_spot_counts_df=cell_type_spot_counts_df, lambda_1=lambda_1)  # 新增

        else:
            index_sc_list = partition_indices(np.arange(scRNA_data_sampled.shape[1]),
                                                split_by_interval_int=number_of_selected_spots,
                                                shuffle=True)
            index_st_list = partition_indices(np.arange(st_data.shape[1]),
                                                split_by_interval_int=number_of_selected_spots,
                                                shuffle=True)

            assigned_locations, cell_ids_selected =\
                apply_linear_assignment(scRNA_data_sampled, st_data, coordinates_data, cell_number_to_node_assignment,
                                        solver_method, solver, seed, distance_metric, number_of_processors,
                                        index_sc_list, index_st_list=index_st_list,
                                        sc_cell_types=sc_cell_types, cell_type_spot_counts_df=cell_type_spot_counts_df, lambda_1=lambda_1)  # 新增


    else:

        if sampling_sub_spots:
            with open(fout_log, "a") as f:
                f.write("Number of selected subspots: "+str(number_of_selected_sub_spots)+"\n\n")
                f.write("Number of processors: "+str(number_of_processors)+"\n\n")
            
            if number_of_selected_sub_spots > np.sum(cell_number_to_node_assignment):
                number_of_selected_sub_spots = np.sum(cell_number_to_node_assignment)

            index_sc_list = partition_indices(np.arange(scRNA_data_sampled.shape[1]),
                                                split_by_interval_int=number_of_selected_sub_spots,
                                                shuffle=True)
            # 1D np.array; repeat each node index by the number of cells in that node
            cell_number_to_node_assignment_aggregate = np.repeat(range(len(cell_number_to_node_assignment)), cell_number_to_node_assignment)
            subsample_spot_idxs_list = partition_indices(cell_number_to_node_assignment_aggregate,
                                                            split_by_interval_int=number_of_selected_sub_spots,
                                                            shuffle=True)
            # aggregate each set of indices in this list back to the (node index) - (count) format
            cell_number_to_node_assignment_list = [np.bincount(subsample_spot_idxs, minlength=(len(cell_number_to_node_assignment)))
                                                    for subsample_spot_idxs in subsample_spot_idxs_list]
                
            assigned_locations, cell_ids_selected =\
                apply_linear_assignment(scRNA_data_sampled, st_data, coordinates_data, cell_number_to_node_assignment,
                                        solver_method, solver, seed, distance_metric, number_of_processors,
                                        index_sc_list, subsampled_cell_number_to_node_assignment_list=cell_number_to_node_assignment_list,
                                        sc_cell_types=sc_cell_types, cell_type_spot_counts_df=cell_type_spot_counts_df, lambda_1=lambda_1)  # 新增
            
        else:

            index_sc_list = partition_indices(np.arange(scRNA_data_sampled.shape[1]), shuffle=False)
            
            assigned_locations, cell_ids_selected =\
                apply_linear_assignment(scRNA_data_sampled, st_data, coordinates_data, cell_number_to_node_assignment,
                                        solver_method, solver, seed, distance_metric, number_of_processors,
                                        index_sc_list,
                                        sc_cell_types=sc_cell_types, cell_type_spot_counts_df=cell_type_spot_counts_df, lambda_1=lambda_1)  # 新增

            
    print(f"Total time to run CSMap core algorithm: {round(time.perf_counter() - t0_core, 2)} seconds")
    with open(fout_log,"a") as f:
        f.write(f"Time to run CSMap core algorithm: {round(time.perf_counter() - t0_core, 2)} seconds\n")
    # ========== 调试代码：在调用save_results前插入 ==========
    print(f"采样后scRNA矩阵维度（行=基因，列=细胞）: {scRNA_data_sampled.shape}")  # 应输出 (基因数, 11323)
    print(f"cell_ids_selected长度: {len(cell_ids_selected)}")  # 应输出 11323
    print(f"assigned_locations长度: {len(assigned_locations)}")  # 应输出 11323
    print(f"ST数据维度（行=基因，列=Spot）: {st_data.shape}")  # 看列数是否=39411
    print(f"原始scRNA矩阵维度: {scRNA_data.shape}")  # 看列数是否=39411
    # ======================================================
    ### Save results
    print('Saving results ...')
    save_results(output_path, output_prefix, cell_ids_selected, scRNA_data_sampled, assigned_locations,
                 cell_type_data, sampling_method, single_cell, cell_mapping=cell_mapping)

    if not plot_off:
        if single_cell:
            plot_results(output_path, output_prefix, max_num_cells=max_num_cells_plot, single_cell_ST_mode=True)
        else:
            plot_results(output_path, output_prefix, coordinates_data=coordinates_data, geometry=geometry, num_cols=num_column, max_num_cells=max_num_cells_plot)

    print(f"Total execution time: {round(time.perf_counter() - start_time, 2)} seconds")
    with open(fout_log,"a") as f:
        f.write(f"Total execution time: {round(time.perf_counter() - start_time, 2)} seconds")


def run_csmap():
    arguments = argument_parser()
    arguments["lambda_1"] = None  # 硬编码默认自适应
    main_csmap(**arguments)
    
if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    # 把 args 转为字典再传参
    main_csmap(**vars(args))
