import numpy as np

default_records = {
    'repeat': 0,
    'actual_num_evals': np.array([]),
    'time': np.array([]),
    'GFLOP/s': np.array([]),
    'cumulative_execution_time': np.array([]),
    'loss': np.array([]),
    'noise': np.array([]),
    'mean_actual_num_evals': 0,
    'mean_GFLOP/s': 0,
    'mean_time': 0,
    'err_actual_num_evals': 0,
    'err_GFLOP/s': 0,
    'err_time': 0,
    'mean_cumulative_strategy_time': 0,
    'mean_cumulative_compile_time': 0,
    'mean_cumulative_execution_time': 0,
    'mean_cumulative_total_time': 0,
}

computable_records = {
    'mean_cumulative_execution_time': lambda records: np.mean(record['cumulative_execution_time'] for record in records)
}
