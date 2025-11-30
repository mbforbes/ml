"""Claude whiffing it again"""

import numpy as np
import torch

print("Advanced indexing edge case attempt:")
print("=" * 60)

# Multiple array indices
np_arr = np.arange(12).reshape(3, 4)
pt_arr = torch.arange(12).reshape(3, 4)

# This works in NumPy - using two integer arrays to index
row_idx = np.array([0, 1, 2])
col_idx = np.array([1, 2, 3])
np_result = np_arr[row_idx, col_idx]
print(f"NumPy with two index arrays: {np_result}")

pt_row_idx = torch.tensor([0, 1, 2])
pt_col_idx = torch.tensor([1, 2, 3])
pt_result = pt_arr[pt_row_idx, pt_col_idx]
print(f"PyTorch with two index arrays: {pt_result}")

print("\nThey're the same! Let me try another...")

# Broadcasting indices with boolean mask
np_arr = np.arange(24).reshape(4, 6)
pt_arr = torch.arange(24).reshape(4, 6)

# Mix of boolean and integer indexing
bool_mask = np.array([True, False, True, False])
int_idx = np.array([1, 2])

try:
    np_result = np_arr[bool_mask][:, int_idx]
    print(f"\nNumPy bool then int indexing: shape {np_result.shape}")
    print(np_result)
except Exception as e:
    print(f"NumPy error: {e}")

try:
    pt_bool_mask = torch.tensor([True, False, True, False])
    pt_int_idx = torch.tensor([1, 2])
    pt_result = pt_arr[pt_bool_mask][:, pt_int_idx]
    print(f"PyTorch bool then int indexing: shape {pt_result.shape}")
    print(pt_result)
except Exception as e:
    print(f"PyTorch error: {e}")
