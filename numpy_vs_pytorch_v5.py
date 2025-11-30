"""after several false starts in a single response, another gemini attempt

this one at least gets a warning

"""

import numpy as np
import torch

# 1. Create a 2D base array/tensor (Shape: 3x3)
data_np = np.arange(9).reshape(3, 3)
data_torch = torch.arange(9).reshape(3, 3)

# 2. Create index tensors
idx_0 = np.array([0, 2])
idx_1 = np.array([0, 1])
idx_t_0 = torch.tensor([0, 2])
idx_t_1 = torch.tensor([0, 1])

# --- NumPy Difference ---
# NumPy interprets a list of arrays (not in a tuple) as indexing one axis.
# This is known as "multi-dimensional indexing," and it expects the indices to
# match the size of the axis (which they don't here).
# It tries to find the elements at indices [0, 2] and then [0, 1] on that axis.
try:
    result_np_list = data_np[[idx_0, idx_1]]
    print(f"NumPy Result [idx_0, idx_1] Shape: {result_np_list.shape}")
except ValueError as e:
    # NumPy raises a ValueError because the inner arrays (the indices)
    # are not the same length (2 and 2 are the same length, so this should pass).
    # If the lengths were different, it would fail. Let's try the successful one:
    print("VALUE ERROR")
    result_np_list = data_np[[0, 2]]  # This is list indexing
    print(f"NumPy Result [[0, 2]] Shape: {result_np_list.shape}")  # (2, 3)

# --- PyTorch Difference ---
# PyTorch always interprets a list of tensors as indexing multiple axes,
# equivalent to passing a tuple.
# data_torch[[idx_t_0, idx_t_1]] is equivalent to data_torch[idx_t_0, idx_t_1]
result_torch_list = data_torch[[idx_t_0, idx_t_1]]
print(f"PyTorch Result [[idx_t_0, idx_t_1]] Shape: {result_torch_list.shape}")  # (2,)

# THE DIFFERENCE:
result_np_tuple = data_np[
    (idx_0, idx_1)
]  # This is NumPy's equivalent of PyTorch's list
print(f"NumPy Result (idx_0, idx_1) Shape: {result_np_tuple.shape}")  # (2,)

# If PyTorch interprets the list of indices as a tuple of indices, and NumPy
# interprets the list of indices as an *array* of indices for a single axis:

print("\n--- The Final Semantic Difference (Tuple vs. List Indexing) ---")

# NumPy: A list of integers/arrays is treated as indexing a *single* axis.
result_np_list = data_np[[0, 1]]  # data_np is (3, 3). This selects rows 0 and 1.
print(f"NumPy: data_np[[0, 1]] Shape: {result_np_list.shape}")  # (2, 3)

# PyTorch: A list of integers/tensors is treated as indexing *multiple* axes (a tuple).
result_torch_list = data_torch[
    [0, 1]
]  # This selects the element at (0, 1) (which is 1)
print(
    f"PyTorch: data_torch[[0, 1]] Shape: {result_torch_list.shape}"
)  # torch.Size([]) (A 0-rank scalar)
