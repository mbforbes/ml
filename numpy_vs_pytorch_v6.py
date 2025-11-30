import numpy as np
import torch

# 1. Create 3D data array/tensor (Shape: 2, 3, 4)
# Think of this as (Batch, Height, Width)
data_np = np.arange(24).reshape(2, 3, 4)
data_torch = torch.arange(24).reshape(2, 3, 4)

# 2. Create a 1D index array (Shape: 3,)
# We want to select the elements at indices 0, 1, 2 along the *second* axis (Height).
index_array = np.array([0, 1, 2])
index_tensor = torch.tensor([0, 1, 2])

print("\n--- The Ellipsis/Advanced Indexing Distinction ---")

# Indexing: Select the elements at indices 0, 1, 2 along the SECOND axis (Height)
# using the Ellipsis to select all of the first and last axes.
# data[Axis 0: All, Axis 1: Selected, Axis 2: All]
# Expected Output should be (2, 3, 4) after the slice, and then (2, 3, 4) for NumPy
# but PyTorch will behave differently due to its stricter broadcasting rules here.

# NumPy: Uses advanced indexing rules.
result_np = data_np[:, index_array, ...]
print(f"NumPy Indexing: data[:, index_array, ...]")
print(f"  Result Array Shape: {result_np.shape}")
print(f"  Result Array:\n{result_np[0, 0]}")  # Just print a slice

# PyTorch: The index array broadcasts against the dimensions *not* explicitly indexed.
# PyTorch treats the index array as a new dimension being added before the ellipsis
# to the index tuple. This is the difference.
result_torch = data_torch[:, index_tensor, ...]
print(f"\nPyTorch Indexing: data[:, index_tensor, ...]")
print(f"  Result Tensor Shape: {result_torch.shape}")
print(f"  Result Tensor:\n{result_torch[0, 0]}")
