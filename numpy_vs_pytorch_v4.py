import numpy as np
import torch

# 1. Create the 1D base data array/tensor
data_np = np.arange(6)  # [0, 1, 2, 3, 4, 5]
data_torch = torch.arange(6)

# 2. Create a 2D index array/tensor
# This index array has shape (2, 3)
indices_np = np.array([[0, 1, 2], [3, 4, 5]])
indices_torch = torch.tensor([[0, 1, 2], [3, 4, 5]])

# 3. Perform the advanced indexing
result_np = data_np[indices_np]
result_torch = data_torch[indices_torch]

print("--- NumPy Result ---")
print(f"NumPy Input Array Shape: {data_np.shape}")  # (6,)
print(f"NumPy Index Array Shape: {indices_np.shape}")  # (2, 3)
print(f"NumPy Result Array:\n{result_np}")
print(f"NumPy Result Shape: {result_np.shape}")  # (2, 3)

print("\n--- PyTorch Result ---")
print(f"PyTorch Input Tensor Shape: {data_torch.shape}")  # torch.Size([6])
print(f"PyTorch Index Tensor Shape: {indices_torch.shape}")  # torch.Size([2, 3])
print(f"PyTorch Result Tensor:\n{result_torch}")
print(f"PyTorch Result Shape: {result_torch.shape}")  # torch.Size([2, 3])

# --- CRITICAL DIFFERENCE ---
# Now, let's try indexing into a 2D array using *two* 1D index arrays.
# This demonstrates the difference in broadcasting indices themselves.

# 1. Create a 2D base array/tensor
data_2d_np = np.arange(9).reshape(3, 3)
data_2d_torch = torch.arange(9).reshape(3, 3)

# 2. Create 1D index arrays for rows and columns
rows = np.array([0, 2])  # Shape (2,)
cols = np.array([0, 1])  # Shape (2,)
rows_t = torch.tensor([0, 2])
cols_t = torch.tensor([0, 1])

# 3. Indexing with two index arrays (Advanced Indexing)
result_2d_np = data_2d_np[
    rows[:, None], cols[None, :]
]  # Broadcasting (2, 1) and (1, 2) to (2, 2)
result_2d_torch = data_2d_torch[rows_t.unsqueeze(1), cols_t.unsqueeze(0)]

print("\n--- Broadcasting Difference (NumPy vs PyTorch) ---")
print(f"NumPy Indexing Result (Broadcasted):")
print(f"  Result Array:\n{result_2d_np}")
print(f"  Result Shape: {result_2d_np.shape}")  # (2, 2)

print(f"\nPyTorch Indexing Result (Broadcasted):")
print(f"  Result Tensor:\n{result_2d_torch}")
print(f"  Result Shape: {result_2d_torch.shape}")  # torch.Size([2, 2])
