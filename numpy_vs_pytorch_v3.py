"""gemini failing to find a difference"""

import numpy as np

# 1. Create a 2D array
A_np = np.array([[1, 2, 3], [4, 5, 6]])

# 2. Slice to create a view with a zero-sized dimension (0 rows)
B_np = A_np[:0, :]  # B_np is a (0, 3) array

print(f"NumPy array A_np shape: {A_np.shape}")  # (2, 3)
print(f"NumPy array B_np shape: {B_np.shape}")  # (0, 3)

# 3. Attempt to add a scalar to the zero-sized array
try:
    result_np = B_np + 1
    print(f"NumPy result shape: {result_np.shape}")
except ValueError as e:
    print(f"\n**NumPy Error on B_np + 1:**\n{e}")
# Output: operands could not be broadcast together with shapes (0,3) (0,)

import torch

# 1. Create a 2D tensor
A_torch = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 2. Slice to create a view with a zero-sized dimension (0 rows)
B_torch = A_torch[:0, :]  # B_torch is a (0, 3) tensor

print(f"PyTorch tensor A_torch shape: {A_torch.shape}")  # torch.Size([2, 3])
print(f"PyTorch tensor B_torch shape: {B_torch.shape}")  # torch.Size([0, 3])

# 3. Attempt to add a scalar to the zero-sized tensor
try:
    result_torch = B_torch + 1
    print(f"\n**PyTorch Result on B_torch + 1:**")
    print(f"PyTorch result tensor: {result_torch}")
    print(f"PyTorch result shape: {result_torch.shape}")  # torch.Size([0, 3])
except RuntimeError as e:
    print(f"PyTorch Error: {e}")
# Output: PyTorch result shape: torch.Size([0, 3])
