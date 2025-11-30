"""gemini: demo of the nonzero difference"""

import numpy as np
import torch

# Setup Data (3x3)
data_np = np.zeros((3, 3))
data_pt = torch.zeros(3, 3)
data_np[0, 0] = 1
data_np[2, 2] = 1
data_pt[0, 0] = 1
data_pt[2, 2] = 1

print("=== The 'Nonzero' Silent Failure ===")

# 1. Get Nonzero Indices
nz_np = np.nonzero(data_np)  # Tuple of 2 arrays: (rows, cols)
nz_pt = torch.nonzero(data_pt)  # 2D Tensor: [[0,0], [2,2]]

print(f"NumPy indices shape/type:   {type(nz_np)} of length {len(nz_np)}")
print(f"PyTorch indices shape/type: {nz_pt.shape} ({nz_pt.type()})")

# 2. Try to index directly
res_np = data_np[nz_np]
res_pt = data_pt[nz_pt]

print("\n--- The Results (CRITICAL DIFFERENCE) ---")
print(f"NumPy Result Shape:   {res_np.shape}")
print(
    f"NumPy Meaning:        Selected {res_np.size} specific elements (0,0) and (2,2)."
)

print(f"PyTorch Result Shape: {res_pt.shape}")
print(f"PyTorch Meaning:      Used the (2,2) tensor to index ONLY the first dimension!")
print(
    f"                      It returned {res_pt.shape[0]}x{res_pt.shape[1]} copies of rows."
)

# 3. The Fix
print("\n--- The Fix for PyTorch ---")
# You must explicitly convert to tuple to match NumPy behavior
nz_pt_fixed = torch.nonzero(data_pt, as_tuple=True)
res_pt_fixed = data_pt[nz_pt_fixed]
print(f"Fixed Result Shape:   {res_pt_fixed.shape} (Matches NumPy)")
