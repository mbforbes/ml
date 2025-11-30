"""gemini -- almost there?"""

import numpy as np
import torch

# Setup Data
data_np = np.zeros((3, 3))
data_pt = torch.zeros(3, 3)
# Set a few values for nonzero test
data_np[0, 0] = 1
data_np[2, 2] = 1
data_pt[0, 0] = 1
data_pt[2, 2] = 1

# Setup Indices (Two 1D arrays)
idx_np = [np.array([0, 2]), np.array([0, 1])]
idx_pt = [torch.tensor([0, 2]), torch.tensor([0, 1])]

print("=== Difference 1: Indexing with a List `[...]` ===")
# NumPy: Interprets list as a single fancy index for the *first* dimension.
# Logic: It stacks the list into a (2, 2) array and indexes dim 0.
# Result Shape: (2, 2) from indices + (3,) remaining dim = (2, 2, 3)
res_np = data_np[idx_np]
print(f"NumPy  `data[[i, j]]` Shape: {res_np.shape}  (Fancy Indexing on Dim 0)")
print(res_np)

# PyTorch: Interprets list as a tuple (one index per dimension).
# Logic: It uses idx[0] for dim 0 and idx[1] for dim 1.
# Result Shape: (2,) because it selects 2 specific elements.
res_pt = data_pt[idx_pt]
print(f"PyTorch `data[[i, j]]` Shape: {res_pt.shape}       (Tuple-style Indexing)")
print(res_pt)


print("\n=== Difference 2: The `nonzero()` Return Type ===")
# NumPy: Returns a TUPLE of arrays (one array per dimension).
# Format: (array([r1, r2]), array([c1, c2]))
nz_np = np.nonzero(data_np)
print(f"NumPy  nonzero type: {type(nz_np)} (Tuple of arrays)")
print(
    f"       Can be used directly as index? {type(data_np[nz_np]) is np.ndarray} (Yes)"
)

# PyTorch: Returns a single 2D TENSOR.
# Format: tensor([[r1, c1], [r2, c2]])
nz_pt = torch.nonzero(data_pt)
print(f"PyTorch nonzero shape: {nz_pt.shape}   (N, Dimensions)")
try:
    # This often fails or behaves unexpectedly if you try to use it directly as an index
    # without unbinding or using as_tuple=True
    _ = data_pt[nz_pt]
except Exception as e:
    print(
        f"       Can be used directly as index? No (needs .unbind() or as_tuple=True)"
    )
