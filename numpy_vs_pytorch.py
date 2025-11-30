"""
TODO:
- fix up demo of array indexing difference
- fix up demo of nonzero difference
- add summary differences from numpy_vs_pytorch (few remaining)
"""

import numpy as np
import torch


def demo_array_indexing():
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


def demo_nonzero():
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
    print(
        f"PyTorch Meaning:      Used the (2,2) tensor to index ONLY the first dimension!"
    )
    print(
        f"                      It returned {res_pt.shape[0]}x{res_pt.shape[1]} copies of rows."
    )

    # 3. The Fix
    print("\n--- The Fix for PyTorch ---")
    # You must explicitly convert to tuple to match NumPy behavior
    nz_pt_fixed = torch.nonzero(data_pt, as_tuple=True)
    res_pt_fixed = data_pt[nz_pt_fixed]
    print(f"Fixed Result Shape:   {res_pt_fixed.shape} (Matches NumPy)")
