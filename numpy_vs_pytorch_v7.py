import numpy as np
import torch

# Create a 2D array/tensor
data_np = np.arange(6).reshape(2, 3)  # Shape (2, 3)
data_torch = torch.arange(6).reshape(2, 3)  # Shape (2, 3)

# --- The "None" Indexing Distinction ---

# NumPy: None can be used without the index being a tuple.
# It is treated as inserting a new axis at that position.
result_np_none = data_np[None]
print(f"NumPy: data_np[None] Shape: {result_np_none.shape}")

# PyTorch: Indexing with a bare None is disallowed. It must be inside a tuple or list.
try:
    result_torch_none = data_torch[None]
    print("PyTorch: data_torch[None] succeeded (Unexpected)")
except Exception as e:
    print(f"\n**PyTorch Difference Found:**")
    print(f"PyTorch: data_torch[None] Error: {e}")

# The PyTorch equivalent must be a tuple:
result_torch_tuple_none = data_torch[None, :]
print(f"PyTorch: data_torch[None, :] Shape: {result_torch_tuple_none.shape}")
