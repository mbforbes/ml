"""Claude mostly whiffing it

my critique:

(1) += also exists on pytorch, and seems to change in-place, just like numpy. why didn't you show that? seems like the most direct obvious analogue to show, since you showed += for numpy.

(2) the type promotion indeeded differed. numpy promotes to float64 and pytroch keeps as float32. what's the brief rule here?

(3) integer division seems the same in that division produces floats, and // produces ints. one difference is that numpy uses float64 and pytorch produces float32, though that's now what you seemed to be pointing out. here's the output, i added // to numpy's tests. if you can clarify what you were trying to show that'd be great.

============================================================
3. INTEGER DIVISION
============================================================
NumPy int / int: [3.5        2.66666667 2.25      ], dtype: float64
NumPy int // int: [2 2 2], dtype: int64
PyTorch int / int: tensor([3.5000, 2.6667, 2.2500]), dtype: torch.float32
PyTorch int // int: tensor([3, 2, 2]), dtype: torch.int64

(4) numpy arrays DO have a device attribute, and even a to_device function. i get they always live on the cpu, but we can say arr.device and get "cpu" back, so saying 'there's no device attribute' is simply incorrect

(5) i hve no idea what you were trying to show with slice vs view. the results are the same. were you trying to say something about contiguous memory? if there's a difference here i don't get it. here's the output:

============================================================
5. VIEW BEHAVIOR
============================================================
NumPy original after modifying slice: [[999   2   3]
 [  4   5   6]]
NumPy slice is view: True
PyTorch slice is_contiguous: True
PyTorch original after modifying slice: tensor([[999,   2,   3],
        [  4,   5,   6]])

(6) the gradients seems clear, numpy simply lacks this feature

(7) i have no idea what you're trying to show here, but the results for both are exactly the same:

NumPy boolean indexing result:
[[1 2 3]]
PyTorch boolean indexing result:
tensor([[1, 2, 3]])
NumPy: (3,0,5) + (3,1,5) = []
NumPy: (3,0,5) + (3,1,5) = shape (3, 0, 5)
PyTorch: (3,0,5) + (3,1,5) = tensor([], size=(3, 0, 5))
PyTorch: (3,0,5) + (3,1,5) = shape torch.Size([3, 0, 5])

Its eventual response

The actual meaningful differences between NumPy and PyTorch are:
- Gradients/Autograd - PyTorch has it, NumPy doesn't
- Type promotion - PyTorch is more conservative (float32 + int → float32), NumPy promotes more aggressively (float32 + int → float64)
- GPU support - PyTorch tensors can live on GPU, NumPy is CPU-only
- Explicit in-place methods - PyTorch has .add_(), .mul_() etc. in addition to +=, *=

My note: pytorch also has explicit copying .add() functions on arrays, which numpy
lacks. in numpy, you have to call np.add() and pass arrays.
"""

import numpy as np
import torch

print("=" * 60)
print("1. IN-PLACE OPERATIONS")
print("=" * 60)

# NumPy: in-place with += modifies original
np_arr = np.array([1, 2, 3])
np_arr_id = id(np_arr)
np_arr += 10
print(f"NumPy after +=: {np_arr}, same object: {id(np_arr) == np_arr_id}")

# PyTorch: in-place with += modifies original too
pt_tensor = torch.tensor([1, 2, 3])
pt_tensor_id = id(pt_tensor)
pt_tensor += 10
print(f"PyTorch after +=: {pt_tensor}, same object: {id(pt_tensor) == pt_tensor_id}")

# PyTorch: explicit in-place with underscore
pt_tensor2 = torch.tensor([1, 2, 3])
pt_tensor2_id = id(pt_tensor2)
pt_tensor2.add_(10)  # explicit in-place
print(
    f"PyTorch after add_(): {pt_tensor2}, same object: {id(pt_tensor2) == pt_tensor2_id}"
)

# PyTorch: without underscore creates new tensor
pt_tensor3 = torch.tensor([1, 2, 3])
pt_tensor3_id = id(pt_tensor3)
pt_tensor3 = pt_tensor3.add(10)  # creates new tensor
print(
    f"PyTorch after add(): {pt_tensor3}, same object: {id(pt_tensor3) == pt_tensor3_id}"
)

print("\n" + "=" * 60)
print("2. TYPE PROMOTION")
print("=" * 60)

# NumPy: liberal type promotion
np_float32 = np.array([1.0], dtype=np.float32)
np_int32 = np.array([2], dtype=np.int32)
np_result = np_float32 + np_int32
print(f"NumPy: float32 + int32 = {np_result.dtype}")

# PyTorch: more conservative, stays float32
pt_float32 = torch.tensor([1.0], dtype=torch.float32)
pt_int32 = torch.tensor([2], dtype=torch.int32)
pt_result = pt_float32 + pt_int32
print(f"PyTorch: float32 + int32 = {pt_result.dtype}")

print("\n" + "=" * 60)
print("3. INTEGER DIVISION")
print("=" * 60)

# NumPy: behavior depends on types
np_int_a = np.array([7, 8, 9])
np_int_b = np.array([2, 3, 4])
np_div = np_int_a / np_int_b
print(f"NumPy int / int: {np_div}, dtype: {np_div.dtype}")

np_int_a2 = np.array([4, 6, 8])
np_int_b2 = np.array([2, 3, 4])
np_div2 = np_int_a2 // np_int_b2
print(f"NumPy int // int: {np_div2}, dtype: {np_div2.dtype}")


# PyTorch: / always produces float
pt_int_a = torch.tensor([7, 8, 9])
pt_int_b = torch.tensor([2, 3, 4])
pt_div = pt_int_a / pt_int_b
print(f"PyTorch int / int: {pt_div}, dtype: {pt_div.dtype}")

# PyTorch: use // for integer division
pt_floor_div = pt_int_a // pt_int_b
print(f"PyTorch int // int: {pt_floor_div}, dtype: {pt_floor_div.dtype}")

print("\n" + "=" * 60)
print("4. DEVICE CONSIDERATIONS")
print("=" * 60)

# PyTorch: tensors must be on same device for operations
pt_cpu = torch.tensor([1, 2, 3])
print(f"Torch Tensor on: {pt_cpu.device}")

# This would fail if we had a GPU and tried to add CPU + GPU tensors:
# pt_gpu = torch.tensor([1, 2, 3]).cuda()
# pt_result = pt_cpu + pt_gpu  # Error!

# NumPy: no device concept, always CPU
np_arr = np.array([1, 2, 3])
print(f"NumPy array on: {np_arr.device} (only option)")

print("\n" + "=" * 60)
print("5. VIEW BEHAVIOR")
print("=" * 60)

# NumPy: slicing creates views (implicit)
np_original = np.array([[1, 2, 3], [4, 5, 6]])
np_slice = np_original[0, :]
np_slice[0] = 999
print(f"NumPy original after modifying slice: {np_original}")
print(f"NumPy slice is view: {np.shares_memory(np_original, np_slice)}")

# PyTorch: similar but more explicit about views
pt_original = torch.tensor([[1, 2, 3], [4, 5, 6]])
pt_slice = pt_original[0, :]
print(f"PyTorch slice is_contiguous: {pt_slice.is_contiguous()}")
pt_slice[0] = 999
print(f"PyTorch original after modifying slice: {pt_original}")

print("\n" + "=" * 60)
print("6. GRADIENTS")
print("=" * 60)

# PyTorch: can track gradients
pt_x = torch.tensor([2.0, 3.0], requires_grad=True)
pt_y = pt_x**2
pt_z = pt_y.sum()
pt_z.backward()
print(f"PyTorch gradients: {pt_x.grad}")

# NumPy: no gradient tracking
np_x = np.array([2.0, 3.0])
np_y = np_x**2
print(f"NumPy has no gradient tracking (manual differentiation needed)")

print("\n" + "=" * 60)
print("7. BROADCASTING EDGE CASE")
print("=" * 60)

# Edge case: Boolean indexing with broadcasting
# NumPy is more permissive with certain advanced indexing patterns

np_arr = np.array([[1, 2, 3], [4, 5, 6]])
np_mask = np.array([True, False])
# NumPy: boolean array indexing along first dimension
np_result = np_arr[np_mask]
print(f"NumPy boolean indexing result:\n{np_result}")

pt_arr = torch.tensor([[1, 2, 3], [4, 5, 6]])
pt_mask = torch.tensor([True, False])
pt_result = pt_arr[pt_mask]
print(f"PyTorch boolean indexing result:\n{pt_result}")

# More subtle edge case: empty dimension broadcasting
np_a = np.ones((3, 0, 5))
np_b = np.ones((3, 1, 5))
try:
    np_result = np_a + np_b
    print(f"NumPy: (3,0,5) + (3,1,5) = {np_result}")
    print(f"NumPy: (3,0,5) + (3,1,5) = shape {np_result.shape}")
except Exception as e:
    print(f"NumPy error: {e}")

pt_a = torch.ones((3, 0, 5))
pt_b = torch.ones((3, 1, 5))
try:
    pt_result = pt_a + pt_b
    print(f"PyTorch: (3,0,5) + (3,1,5) = {pt_result}")
    print(f"PyTorch: (3,0,5) + (3,1,5) = shape {pt_result.shape}")
except Exception as e:
    print(f"PyTorch error: {e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
