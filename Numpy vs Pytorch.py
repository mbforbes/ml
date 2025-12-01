import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Numpy vs Pytorch

    _The different bits_

    I have this divided up into two sections: potential gotchas, and minor/obvious differences.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    return mo, np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gotchas

    _Subtle behavior differences you might not expect._
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Casting: Numpy values precision, Pytorch values consistency

    When combining mixed types:
    - **numpy** errs on the side of **precision**: a `float32` can't represent all values that `int32` can, so combining `float32` and `int32` results in `float64`
    - **pytorch** errs on the side of **consistency**: promote an integer to the floating point type of the other operand. For example, `float16` + `uint64` becomes `float16`.

    Similarly, for dividing ints, each has the same behavior whether you're dividing int8/int8 up through int64/int64
    - numpy: float64
    - pytorch: float32

    For both, combining two different widths of the same type (e.g., `float16` and `float64`) results in the wider one (`float64`).

    See the code snippets below for demos of (a) float + int casting, (b) int / int results.
    """)
    return


@app.cell(hide_code=True)
def _(np, torch):
    def casting():
        print("Both adding f32 + i32")
        print("---------------------")
        np_f32 = np.array([1.0], dtype=np.float32)
        np_i32 = np.array([2], dtype=np.int32)
        np_res = np_f32 + np_i32
        print(f"numpy:       {np_f32.dtype} +       {np_i32.dtype} =       {np_res.dtype}")

        t_f32 = torch.tensor([1.0], dtype=torch.float32)
        t_i32 = torch.tensor([2], dtype=torch.int32)
        t_res = t_f32 + t_i32
        print(f"torch: {t_f32.dtype} + {t_i32.dtype} = {t_res.dtype}")

    casting()
    return


@app.cell(hide_code=True)
def _(np, torch):
    def div():
        print("Both diving i8/i8:")
        print("----------------------")
        a = np.array([1], dtype='int8')
        b = np.array([2], dtype='int8')
        c = a / b
        print(f'numpy: {a.item()}/{b.item()} = {c.item()} (      {c.dtype})')

        a = torch.tensor([1], dtype=torch.int8)
        b = torch.tensor([2], dtype=torch.int8)
        c = a / b
        print(f'torch: {a.item()}/{b.item()} = {c.item()} ({c.dtype})')


    div()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. List Indexing

    Both numpy and pytorch allow you to index an array/tensor with a **tuple** of arrays/tensors, and it will unpack the tuple as if you had passed them directly (i.e., perform the same as **advanced indexing**).
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def idxing_tuple():
        print("The array (3, 3):")
        a = np.arange(9).reshape(3,3)
        print(a)

        print()
        print("The index: a tuple of two (2,) arrays:")
        a_idx = tuple([np.array([0,2]), np.array([0,1])])
        print(f"a[{a_idx}]")

        print()
        print(f"The result is: {a[a_idx].shape}")
        print(a[a_idx])

        print()
        print("This is unpacked as if we'd indexed by the tensors directly:")    
        print(f"a[np.array([0,2]), np.array([0,1])]")
        print()
        print("Same result:")
        print(a[np.array([0,2]), np.array([0,1])])

    idxing_tuple()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, numpy allows you to index by a **list** of arrays, and it will construct a stacked index for that dimension using it. Pytorch will give a warning if you try to do this, and instead treat it like a tuple.

    However, it seems like pytorch 2.9 will switch to numpy's behavior.
    """)
    return


@app.cell(hide_code=True)
def _(np, torch):
    def idxing_np():
        print("Numpy:")
        print()
        print("The array (3, 3):")
        a = np.arange(9).reshape(3,3)
        print(a)

        print()
        print("The index: a list of two (2,) arrays:")
        a_idx = [np.array([0,2]), np.array([0,1])]
        print(a_idx)

        print()
        print(f"The result is: {a[a_idx].shape}")
        print(a[a_idx])

        print()
        a_alt_idx = np.array([[0,2],[0,1]])
        print("This is as if we'd indexed the first dimension by")    
        print(f"the stacked {a_alt_idx.shape} matrix:")
        print(a_alt_idx)
        print()
        print("Same result:")
        print(a[a_alt_idx])

    def idxing_torch():
        print("Pytorch:")
        print()
        print("The array (3, 3):")
        a = torch.arange(9).reshape(3,3)
        print(a)

        print()
        print("The index: a list of two (2,) arrays:")
        a_idx = [torch.tensor([0,2]), torch.tensor([0,1])]
        print(a_idx)

        print()
        warning_res = a[a_idx]
        print(f"The result is: {warning_res.shape}")
        print(warning_res)

        print()
        a_alt_idx = torch.tensor([[0,2],[0,1]])
        print("This is DIFFERENT from if we'd indexed the first dimension by")    
        print(f"the stacked {a_alt_idx.shape} matrix:")
        print(a_alt_idx)
        print()
        print("...which results in:")
        print(a[a_alt_idx,:])
    return idxing_np, idxing_torch


@app.cell(hide_code=True)
def _(idxing_np):
    idxing_np()
    return


@app.cell
def _(idxing_torch):
    idxing_torch()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. `nonzero()`

    Say our array/tensor has rank (num dims) R, and n nonzero elements. `nonzero()` returns:
    - numpy: an R-tuple of n-dimensional arrays
    - pytorch: a single (R x n) tensor

    `as_tuple=True` may be passed for pytorch to match numpy's semantics. See below:
    """)
    return


@app.cell(hide_code=True)
def _(np, torch):
    a = np.zeros(9, dtype=int).reshape(3,3)
    a[0,0] = 1
    a[0,2] = 1
    a[2,2] = 1
    print("Here's our 3x3 matrix of study:")
    print(a)
    print()
    print("Numpy: a.nonzero() returns a rank-tuple of n-dimensional numpy arrays,")
    print("where R is the rank (num dims) of the original array (here 2), and there")
    print("are n nonzero coordinates (here 3):")
    print(a.nonzero())

    print()

    t = torch.zeros(9, dtype=int).reshape(3,3)
    t[0,0] = 1
    t[0,2] = 1
    t[2,2] = 1
    print(f"Pytorch: t.nonzero() returns a single (rank x num-nonzeros) {t.nonzero().shape} tensor:")
    print(t.nonzero())
    print()
    print("... unless as_tuple=True is passed, in which case it matches numpy's semantics:")
    print(t.nonzero(as_tuple=True))
    return a, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because tuples and tensors index differently, `m[m.nonzero()]` gives different results for numpy and pytorch. Again, using `as_tuple=True` with pytroch matches them up.
    """)
    return


@app.cell(hide_code=True)
def _(a, t):
    print("a[a.nonzero()]:")
    print(a[a.nonzero()])
    print()
    print("t[t.nonzero()]:")
    print(t[t.nonzero()])
    print()
    print("t[t.nonzero(as_tuple=True)]:")
    print(t[t.nonzero(as_tuple=True)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Minor / Obvious Differences

    These differences are either minor (e.g., there are new methods) or obvious (e.g., Pytorch tracks computations and can compute gradients, because it was written to do this).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. On-Object Operation Methods

    Both numpy and pytorch support:
    - library `.add()` (`np.add()` / `torch.add()`)
    - `x + y` (new)
    - `x += y` (mutating)

    Pytorch adds operation methods directly on tensors:
    - `t.add()` (new)
    - `t.add_()` (mutating)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Pytorch supports GPUs

    Numpy now does also have `.device` and `.to_device()`, but it's just hardcoded `"cpu"` and no-op.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Pytorch differentiates and computes gradients

    This is why it was written.

    Here's a simple example:

    $$
    y = x^2 \\
    \frac{dy}{dx} = 2x
    $$

    So, if $x=5$, then $\frac{dy}{dx} = 2x = 10$. Let's write this in pytorch and verify:
    """)
    return


@app.cell
def _(torch):
    def deriv():
        x = torch.tensor(5.0, requires_grad=True)
        y = x**2
        print("y's gradient fn:", y.grad_fn)
        y.backward()
        print("x's gradient:   ", x.grad)

    deriv()
    return


if __name__ == "__main__":
    app.run()
