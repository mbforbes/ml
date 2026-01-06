import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    return mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # General Pytorch and Numpy Notes

    ## Data

    - how to get bytes used by an array:
        - numpy: `a.nbytes()`
        - torch: `t.element_size() * t.nelement()`
    - torch shares memory with numpy upon `t = torch.from_numpy(a)`
        - i.e., `a[0][0] = 42` mutates `t`, and vice versa

    ## Utils

    - `outer(a, b)` computes the m x n elementwise product `a` (m) and `b` (n)
    - many diagonal-flavored utils: `diag()`, `eye()`, `triu()`, `tril()`
    - use `arange(start, end, step)` instead of `range(...)` (historical)

    todo
    - `np.repeat()`
    - `np.random.normal` sampling over new empty broadcast dims
    - `np.linalg.norm`

    ## Shapes

    - use `dim()` instead of `ndimension()` (historical) to get rank
    - use `.shape` instead of `size()` to get shape (`torch.Size([...]`)
    - use `size(d)` to get length of dimension d
    - 0-dimensional scalars exist (`dim() == 0`)

    ## Types

    `jaxtyping` is probably the best start to try to get working, but on an initial attempt Ruff thought there were type errors with its syntax, so I abandoned for now.

    ## Viz

    I'm most interested in using
    - Tensorboard (builtin, classic, lightweight)
    - Trackio

    The industry standards seem to be
    - Weights & Biases
    - MLflow

    Other newcomers like Neptune, Aim, or ClearML may be worth considering, esp. for LLM-specific viz utils (like comparing text outputs and ratings).
    """)
    return


@app.cell
def _(torch):
    # solutions to tensor puzzles so far (saving for future scratchpad reference)

    def arange(i: int):
        "Use this function to replace a for-loop."
        return torch.tensor(range(i))

    def where(q, a, b):
        "Use this function to replace an if-statement."
        return (q * a) + (~q) * b

    def ones(i):
        return arange(i) * 0 + 1

    def sum(a):
        return (a @ ones(a.size(0)))[None]

    def outer(a,b):
        return a[:,None] * b

    def diag(a):
        return a[arange(a.size(0)), arange(a.size(0))]

    def eye(j):
        return (arange(j)[:,None] == arange(j)) * 1

    def triu(j):
        return (arange(j)[:,None] <= arange(j)) * 1

    def cumsum(a):
        return ones(a.size(0)) @ (triu(a.size(0)) * (a[:,None] * ones(a.size(0))))

    def diff(a, i):
        """a (vec), i (int) is len(a). not sure why i given here.
        note that this differs from numpy/pytorch diff:
        - numpy/pytorch: out[i] = in[i+1] - in[i]                  --- output len i-1
        - this:          out[i] = in[i] - in[i-1] (or in[0] for 0) --- output len i
        also, numpy/pytorch do take an int argument, which runs this repeatedly.
        """
        return where(arange(i) == 0, a, a - a[arange(i) - 1])

    def og_diff(a):
        """numpy/pytorch diff, w/o iteration. i guess this was too easy."""
        return a[1:] - a[:-1]
    return (arange,)


@app.cell
def _(arange):
    a = arange(10)
    # curspot: next up: puzzle 9 vstack
    return


if __name__ == "__main__":
    app.run()
