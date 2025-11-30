import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tensor Indexing

    Working through the basics of how to index pytorch tensors.

    A tensor `t` with _n_ dimensions can be indexed with up to _n_ selection operations `t[d1,...,dn]`. We cover `di` being:
    - an integer
    - `None`
    - a slice `:`, which can look like
        - `:`
        - `start:stop:increment`
        - ... or anything in-between (all optional)
    - `...`
    - a tensor

    Of particular note, we look at the mechanics of three complex operations:
    1. broadcasting
    2. multi-tensor indexing
    3. how the resulting shape is computed

    See the summary at the end.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    return mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AI Prompt

    _Below is the AI Prompt I used to get an ordered walkthrough of self play experiments to expose myself to the material._

    I'm going to be working through Sasha Rush's "Tensor-Puzzles." Here's what the material is about:

    ---

    These puzzles are about *broadcasting*. Know this rule
    > There is a rule you should learn at long last,<br/>
    > combination of tensors the task.<br/>
    > Dims right-aligned,<br/>
    > extra left 1s assigned<br/>
    > match paired dimensions. Broadcast!
    >
    > ```text
    >  9  x 1 x 3
    > (1) x 8 x 1
    > ---------
    >  9  x 8 x 3
    > ```

    You are allowed @, arithmatic, comparison, `shape`, any indexing (e.g., `a[:j], a[:,None], a[arange(10)]`), and previous puzzle functions.

    You are _not allowed_ anything else. No `view`, `sum`, `take`, `squeeze`, `tensor`.

    ---

    Given this, I'd like to start by (re)learning the elementary pytorch operations that are allowed. Minimally, what they list: "@, arithmetic, comparison, shape, any indexing (e.g. a[:j], a[:, None], a[arange(10)])." But most crucially, I need to (re)learn broadcasting rules in practice.

    I want to play with these experimentally, so give me a sequence of concepts (above, plus broadcasting rules) and concrete toy tasks to try with them. Explain the basic concept, but don't write out all the solutions; give me the concept, an idea of a varaible to play with, and let me go from there.

    Once I'm finished with these, I'll go and work on the Tensor Puzzles. We're just doing primer material right now.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic Indexing
    """)
    return


@app.cell
def _(torch):
    a = torch.arange(10)
    a
    return (a,)


@app.cell
def _():
    # somehow, i'd completely forgotten (or never learned?) normal
    # python slicing has a step option too
    l = list(range(10))

    print()
    print("Forward:")
    print(l[::])
    print(l[0:len(l):1])
    print(l[2:9:2])

    # if the step is negative, it changes the default start/stop and
    # flips the iteration condition check (from < to >)
    print()
    print("Reverse:")
    print(l[::-1])
    print(l[len(l)-1:-len(l)-1:-1])
    print(l[8:1:-2])
    return


@app.cell
def _(a):
    # tensor indexing by [x,y] or [x][y]
    a_2x5 = a.reshape(2,5)
    print(a_2x5)
    print(a_2x5[1,3])
    print(a_2x5[1][3])

    # basic tensor slicing: negative steps don't work
    print()
    print(a[2:9:2])
    try:
        print(a[8:1:-2])
    except Exception as e:
        print("Exception attempting a[8:1:-2]:", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Slicing `[::,...,::]` to make views

    If a has d dimensions, then we can write up to d comma-separated
    slice expressions: `a[::,::,::,...,::]`

    Missing expressions are implicitly filled to the right as `:`
    """)
    return


@app.cell
def _(torch):
    # Basic slicing
    b = torch.arange(20).reshape(4,5)
    print("Original:")
    print(b)

    print()
    print("Row:")
    print(b[0])

    print()
    print("Col:")
    print(b[:,0])

    print()
    print("Subtensor:")
    print(b[1:3,1:4])

    print()
    print("Complex slice:")
    print(b[1::2,:3])
    return (b,)


@app.cell
def _(b):
    # missing dimensions to the right become :
    r = b.reshape(2,2,5)
    print(r)

    print()
    print("r[1] == r[1,:,:]")
    print(r[1])
    print(r[1, :, :])

    print()
    print("r[:,1] == r[:,1,:]")
    print(r[:,1])
    print(r[:,1,:])
    return


@app.cell
def _(torch):
    # using lots of indexing. at this point it's hard to even
    # follow the slicing visually.
    c = torch.arange(36).reshape(2,3,3,2)
    print(c)
    print(c[:1,::2,1::,1:])
    return


@app.cell
def _(b):
    # mutating a view mutates the original
    d = b.clone()

    print()
    print("Mutating a view:")
    st = d[1:3,1:4]
    st[0][0] = -42
    print(st)
    share_data = st.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    print(f"View shares data with original? {share_data}")

    print()
    print("... mutates the original:")
    print(d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ellipses (`...`) expands to fill all `:`
    """)
    return


@app.cell
def _(b):
    print(b)
    print()
    print(b.shape)
    return


@app.cell
def _(b):
    print("b[...,0]")
    print(b[...,0])
    print()

    print("b[0,...]")
    print(b[0,...])
    print()

    print("b[...]")
    print(b[...])
    print()
    return


@app.cell
def _(torch):
    bb = torch.arange(2*3*4*2).reshape(2,3,4,2)
    bb
    return (bb,)


@app.cell
def _(bb):
    bb[:,:,:,1]
    return


@app.cell
def _(bb):
    bb[...,1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pytorch currently supporst multiple ellipses `bb[...,1,...]`, but it's an error in numpy. Either way, it seems like bad style.

    https://github.com/pytorch/pytorch/issues/59787
    """)
    return


@app.cell
def _(bb):
    # selects last dimension. seems bad to use.
    bb[...,1,...]
    return


@app.cell
def _(mo):
    mo.md(r"""
    if `x` is `[batch, channel, height, width]`, then we can do

    ```python
    center_row = b[..., x.size(-2)//2, :]
    ```

    ...to not worry about exact number of previous dimensions. In other words, we can write this as shorthand if we know the shape ends with (height, width).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `None` injects new dimensions
    """)
    return


@app.cell
def _(b):
    b
    return


@app.cell
def _(b):
    b.shape
    return


@app.cell
def _(b, mo):
    print(b[:,None].shape)
    print(b[:,None])

    mo.md(r"""`None` in a slice injects a new dimension of size one.""")
    return


@app.cell
def _(b, mo):
    print(b[None].shape)
    print(b[None,:].shape)

    print()
    print(b[None])
    print(b[None,:])

    mo.md(r"""`None` follows the same rule that implicit `:` follow to fill all dimensions.""")
    return


@app.cell
def _(b, mo):
    print(b)
    print()
    print(b[0,None,2:4,None])

    mo.md(r"""`None` can be mixed along with normal slicing.""")
    return


@app.cell
def _(b, mo):
    print(b[None,None,0,1,None])
    print(b[0,None,None,None,1])

    mo.md(r"""Multiple ways of adding new dimensions result may give the same result.""")
    return


@app.cell
def _(b, mo):
    print(b[None,None,None,None,None,:,None,None,None,:,None].shape)
    print(b[None,None,None,None,None,:,None,None,None,:,None])

    mo.md(r"""You can go nuts with lots of `None`s""")
    return


@app.cell
def _(torch):
    # vectors? no problem
    v = torch.arange(5)
    print(v[None,None].shape)
    v[None,None]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple tensor indexing
    """)
    return


@app.cell
def _(b):
    print(b.shape)
    print()
    print(b)
    return


@app.cell
def _(b, mo, torch):
    print(torch.arange(3))
    print()
    print(b[torch.arange(3)])

    mo.md(r"""Tensors pick specific coordinates in a dimension""")
    return


@app.cell
def _(b, mo, torch):
    print(torch.arange(0,5,2))
    print()
    print(b[None,1:3,torch.arange(0,5,2)])

    mo.md(r"""This works alongside other dimension's slices""")
    return


@app.cell
def _(b, mo, torch):
    # b[(0,1,2),(0,1,2)] is matched up coordinate by coordinate
    # selects (0,0) and (1,1) and (2,2)
    print(b[torch.arange(3), torch.arange(3)])

    try:
        b[torch.arange(2), torch.arange(3)]
    except Exception as e:
        print(e)

    mo.md(r"""Multiple tensors in different dimensions are aligned point-wise to select specific coordinates. We'll come back after we learn about broadcasting.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Broadcasting

    For elementwise operations:
    - `+, -, *, /, // %, **`
    - `==, !=, <, <=, >, >=`
    - `&, |, ^, <<, >>`
    - operator equivalents (like `torch.add(a,b)`)
    - a ~dozen non-operator functions like `torch.maximum()` and `torch.where()`)
    """)
    return


@app.cell
def _(torch):
    t1 = torch.tensor([1,2,3])
    print("(3)")
    print("t1 =", t1)
    # print(t1.shape)

    t2 = torch.tensor([[10], [20]])
    print()
    print("(2,1)")
    print("t2 =", t2)
    # print(t2.shape)

    t3 = t1 + t2
    print()
    print("(2,3)")
    print("t3 = t1 + t2")
    print(t3)
    # print(t3.shape)
    return t1, t2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What happens:
    1. align right
    2. pad left with 1s
    3. for each dimension
        - if match, no-op
        - if one is 1, duplicate it to the other (target) dimension
        - otherwise, error
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Worked example

    ```text
    our input tensors:
    ------------------
    3
    2 x 1

    1. align right:
    ------------------
        3
    2 x 1

    2. pad left with 1s:
    ------------------
    1 x 3
    2 x 1

    3. match/expand by dim:
    -----------------------
    1 x 3
    2 x 1
    ^   ^
    |   1 can expand to 3
    |
    1 can expand to 2

    result:
    2 x 3
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rendered Intermediate Broadcasting
    """)
    return


@app.cell
def _(t1, t2):
    # Here's how the expansions work
    print("t1 is shape (3), expanded to a virtual view of (2,3)")
    print(t1)
    print(t1.expand(2,3))

    print()
    print("t2 is shape (2,1), expanded to a virtual view of (2,3)")
    print(t2)
    print(t2.expand(2,3))

    print()
    print("Our elementwise addition is now trivial")
    print(t1.expand(2,3))
    print("+")
    print(t2.expand(2,3))
    print("=")
    print(t1 + t2)
    return


@app.cell
def _(torch):
    # practice
    x = torch.arange(12).reshape(4,3,1)
    print("x:")
    print(x.shape)

    y = torch.arange(6).reshape(3,2)
    print()
    print("y:")
    print(y.shape)

    # x + y shape should be
    # (4,3,2)
    print()
    print("x+y:")
    print((x + y).shape)
    return


@app.cell
def _(torch):
    # practice
    row = torch.arange(5)          # (5)
    col = torch.arange(4)[:,None]  # (4,1)

    # row op col will be (4,5)
    (row ** col).shape

    # simple 5 x 5 multiplication table
    m = torch.arange(1,6)
    m[:,None] * m
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison operators
    """)
    return


@app.cell
def _(a):
    a
    return


@app.cell
def _(a):
    a > 5
    return


@app.cell
def _(a):
    a == 3
    return


@app.cell
def _(torch):
    grid = torch.arange(6).reshape(2,3)
    grid
    return (grid,)


@app.cell
def _(torch):
    o = torch.tensor([1,2,1])
    o
    return (o,)


@app.cell
def _(o):
    o.shape
    return


@app.cell
def _(grid, o):
    # comparison of (2,3) vs (3) results in (3) being broadcast
    #               (2,3) <-- will repeat along first dimension
    o < grid
    return


@app.cell
def _(grid, torch):
    # (2,1) broadcast along second dimension
    grid == torch.tensor([[1],[2]]) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix multiplication (`@`)
    """)
    return


@app.cell
def _(torch):
    m1 = torch.arange(12).reshape(3,4)
    m1
    return (m1,)


@app.cell
def _(torch):
    m2 = torch.arange(20).reshape(4,5)
    m2
    return (m2,)


@app.cell
def _(m1, m2):
    m1 @ m2
    return


@app.cell
def _(m1, m2):
    # inner dims gotta align
    try:
        m2 @ m1
    except Exception as e:
        print(e)
    return


@app.cell
def _(m1, m2):
    # leading dimensions will result in batched matrix multiplication
    m1.expand(2,3,4) @ m2
    return


@app.cell
def _(m1, m2):
    # multiple leading dimensions? no prob
    m1.expand(2,2,3,4) @ m2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Advanced tensor indexing
    """)
    return


@app.cell
def _(a):
    aa = a + 10
    aa
    return (aa,)


@app.cell
def _(aa, torch):
    indices = torch.tensor([2,5,2,8])
    aa[indices]
    return


@app.cell
def _(b):
    b
    return


@app.cell
def _(b):
    b.shape
    return


@app.cell
def _(b, torch):
    i2 = torch.tensor([2,3])
    b[i2]
    return (i2,)


@app.cell
def _(b, i2):
    # perhaps not what you'd expect!
    b[i2,i2]
    return


@app.cell
def _(b, i2):
    # this would be more intuitive. what's going on?
    b[i2,2:4]
    return


@app.cell
def _(mo):
    mo.md(r"""
    Index tensors:
    1. broadcast together
    2. pairs elements position-by-position
    """)
    return


@app.cell
def _(i2, torch):
    # 1. broadcast together
    i2b = torch.broadcast_tensors(i2,i2)
    i2b
    return (i2b,)


@app.cell
def _(i2b):
    inner_len = len(i2b[0])
    inner_len
    return (inner_len,)


@app.cell
def _(i2b, inner_len):
    # 2. reconstruct the index by iterating coordinate by coordinate
    index = []
    for i in range(inner_len):
        coordinate = []
        for dim in i2b:
            coordinate.append(int(dim[i]))
        index.append(coordinate)
    print(index)
    return


@app.cell
def _(i2, torch):
    # equivalently (maybe more accurately if they become tuples)
    # NOTE: Not for higher-dimensional tuples
    print(list(zip(*[x.tolist() for x in torch.broadcast_tensors(i2, i2)])))
    return


@app.cell
def _(i2, torch):
    # use 3 tensors to select 3d points
    print(list(zip(*[x.tolist() for x in torch.broadcast_tensors(i2, i2, i2)])))
    return


@app.cell
def _(b):
    # in other words, each tensors specifies a list of coordinates in its
    # dimension
    b
    return


@app.cell
def _(b, torch):
    # for example, let's grab the zig-zag pattern 2, 8, 12, 18 from b
    d1 = torch.tensor([0,1,2,3]) # d1 coordinates
    d2 = torch.tensor([2,3,2,3]) # d2 coordinates
    b[d1, d2]
    return


@app.cell
def _(b, torch):
    # let's demonstrate how broadcasting works here
    d3 = torch.tensor([1])
    d4 = torch.tensor([2,3,2,3])
    # [1] is broadcast to [1,1,1,1]
    # so this becomes [(1,2), (1,3), (1,2), (1,3)]
    print(b[d3, d4])
    print(b[d3,d4].shape)
    return


@app.cell
def _(torch):
    d5 = torch.tensor([1])           # (1)
    d6 = torch.tensor([[[2,3,2,3]]]) # (1, 1, 4)
    #                 broadcast shape: (1, 1, 4)

    # do we expect this to work?
    d5b, d6b = torch.broadcast_tensors(d5, d6)
    print(d5b)
    print(d6b)
    # note: our tolist/zip trick doesn't work any more because these
    #       are too nested! proper function below (show_index_pairs())
    return d5, d6


@app.cell
def _(b, d5, d6, mo):
    print(b[d5, d6])
    print(b[d5, d6].shape)

    mo.md(r"""Interesting! It still selected coordinate-pair-wise, but the output shape now matches the broadcast index shape. instead of (4), it's (1, 1, 4).""")
    return


@app.cell
def _(d5, d6, mo, torch):
    # let's dive in

    # dont' worry about this function's mechanics for now
    def show_index_pairs(*idx):
        b = torch.broadcast_tensors(*idx)
        pairs = torch.stack(b, dim=-1)   # shape (..., n_idx)
        return pairs.reshape(-1, pairs.shape[-1])  # flatten for display

    ip = show_index_pairs(d5, d6)
    print(ip)
    print(ip.shape)

    mo.md(r"""There are 4 2d coordinates.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pytorch preserves the index shape. The shape of the output begins with the broadcasted shape of all index tensors.
    """)
    return


@app.cell
def _(torch):
    a24 = torch.arange(4*3*2).reshape(4,3,2)
    a24
    return (a24,)


@app.cell
def _(a24, torch):
    idx1 = torch.tensor([2,3])
    idx2 = torch.tensor([1,0])

    # this produces
    # (2,1,:)  <-- 3rd group, 2nd row, all columns = [14, 15]
    # (3,0,:)  <-- 4th group, 1st row, all columns = [18, 19]

    a24[idx1, idx2, :]
    return idx1, idx2


@app.cell
def _(a24, idx1, idx2, torch):
    # if we check the dimesions
    print(idx1.shape)  # (2)
    print(idx2.shape)  # (2)
    # each broadcast shape is still (2)
    print(torch.broadcast_tensors(idx1, idx2)) 

    # result is [2,2]
    print(a24[idx1, idx2, :].shape)
    return


@app.cell
def _(torch):
    # non-adjacent dimensions
    b24 = torch.arange(2*3*4).reshape(2,3,4)
    print(b24)

    ii = torch.tensor([1,1])  # (2)
    kk = torch.tensor([2,1])  # (2)
    #          broadcast shape: (2)

    print()
    print(b24[ii, :, kk])
    print(b24[ii, :, kk].shape)

    # picks [1,:,2] and [1,:,1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's the full rule:
    1. pull out tensor indices in order, broadcast. this is the leading shape
    2. append remaining dimensions (slices, integers, ellipses) in order
        - integers are removed
        - the rest produce shapes
    """)
    return


@app.cell
def _(torch):
    pp = torch.arange(2*3*4*5).reshape(2,3,4,5)
    pp
    return (pp,)


@app.cell
def _(torch):
    p1 = torch.tensor([1,0])
    p2 = 1
    p3 = torch.tensor([3,1])
    p4 = slice(None)  # TIL this is builtin
    return p1, p2, p3, p4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's figure out what will happen.

    1. **pull out tensor indices in order, broadcast. this is the leading shape.**

    our tensor indices are dimensions 1 and 3. These each have shape (2). They'll be broadcast to shape (2). They'll define 2 coordinates, (1,3) and (0,1). So our output shape should start with (2,).

    2. **append remaining dimensions (slices, integers, ellipses) in order**

    our remaining dimensions in order are:
    - 2: just an integer, so it's removed
    - 4: a slice, so ts full dimension should be kept (5)

    So, our final output shape should be (2,5)

    What will it be?

    ```text
    [1, 1, 3, :] should grab
    [ 95,  96,  97,  98,  99]

    [0, 1, 1, :] should grab
    [ 25,  26,  27,  28,  29]

    so our result should look like:
    [[ 95,  96,  97,  98,  99]
     [ 25,  26,  27,  28,  29]]

    which matches our expected shape of (2,5)
    ```
    """)
    return


@app.cell
def _(p1, p2, p3, p4, pp):
    pp[p1,p2,p3,p4]
    return


@app.cell
def _(a):
    # boolean operations create masks that can then be used as selection
    print(a)
    print(a > 5)
    print(a[a > 5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mini challenges
    """)
    return


@app.cell
def _(a, torch):
    # 1. create a pairwise distance matrix, where dist[i][j] is |a[i] - a[j]|
    #    (10) right-align
    # (10, 1)
    # -------
    # ( 1,10) left-pad 1
    # (10,10)
    # -------
    # (10,10) final broadcast shape. each gets copied along the other dim.
    torch.abs(a[:,None] - a)
    return


@app.cell
def _(a):
    # 2. checkerboard pattern alternting 0s and 1s
    # the key is that am2[:,None] gets broadcast along cols
    # and am2 gets broadcast along rows.
    am2 = a % 2
    (am2[:,None] + am2) % 2
    return


@app.cell
def _(a):
    # 3. upper triangular boolean mask
    (a[:,None] - a) < 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - both python and pytorch slices support `[start:end:increment]`
        - but only python's increment can be negative
    - specify `[d1,d2,...,dn]` for n dimensions
    - each dimension can be an integer, slice (`s:e:i`), `None`, `...`, or tensor index
    - unspecified dimensions are filled with `:` to the right
    - `None` injects a new dimension, as many as you like
    - `...` (ellipses) fills as many `:` as will fit
        - use at most one `...`, though pytorch (not numpy) allows multiple
    - `slice()` is actually a python builtin. `slice(None)` means `:`
    - tensor slices create views, which share memory
    - broadcasting's procedure:
        1. right-align
        2. pad left with 1s
        3. match, expand 1s to target, or fail if not equal
    - broadcasting doesn't copy memory, it's a view on the data
    - **but think of broadcasting like copying data on a broadcasted dimension, because that's how it behaves**
    - there are tons of elementwise operations. they all broadcast.
        - elementwise functions include include math (`+`), comparison (`>=`), and bitwise (`|`) operators, their non-operator named functions, and over a dozen other functions
    - matrix multiplication applies using normal rules (isn't elementwise), and can be batched along preceeding dimensions
    - for multiple tensors as indices, they are broadcast, and coordinates are constructed from their aligned tuples
        - i.e., the ith number from each of d tensors creates a d-dimensional coordinate. n-length tensors make n coordinates (each of d-dims).
    - the full resulting shape of an index `[]` is computed as follows:
        1. grab all tensor dimensions in order (L&rarr;R); broadcast; begin the output shape with this broadcast shape
        2. for all remaining dimensions in order (L&rarr;R), integers are removed, and slices are computed and their dimensions appended
    """)
    return


if __name__ == "__main__":
    app.run()
