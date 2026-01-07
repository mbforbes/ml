"""Iterative k-means"""

import code

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def kmeans(points: np.ndarray, n_means: int):
    """points: (n, d)"""
    # print(f"Running {len(points)} with {n_means} means")
    # settings
    n, d = points.shape
    max_steps = 1000
    stop_delta = 0.01

    # init randomly, but take into account observed mean and variance of all points.
    # we'll init using a gaussian, which happens to match how the true means were
    # generated, so we could init with a uniform if we wanted to remove that nice
    # coincidence.
    dmeans = np.mean(points, axis=0)
    dstds = np.std(points, axis=0)
    means = np.random.normal(dmeans[None, :], dstds[None, :], (n_means, d))
    # other init ideas:
    # - choose k random points
    # - split all points into k groups and mean each

    assignments = np.zeros(n, dtype=int)
    prev_total_sq_dists = float("inf")
    for _step in range(max_steps):
        total_sq_dists = 0.0

        # assign points to means.
        # points:  (n, d)
        # means:   (k, d)
        # results: (n, k)

        # slow loop version:
        # for i, point in enumerate(points):
        #     # means is (k x d), point is (d), result is (k)
        #     sq_dists = ((means - point) ** 2).sum(axis=1)
        #     assignment = sq_dists.argmin()
        #     total_sq_dists += sq_dists[assignment]
        #     assignments[i] = assignment

        # vectorized solution with intermediate (n x k x d)
        # sq_dists = ((points[:, None, :] - means[None, :, :]) ** 2).sum(axis=2) # (n x k)
        # assignments = sq_dists.argmin(axis=1)  # (n)
        # total_sq_dists = sq_dists[np.arange(n), assignments].sum()

        # vectorized solution without intermediate (n, k, d)
        # ||a - b||^2 == a^2 - 2ab + b^2
        # for a given point, ||p(d,) - m(d,)||^2
        psq = (points**2).sum(axis=1)  # (n,)
        msq = (means**2).sum(axis=1)  # (k,)
        pm = points @ means.T  # (n,d) x (d,k) -> (n,k)
        sq_dists = psq[:, None] - 2 * pm + msq[None, :]  # (n,k)
        assignments = sq_dists.argmin(axis=1)
        total_sq_dists = sq_dists[np.arange(n), assignments].sum()

        # print(f"Step {step}, total dists: {total_dists}")

        # compute new means from current assignments
        for k in range(n_means):
            mean_points = points[assignments == k]

            # interesting degen case: no points! we'll re-init randomly. may be fine to
            # accept the L and just let next run get it.
            if len(mean_points) == 0:
                means[k] = np.random.normal(dmeans, dstds)
                continue

            means[k] = mean_points.mean(axis=0)

            # note: could track (prev, cur) means and print how much shifted

        delta = abs(prev_total_sq_dists - total_sq_dists)
        prev_total_sq_dists = total_sq_dists  # save immediately so we can return
        if delta < stop_delta:
            # print(f"Stopping as deltas differ by < {stop_delta}")
            break

    return assignments, prev_total_sq_dists


def rand_index(
    gold_assignments: np.ndarray, candidate_assignments: np.ndarray
) -> float:
    """returns rand index in [0, 1]."""
    # for each pair of points (i, j), the candidate assignment gets a mark if it
    # matches gold assignments in terms of whether i and j shared a cluster.

    marks = 0  # just to not reuse the word "point"
    n = len(gold_assignments)
    g = gold_assignments
    c = candidate_assignments
    total = n * (n - 1) / 2

    assert g.shape == c.shape

    # # loop version
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         g_same = g[i] == g[j]
    #         c_same = c[i] == c[j]
    #         marks += 1 if g_same == c_same else 0

    # constructing the two n^2 matrices, each entry (i,j) is whether points i and j
    # are assigned to the same cluster. we only look above the diagonal of this
    # symmetric matrix to only check the unique n(n-1)/2 entries where i < j. (we still
    # compare full matrices, so offset counts by diagonal and below)
    g_same = np.triu(g[None, :] == g[:, None], 1)
    c_same = np.triu(c[None, :] == c[:, None], 1)
    offset = n**2 - total
    marks = int((g_same == c_same).sum()) - offset

    return marks / total


def main():
    # settings
    d = 2  # note that only first 2 will be plotted
    k = 10
    n = 100  # from each k
    trials = 10  # how many times to run each candidate k for k means
    trial_ks = 19
    sigma = 2  # k-means assumes isotropic Gaussians (same covariance in all dims)

    # generate underlying data
    # k means, each in d dimensions
    means = np.random.normal(size=(k, d)) * 10
    # stds = np.abs(np.random.normal(size=(k, d)))  # real stds
    stds = np.ones((k, d)) * sigma  # isotropic
    # stds = np.ones((k, d))  # unit

    # generate points
    points = np.random.normal(means[:, None, :], stds[:, None, :], (k, n, d))

    # compute gold total dists from mean.
    # orig:
    # gold_total_dists = 0.0
    # for cur_k in range(k):
    #     # points[k] is (n, d), means[k] is (d)
    #     gold_total_dists += ((points[cur_k] - means[cur_k]) ** 2).sum(axis=1).sum()
    #
    # vectorized:
    gold_total_sq_dists = (
        ((points - means[:, None, :]) ** 2).reshape(n * k, d).sum(axis=1).sum()
    )

    # flatten and shuffle w/ an index to preserve original order
    flat_points = points.reshape(n * k, d)
    shuffled_index = np.arange(len(flat_points))
    np.random.shuffle(shuffled_index)
    shuffled_points = flat_points[shuffled_index]
    results = {}
    for candidate_k in tqdm(range(1, trial_ks + 1)):
        best_sq_dist, best_assignments = float("inf"), None
        for _ in range(trials):
            assignments, total_sq_dists = kmeans(shuffled_points, candidate_k)
            if total_sq_dists < best_sq_dist:
                best_sq_dist = total_sq_dists
                best_assignments = assignments
        results[candidate_k] = (best_sq_dist, best_assignments)

    # get gold cluster assignments.
    # do this by selecting  [0, 0, ..., 0, 1, 1, ..., k, k, ..., k] with shuffled index.
    #
    #                                (perfect)
    #                    orig         kmeans
    # idx   shuf idx   assignments  assignments
    # ---   ---           ---          ---
    # 0      2             0            1
    # 1      0             0            0
    # 2      1             1            0
    # 3      3             1            1
    gold_assignments = np.repeat(range(k), n)[shuffled_index]

    # quick hack as reminder to update this
    assert 5 * 4 == trial_ks + 1
    fig_render, axes = plt.subplots(5, 4, figsize=(18, 12))
    fig_ksse, axes_ksse = plt.subplots()
    k_vs_sse_data = []
    # code.interact(local=dict(globals(), **locals()))
    for i, ax in enumerate(axes.flat):
        if i == 0:
            # gold.
            # print info
            gold_ri = rand_index(gold_assignments, gold_assignments)
            print(
                f"Gold total sq dists (k={k}) (RI={gold_ri:.4f}):", gold_total_sq_dists
            )

            # plot scatter
            ax.scatter(
                flat_points[:, 0],
                flat_points[:, 1],
                c=np.repeat(range(k), n),
                alpha=0.5,
            )
            ax.set_title(f"Gold assignments (k={k})")

            # draw gold lines at x=k, y=gold total sq dists on k vs SSE plot
            axes_ksse.axvline(x=k, color="gold")
            axes_ksse.axhline(y=gold_total_sq_dists, color="gold")
        else:
            # k assignment.
            # print info
            ri = rand_index(gold_assignments, results[i][1])
            total_sq_dist, assignments = results[i]
            print(f"Best estimated k={i} (RI={ri:.4f}) total sq dists:", total_sq_dist)

            # plot scatter
            ax.scatter(
                shuffled_points[:, 0], shuffled_points[:, 1], c=assignments, alpha=0.5
            )
            ax.set_title(f"K-means (k={i})")

            # save for drawing on k vs sse
            k_vs_sse_data.append((i, total_sq_dist))

        axes_ksse.plot(
            [d[0] for d in k_vs_sse_data],
            [d[1] for d in k_vs_sse_data],
            marker="o",
            linestyle="-",
        )
        axes_ksse.set_title("k vs Sum of Squared Error (SSE)")

    fig_render.tight_layout()
    fig_ksse.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
