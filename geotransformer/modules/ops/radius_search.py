import importlib

import open3d.ml.torch as ml3d
import torch

# ext_module = importlib.import_module("geotransformer.ext")


# def radius_search(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
#     r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

#     This function is implemented on CPU.

#     Args:
#         q_points (Tensor): the query points (N, 3)
#         s_points (Tensor): the support points (M, 3)
#         q_lengths (Tensor): the list of lengths of batch elements in q_points
#         s_lengths (Tensor): the list of lengths of batch elements in s_points
#         radius (float): maximum distance of neighbors
#         neighbor_limit (int): maximum number of neighbors

#     Returns:
#         neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
#             Filled with M if there are less than k neighbors.
#     """

#     # make tensor to cpu
#     q_points = q_points.cpu()
#     s_points = s_points.cpu()
#     q_lengths = q_lengths.cpu()
#     s_lengths = s_lengths.cpu()

#     neighbor_indices = ext_module.radius_neighbors(
#         q_points, s_points, q_lengths, s_lengths, radius
#     )
#     if neighbor_limit > 0:
#         neighbor_indices = neighbor_indices[:, :neighbor_limit]
#     return neighbor_indices
#     # knn=KNN(q_points,s_points,neighbor_limit)
#     # return knn(q_points)


def radius_search_ml3d(
    q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit
):
    r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

    This function is implemented on CPU.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    # assert the sum of lengths is equal to the number of points
    assert q_points.shape[0] == q_lengths.sum()  # N
    assert s_points.shape[0] == s_lengths.sum()  # M
    N = q_points.shape[0]
    M = s_points.shape[0]

    # make all tensor to the default device
    # DEFAULT_DIVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if q_points.device != DEFAULT_DIVICE:
    #     q_points = q_points.to(DEFAULT_DIVICE)
    # if s_points.device != DEFAULT_DIVICE:
    #     s_points = s_points.to(DEFAULT_DIVICE)
    # if s_lengths.device != DEFAULT_DIVICE:
    #     s_lengths = s_lengths.to(DEFAULT_DIVICE)
    # if q_lengths.device != DEFAULT_DIVICE:
    #     q_lengths = q_lengths.to(DEFAULT_DIVICE)

    # points_row_splits is pre-sum of lengths
    s_points_row_splits = torch.cat(
        [
            torch.tensor([0], device=s_lengths.device),
            torch.cumsum(s_lengths, dim=0).to(s_lengths.device),
        ]
    )
    q_points_row_splits = torch.cat(
        [
            torch.tensor([0], device=q_lengths.device),
            torch.cumsum(q_lengths, dim=0).to(q_lengths.device),
        ]
    )

    table = ml3d.ops.build_spatial_hash_table(
        s_points,
        radius,
        points_row_splits=s_points_row_splits,
        hash_table_size_factor=1 / 32,
    )

    # now run the fixed radius search
    (
        neighbors_index,
        neighbors_row_splits,
        neighbors_distance,
    ) = ml3d.ops.fixed_radius_search(
        s_points,
        q_points,
        radius,
        points_row_splits=s_points_row_splits,
        queries_row_splits=q_points_row_splits,
        return_distances=True,
        **table._asdict(),
    )
    neighbors_index = neighbors_index.to(torch.int64)
    neighbors_row_splits = neighbors_row_splits.to(torch.int64)
    neighbors_distance = neighbors_distance.to(torch.float64)

    # if neighbor_limit <= 0:
    #     neighbor_limit = torch.max(neighbors_row_splits[1:] - neighbors_row_splits[:-1])
    neighbor_cnt = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
    neighbor_limit = min(neighbor_limit, torch.max(neighbor_cnt).int().item())

    neighbor_matrix = []
    for i in range(N):
        start_idx = neighbors_row_splits[i]
        end_idx = neighbors_row_splits[i + 1]
        if end_idx - start_idx < neighbor_limit:
            # fill with int64(M)
            neighbor_mtx_i = torch.cat(
                [
                    neighbors_index[start_idx:end_idx],
                    torch.full(
                        (neighbor_limit - (end_idx - start_idx),),
                        M,
                        dtype=torch.int64,
                        device=neighbors_index.device,
                    ),
                ]
            )
        elif end_idx - start_idx >= neighbor_limit:
            # sort by distance
            _, sort_idx = torch.sort(neighbors_distance[start_idx:end_idx])
            neighbor_mtx_i = neighbors_index[start_idx:end_idx][
                sort_idx[:neighbor_limit]
            ]
        else:
            raise Exception(f"all posible should have been exhausted, please check")

        neighbor_matrix.append(neighbor_mtx_i)
    neighbor_matrix = torch.stack(neighbor_matrix)

    return neighbor_matrix


if __name__ == "__main__":
    import time

    q_points = torch.rand(23_000, 3)
    s_points = torch.rand(30_000, 3)
    q_lengths = torch.tensor([10_000, 13_000])
    s_lengths = torch.tensor([15_000, 15_000])
    radius = 1
    neighbor_limit = 20
    t0 = time.time()
    ans1 = radius_search(
        q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit
    )
    t1 = time.time()
    ans2 = radius_search_ml3d(
        q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit
    )
    t2 = time.time()
    ans2 = ans2.to(ans1.device)
    torch.testing.assert_allclose(ans1, ans2)
    # radius_search: 13.205s, radius_search_ml3d: 3.804s
    print(f"radius_search: {t1 - t0:.3f}s, radius_search_ml3d: {t2 - t1:.3f}s")
