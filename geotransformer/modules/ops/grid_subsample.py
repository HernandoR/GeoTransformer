import importlib

import open3d.ml.torch as ml3d
import torch

# ext_module = importlib.import_module("geotransformer.ext")


# def grid_subsample(points, lengths, voxel_size):
#     """Grid subsampling in stack mode.

#     This function is implemented on CPU.

#     Args:
#         points (Tensor): stacked points. (N, 3)
#         lengths (Tensor): number of points in the stacked batch. (B,)
#         voxel_size (float): voxel size.

#     Returns:
#         s_points (Tensor): stacked subsampled points (M, 3)
#         s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
#     """
#     s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
#     return s_points, s_lengths


def grid_subsample_o3d(points, lengths, voxel_size):
    point_row_splits = torch.cat(
        (
            torch.tensor([0], device=points.device, dtype=torch.int64),
            torch.cumsum(lengths, dim=0, dtype=torch.int64),
        ),
    )
    s_points = []
    s_lengths = []
    for batch_idx in range(len(lengths)):
        start = point_row_splits[batch_idx]
        end = point_row_splits[batch_idx + 1]
        points_batch = points[start:end]
        features = torch.ones((points_batch.shape[0], 1), device=points.device)
        (
            voxel_centers,
            features,
        ) = ml3d.ops.voxel_pooling(
            points_batch,
            features,
            voxel_size,
            position_fn="average",
            feature_fn="average",
        )
        s_points.append(voxel_centers)
        s_lengths.append(voxel_centers.shape[0])
    s_points = torch.cat(s_points, dim=0)
    s_lengths = torch.tensor(s_lengths, device=points.device)
    return s_points, s_lengths

@PendingDeprecationWarning
def grid_subsample_o3d_complex(points, lengths, voxel_size):
    """Grid subsampling in stack mode using open3d.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    voxel_size = torch.tensor([voxel_size] * 3, device=points.device)
    DEVICE = points.device
    row_splits = torch.cat(
        (
            torch.tensor([0], device=DEVICE, dtype=torch.int64),
            torch.cumsum(lengths, dim=0, dtype=torch.int64),
        ),
    )
    points_range_min = points.min(dim=0).values
    points_range_max = points.max(dim=0).values
    (
        voxel_cord,  # Tensor: (M, 3)
        point_indices,  # Tensor: (N,)
        voxel_point_row_splits,  # Tensor: (M+1,)
        batch_splits,  # Tensor: (B+1,)
    ) = ml3d.ops.voxelize(
        points=points,
        row_splits=row_splits,
        voxel_size=voxel_size,
        points_range_min=points_range_min,
        points_range_max=points_range_max,
    )

    s_points = []
    s_lengths = []

    # Calculate mean of points in each voxel using vectorized operations
    for i in range(len(batch_splits) - 1):
        start = batch_splits[i]
        end = batch_splits[i + 1]
        # voxels = voxel_cord[start:end]
        # for voxel_idx, _ in enumerate(voxels):
        for voxel_idx in range(start, end):
            points_start = voxel_point_row_splits[voxel_idx]
            points_end = voxel_point_row_splits[voxel_idx + 1]
            points_indice = point_indices[points_start:points_end]
            s_points.append(points[points_indice].mean(dim=0))
        s_lengths.append(end - start)

    s_points = torch.stack(s_points, dim=0)
    s_lengths = torch.tensor(s_lengths, device=DEVICE)
    return s_points, s_lengths


if __name__ == "__main__":
    import time

    N = 23_0000
    SPLIT = 3

    # split_points = torch.randint(1, N, (SPLIT,)).sort()
    q_points = torch.rand(N, 3).cpu()
    q_lengths = torch.tensor([30000, 8_0000, 12_0000], device=q_points.device)

    print(q_lengths)

    voxel_size = 0.1
    t0 = time.time()

    ans1, _ = grid_subsample(q_points, q_lengths, voxel_size)
    print(f"ans1_done")
    t1 = time.time()
    ans2, _ = grid_subsample_o3d(q_points, q_lengths, voxel_size)
    t2 = time.time()
    # sort ans1

    list_ans1 = ans1.tolist()
    # sort based on x,y,z
    list_ans1.sort(key=lambda x: (x[0], x[1], x[2]))
    ans1 = torch.tensor(list_ans1, device=ans1.device)

    list_ans2 = ans2.tolist()
    # sort based on x,y,z
    list_ans2.sort(key=lambda x: (x[0], x[1], x[2]))
    ans2 = torch.tensor(list_ans2, device=ans2.device)

    print(f"radius_search: {t1 - t0:.3f}s, radius_search_ml3d: {t2 - t1:.3f}s")

    torch.testing.assert_allclose(ans1, ans2)
    pass
    # radius_search: 13.205s, radius_search_ml3d: 3.804s
