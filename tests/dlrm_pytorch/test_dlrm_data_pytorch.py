import torch


def test_collate_wrapper_criteo_offset_to_understand_it():
    """
    The output after running the test is:

    Output tensor 0 (x_int)
    tensor([[2.3979, 4.6151, 6.9088],
            [3.0445, 5.3033, 7.6014],
            [3.4340, 5.7071, 8.0067]])
    Output tensor 1 (sparse_features_offsets)
    tensor([[0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]])
    Output tensor 2 (sparse_features_indices)
    tensor([[  1,   2,   3],
            [ 11,  22,  33],
            [111, 222, 333]])
    Output tensor 3 (T)
    tensor([[0.],
            [1.],
            [0.]])
    """
    x_int = [[10, 100, 1000], [20, 200, 2000], [30, 300, 3000]]
    x_cat = [[1, 11, 111], [2, 22, 222], [3, 33, 333]]
    y = [0.0, 1.0, 0.0]
    inputs = [
        (x_int[0], x_cat[0], y[0]),
        (x_int[1], x_cat[1], y[1]),
        (x_int[2], x_cat[2], y[2]),
    ]

    outputs = collate_wrapper_criteo_offset(inputs)

    print()
    for i, output in enumerate(outputs):
        print(f"Output tensor {i}")
        print(output)


def collate_wrapper_criteo_offset(list_of_tuples):
    """Copied from "dlrm_data_pytorch"."""
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    sparse_features_indices = [X_cat[:, i] for i in range(featureCnt)]
    sparse_features_offsets = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

    return X_int, torch.stack(sparse_features_offsets), torch.stack(sparse_features_indices), T
