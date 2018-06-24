
def one_hot(ids, out_tensor):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    ids = torch.LongTensor(ids)
    out_tensor.zero_()
    return out_tensor.scatter_(dim=1, index=ids, value=1)
    # out_tensor.scatter_(1, ids, 1.0)
