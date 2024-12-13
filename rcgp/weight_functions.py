import torch as tc

def IMQ(Y : tc.Tensor, m : tc.Tensor, beta : float, c : float):
    assert Y.shape == m.shape, f"Y has shape {Y.shape} and m has shape {m.shape}. They must be of the same size."
    return beta * ( 1 + ((Y - m) / c)**2 )**(-1/2)

def partial_y_IMQ(Y : tc.Tensor, m : tc.Tensor, beta : float, c : float):
    assert Y.shape == m.shape, f"Y has shape {Y.shape} and m has shape {m.shape}. They must be of the same size."
    return -beta * ( 1 + ( (Y - m) / c )**2 )**(-3/2) * (Y - m) / c