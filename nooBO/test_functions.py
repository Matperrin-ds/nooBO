import torch

def langermann(x):
    m = 5
    c = torch.tensor([1, 2, 5, 2, 3], dtype=torch.float32)
    A = torch.tensor([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]], dtype=torch.float32)
    
    result = 0
    for i in range(m):
        dist_sq = torch.sum((x - A[i, :])**2, axis=-1)
        result += c[i] * torch.exp(-dist_sq / torch.pi) * torch.cos(torch.pi * dist_sq)

    # Ensure the output is a 2D tensor with shape (n, 1)
    return torch.atleast_2d(result.unsqueeze(-1))

def eggholder(x):
    # Ensure x is a 2D tensor with shape (n, 2)
    assert x.shape[1] == 2, "Input tensor must have shape (n, 2) for two variables x and y."
    
    x1 = x[:, 0]  # Extract the first column (x)
    x2 = x[:, 1]  # Extract the second column (y)
    
    term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))
    
    result = term1 + term2
    
    # Ensure the output is a 2D tensor with shape (n, 1)
    return torch.atleast_2d(result.unsqueeze(-1))  