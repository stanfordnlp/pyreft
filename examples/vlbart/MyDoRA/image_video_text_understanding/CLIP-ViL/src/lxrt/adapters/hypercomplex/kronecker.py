# The codes are from https://github.com/bayer-science-for-a-better-life/phc-gnn
import torch

# TODO: change this with torch.kron
"""A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk"""
def kronecker_product(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    #return torch.stack([torch.kron(ai, bi) for ai, bi in zip(a,b)], dim=0)
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out


def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3
    res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0),
                                                    A.size(1)*B.size(1),
                                                    A.size(2)*B.size(2))
    return res