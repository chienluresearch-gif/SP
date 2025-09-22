import torch
import torch.nn as nn
import numpy as np


def sigma_estimation(X, Y):
    """sigma from median distance"""
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1e-2:
        med = 1e-2
    return med


def distmat(X):
    """distance matrix"""
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def kernelmat(X, sigma):
    """kernel matrix baker"""
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])
    Dxx = distmat(X)

    if sigma:
        variance = 2.0 * sigma * sigma * X.size()[1]
        Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    else:
        try:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2.0 * sx * sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError(
                "Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)
                )
            )

    Kxc = torch.mm(Kx, H)

    return Kxc


def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp(-X / (2.0 * sigma * sigma))
    return torch.mean(X)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def hsic_regular(x, y, sigma=None):
    """ """
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy

def hsic_normalized(x, y, sigma=None, to_numpy=True):
    """ """
    device = x.device
    y = y.to(device)
    
    Pxy = hsic_regular(x, y, sigma)
    Px = torch.sqrt(hsic_regular(x, x, sigma))
    Py = torch.sqrt(hsic_regular(y, y, sigma))
    thehsic = Pxy / (Px * Py)
    return thehsic

def mmd(x, y, kernel="rbf"):
    """Empirical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device  # Get the device of x (assuming x and y are on the same device)
    
    # Move tensors to the same device as x and y
    xx, yy, zz = torch.mm(x, x.t()).to(device), torch.mm(y, y.t()).to(device), torch.mm(x, y.t()).to(device)
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape, device=device),
                  torch.zeros(xx.shape, device=device),
                  torch.zeros(xx.shape, device=device))
    
    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)