import torch
from manifolds.manifold import Manifold
from utils.math_utils import artanh, arcosh, arsinh, cosh, sinh, tanh


class Hyperboloid(Manifold):
    """
    Hyperboloid Manifold class.
    
    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -c

    Note that isqrt(c) is the hyperboloid radius.

    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = 1e-12
        self.norm_clip = 1
        self.max_norm = 1e6

    def init_weights(self, w, c, irange=1):
        w.data.uniform_(-irange, irange)
        w.data.copy_(self.proj(w.data, c))
        return w

    def minkowski_product(self, x, y, keepdim=True):
        res = torch.sum(x * y, -1, keepdim=keepdim) - 2 * x[..., 0:1] * y[..., 0:1]
        return res

    def minkowski_norm(self, u, keepdim=True):
        u_prod = self.minkowski_product(u, u, keepdim)
        u_norm = torch.sqrt(torch.clamp(u_prod, min=0))
        u_norm = torch.clamp(torch.clamp(u_norm, max=self.max_norm), min=self.eps)
        return u_norm

    def dist(self, x, y, c):
        prod = self.minkowski_product(x, y)
        theta = - (torch.clamp(prod + c, max=-self.eps) - c) / c
        dist = (c ** 0.5) * arcosh(theta)
        return dist

    def egrad2rgrad(self, x, euc_grad, c):
        """Riemannian gradient for hyperboloid"""
        raise NotImplementedError

    def proj(self, w, c):
        """Normalize vector such that it is located on the hyperboloid"""
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed = narrowed.view(-1, d).renorm(p=2, dim=0, maxnorm=self.max_norm)
        tmp = torch.sqrt(c + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True))
        mask = torch.ones_like(w)
        mask[:, 0] = 0
        vals = torch.zeros_like(w)
        vals[:, 0:1] = tmp
        w = vals + mask * w
        return w

    def proj_tan(self, u, x, c):
        return u.addcmul((self.minkowski_product(x, u)).expand_as(x), x / c)

    def proj_tan0(self, u, c):
        mask = torch.zeros_like(u)
        mask[:, 0] += 1.
        return u - u * mask

    def expmap(self, u, x, c):
        sqrt_c = c ** 0.5
        u_norm = self.minkowski_norm(u)
        theta = torch.clamp(u_norm / sqrt_c, min=1. + self.eps)
        return (cosh(theta) * x).addcdiv(sqrt_c * sinh(theta) * u, u_norm)

    def logmap(self, x, y, c):
        xy = torch.clamp(self.minkowski_product(x, y) + c, max=-self.eps) - c
        dist = self.dist(x, y, c)
        v = dist.div(
            torch.clamp(torch.sqrt(xy * xy - 1), min=self.eps)
        ) * torch.addcmul(y, xy / c, x)
        return v

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        theta = self.minkowski_norm(u) / sqrt_c
        res = torch.zeros_like(u)
        res[:, 0:1] = sqrt_c * cosh(theta)
        res[:, 1:] = sqrt_c * sinh(theta) * u[:, 1:] / torch.sqrt(torch.sum(u ** 2, -1, keepdim=True))
        return res

    def logmap0(self, x, c):
        sqrt_c = c ** 0.5
        mask = torch.zeros_like(x)
        mask[:, 0] = 1.
        v = x - x * mask
        v_norm = v / torch.clamp(torch.sqrt(torch.sum(v[:, 1:] ** 2, -1, keepdim=True)), min=self.eps)
        return sqrt_c * arcosh(x[:, 0:1] / sqrt_c) * v_norm

    def mobius_add(self, x, y, c):
        sqrt_c = c ** 0.5
        x1 = x[:, 1:]
        y1 = y[:, 1:]
        x1_norm = x1 / torch.norm(x1, p=2, dim=-1)
        y1_norm = y1 / torch.norm(y1, p=2, dim=-1)
        prod = torch.sum(x1_norm * y1_norm, -1)
        ptrans_y = torch.zeros_like(x)
        ptrans_y[:, 0:1] = - (c - x[:, 0] ** 2) * prod / torch.norm(x1, p=2, dim=-1, keepdim=True)
        ptrans_y[:, -1] = sqrt_c * y1_norm - (sqrt_c - x[:, 0]) * prod * x1_norm
        ptrans_y = ptrans_y * arcosh(y[:, 0] / sqrt_c)
        return self.expmap(ptrans_y, x, c)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x1 = x[:, 1:]
        x1_norm = torch.clamp(torch.norm(x1, p=2, dim=-1, keepdim=True), min=self.eps)
        Mx1 = torch.matmul(x1, m[1:, 1:].transpose(-1, -2))
        Mx1_norm = torch.clamp(torch.norm(Mx1, p=2, dim=-1, keepdim=True), min=self.eps)
        theta = arcosh(x[:, 0:1] / sqrt_c) * torch.norm(Mx1, p=2, dim=-1, keepdim=True) / x1_norm
        res = torch.zeros((x.shape[0], m.shape[0])).type_as(x)
        res[:, 0:1] = sqrt_c * cosh(theta)
        res[:, 1:] = sqrt_c * sinh(theta) * Mx1 / Mx1_norm
        return res

    def add_euc_bias(self, x, b, c):
        sqrt_c = c ** 0.5
        x1 = x[:, 1:]
        x1_norm = torch.norm(x1, p=2, dim=-1, keepdim=True)
        b1 = b[1:]
        ptrans_b = torch.zeros_like(x)
        coeff = torch.sum(x1 * b1, 1, keepdim=True) * (1 - (x[:, 0:1]/sqrt_c)) / torch.clamp(x1_norm ** 2, min=self.eps)
        ptrans_b[:, 0:1] = - coeff * (x[:, 0:1] + sqrt_c)
        ptrans_b[:, 1:] = b1 - coeff * x1
        # return ptrans_b
        return self.expmap(ptrans_b, x, c)
        # return self.proj(self.expmap(self.proj_tan(ptrans_b, x, c), x, c), c)

    def ptransp(self, x, y, w, c):
        """Parallel transport for hyperboloid"""
        u = self.logmap(x, y, c)
        v = self.logmap(y, x, c)
        dist = self.dist(x, y, c)
        return w - (self.minkowski_product(u, w) / dist ** 2) * (u + v)
