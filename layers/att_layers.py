"""Attention layers (some modules are copied from https://github.com/Diego999/pyGAT."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout, act):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.act = act
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        # n x 1 x d
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        # 1 x n x d
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)

        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = F.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.act = activation

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj._indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return self.act(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activation, alpha, nheads, concat):
        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.output_dim = output_dim
        self.attentions = [SpGraphAttentionLayer(input_dim,
                                                 output_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 activation=activation) for _ in range(nheads)]
        self.concat = concat
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, input):
        x, adj = input
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
            h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            h_cat = torch.cat([att(x, adj).view((-1, self.output_dim, 1)) for att in self.attentions], dim=2)
            h = torch.mean(h_cat, dim=2)
        h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)
