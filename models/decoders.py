"""Graph decoders."""
import manifolds
import torch.nn as nn
import torch.nn.functional as F

from models.att_layers import GraphAttentionLayer
from models.layers import GraphConvolution, Linear


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        raise NotImplementedError


class GCN(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        act = lambda x: x
        self.gc_cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)

    def decode(self, h, adj):
        input = (h, adj)
        probs, _ = self.gc_cls.forward(input)
        return probs


class GAT(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        act = F.elu
        n_heads = 1
        self.gat_cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, act, args.alpha, n_heads,
                                           concat=False)

    def decode(self, h, adj):
        input = (h, adj)
        output = self.gat_cls.forward(input)
        probs, _ = output
        return probs


class LinearClassifier(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearClassifier, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        act = lambda x: x
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, act, self.bias)

    def decode(self, x, adj):
        h = self.manifold.logmap0(x, c=self.c)
        probs = self.cls.forward(h)
        return probs

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


class HNN(LinearClassifier):
    """
    Hyperbolic Neural Networks Euclidean classifier.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c, args)


class HyperGCN(LinearClassifier):
    """
    Hyperbolic Neural Networks Euclidean classifier.
    """

    def __init__(self, c, args):
        super(HyperGCN, self).__init__(c, args)


class HGCN(LinearClassifier):
    """
    Linear classifier on top of pre-trained shallow embeddings.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c, args)


class MLP(LinearClassifier):
    """
    Hyperbolic Neural Networks Euclidean classifier.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c, args)


class Shallow(LinearClassifier):
    """
    Linear classifier on top of pre-trained shallow embeddings.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c, args)
