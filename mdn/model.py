import torch
from torch import nn, functional as F
from collections import namedtuple
from mdn import loss

MDNPos = namedtuple("MDNPos", ["pi", "mu", "sigma"])



class MDN(nn.Module):
    def __init__(self, inf, outf, ncomp, beta=1., threshold=20):
        super(self).__init__()
        self.inf = inf
        self.outf=outf
        self.ncomp = ncomp
        self.pi = nn.Sequential(
            nn.Linear(inf, ncomp, bias=True),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Sequential(
            nn.Linear(inf, outf * ncomp, bias=True),
            nn.Softplus(beta=beta, threshold=threshold)
        )
        self.mu = nn.Linear(inf, ncomp, bias=True)

    def forward(self, inp):
        pi = self.pi(inp)
        mu = self.mu(inp).view(-1, self.ncomp, self.outf)
        sigma = self.sigma(inp).view(-1, self.ncomp, self.outf)


class GenericMixtureNetwork(nn.Module):
    def __init__(self, inf, ncomp, extra_layers=[]):
        self.extra_layers = extra_layers
        super(self).__init__()
        self.inf = inf
        self.ncomp = ncomp
        self.pi = nn.Sequential(nn.Linear(inf, outf * ncomp), self.Softmax(dim=1))

    def __init__(self, inp):
        pi = self.pi(inp)
        if self.extra_layers:
            for el in self.extra_layers:
                inp = el(inp)
        return pi, inp


__all__ = [mdn_logloss, mdnpos_losloss, GenericMixtureNetwork, MDN, MDNPos]
