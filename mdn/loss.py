import torch


def mdn_logloss_base(pi, mu, sigma, actual, dist=torch.distributions.Normal):
    dist = dist(mu, sigma)
    return -(pi * dist.log_prob(actual).sum(dim=-1)).sum(dim=-1)


def mdn_logloss_laplace(pi, mu, sigma, actual):
    return mdn_logloss_base(pi, mu, sigma, dist=torch.distributions.Laplace)


def mdn_logloss(pi, mu, sigma, actual):
    return mdn_logloss_base(pi, mu, sigma)


def mdnpos_losloss(item, actual, dist=torch.distributions.Normal):
    return mdn_logloss_base(item.pi, item.mu, item.sigma, actual, dist=dist)



# zinb and nb loss are derived from
# scVI, accessible here:
# https://github.com/YosefLab/scVI/blob/master/scvi/models/log_likelihood.py
# under the MIT license.


EPS=1e-8

def zinb_loss(x, mu, theta, pi, eps=EPS):
    if theta.ndim() == 1:
        theta = theta.unsqueeze(0)
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)
    res = mul_case_zero + mul_case_non_zero
    return torch.sum(res, dim=1)


def nb_loss(x, mu, theta, eps=EPS):
    """
    log likelihood (scalar) of a minibatch according to a nb model.
    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    if theta.ndimension() == 1:
        theta = theta.unsqueeze(0);

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return torch.sum(res, dim=-1)

__all__ = [mdn_logloss, mdnpos_losloss, zinb_loss, nb_loss]
