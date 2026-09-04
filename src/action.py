import torch
from torch_geometric.utils import scatter, group_argsort
from src.utils import group_index, group_gather, group_to_dense

EPS = 1e-8

def sample_neg_stochastic(logits):
    probs = torch.sigmoid(logits)
    neg = torch.bernoulli(probs).long()
    return neg

def logprob_and_entropy_neg(logits, neg):
    probs = torch.sigmoid(logits)
    neg = neg.flatten()
    logprob = torch.log(probs + EPS) * neg + torch.log(1 - probs + EPS) * (1 - neg)
    entropy = -(probs * torch.log(probs + EPS) + (1 - probs) * torch.log(1 - probs + EPS))
    return logprob, entropy

def sample_perm_stochastic(logits, index):
    # Apply the Gumbel-Softmax trick for sampling permutations
    gumbel = torch.rand_like(logits).clamp(EPS, 1 - EPS)
    gumbel = -torch.log(-torch.log(gumbel))
    logits = logits + gumbel
    perm = group_argsort(logits, index)
    return perm

def logprob_and_entropy_perm(logits, index, perm):
    logits_sorted = group_gather(logits, index, perm.flatten())
    dense, mask = group_to_dense(logits_sorted, index)

    # Compute the log probabilities using the Plackett-Luce model.
    ell = torch.logcumsumexp(dense, dim=1)
    logprob = (dense - ell)[mask]

    # Entropy of the sampled permutation, not of the full PL distribution
    # This is a Rao-Blackwellized estimator
    width = dense.size(1)
    tril = torch.ones(width, width, dtype=torch.bool, device=dense.device).tril()
    valid = tril & mask.unsqueeze(1)
    B = ell.unsqueeze(2) - dense.masked_fill(~mask, 0.0).unsqueeze(1)
    P = torch.exp((-B).masked_fill(~valid, float("-inf")))
    entropy = (P * B.masked_fill(~valid, 0.0)).sum(dim=2)[mask]

    return logprob, entropy

def sample_npn(in_neg_logits, in_perm_logits, out_neg_logits, index, group_size: int = 1, temp: float = 1.0):
    if temp > EPS:
        in_neg_logits = (in_neg_logits / temp).repeat(group_size)
        in_perm_logits = (in_perm_logits / temp).repeat(group_size)
        out_neg_logits = (out_neg_logits / temp).repeat(group_size)
        index = group_index(index, group_size)

        in_neg = sample_neg_stochastic(in_neg_logits).reshape(group_size, -1)
        in_perm = sample_perm_stochastic(in_perm_logits, index).reshape(group_size, -1)
        out_neg = sample_neg_stochastic(out_neg_logits).reshape(group_size, -1)
    else:
        in_neg = (in_neg_logits > 0).long().unsqueeze(0)
        in_perm = group_argsort(in_perm_logits, index).unsqueeze(0)
        out_neg = (out_neg_logits > 0).long().unsqueeze(0)
    return in_neg, in_perm, out_neg

def logprob_and_entropy_npn(in_neg_logits, in_perm_logits, out_neg_logits, index, in_neg, in_perm, out_neg, group_size: int = 1, temp: float = 1.0):
    assert temp > EPS, "Temperature must be greater than EPS to compute log probabilities and entropy."
    in_neg_logits = (in_neg_logits / temp).repeat(group_size)
    in_perm_logits = (in_perm_logits / temp).repeat(group_size)
    out_neg_logits = (out_neg_logits / temp).repeat(group_size)
    index = group_index(index, group_size)

    in_neg_logprob, in_neg_entropy = logprob_and_entropy_neg(in_neg_logits, in_neg)
    in_neg_logprob = scatter(in_neg_logprob, index, dim=0, reduce="sum").reshape(group_size, -1)
    in_neg_entropy = scatter(in_neg_entropy, index, dim=0, reduce="sum").reshape(group_size, -1)

    in_perm_logprob, in_perm_entropy = logprob_and_entropy_perm(in_perm_logits, index, in_perm)
    in_perm_logprob = scatter(in_perm_logprob, index, dim=0, reduce="sum").reshape(group_size, -1)
    in_perm_entropy = scatter(in_perm_entropy, index, dim=0, reduce="sum").reshape(group_size, -1)

    out_neg_logprob, out_neg_entropy = logprob_and_entropy_neg(out_neg_logits, out_neg)
    out_neg_logprob = out_neg_logprob.reshape(group_size, -1)
    out_neg_entropy = out_neg_entropy.reshape(group_size, -1)

    logprob = in_neg_logprob + in_perm_logprob + out_neg_logprob
    entropy = in_neg_entropy + in_perm_entropy + out_neg_entropy
    return logprob, entropy

if __name__ == "__main__":
    # Example usage
    in_logits = torch.tensor([2.0, -2.0, 1.0, 0.0, -1.0])
    out_logits = torch.tensor([1.0, -1.0])
    index = torch.tensor([0, 0, 1, 1, 1])
    group_size = 2
    temp = 0.5

    in_neg_det, in_perm_det, out_neg_det = sample_npn(in_logits, in_logits, out_logits, index, group_size, 0.0)
    print("Deterministic NPN Sample:")
    print(" in_neg:\n", in_neg_det)
    print(" in_perm:\n", in_perm_det)
    print(" out_neg:\n", out_neg_det)

    in_neg, in_perm, out_neg = sample_npn(in_logits, in_logits, out_logits, index, group_size, temp)
    print("Stochastic NPN Sample:")
    print(" in_neg:\n", in_neg)
    print(" in_perm:\n", in_perm)
    print(" out_neg:\n", out_neg)

    logprob_npn, entropy_npn = logprob_and_entropy_npn(in_logits, in_logits, out_logits, index, in_neg, in_perm, out_neg, group_size, temp)
    print("Log Probability:\n", logprob_npn)
    print("Entropy:\n", entropy_npn)
