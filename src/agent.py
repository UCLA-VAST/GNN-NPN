import time
from tqdm import tqdm

import torch
from torch import nn

from src.action import sample_npn, logprob_and_entropy_npn
from src.deepgate import Model
from src.rollout import collate_rollouts

class Agent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = Model(
            num_rounds=cfg.num_rounds,
            dim_hidden=cfg.dim_hidden,
            enable_reverse=cfg.enable_reverse,
            enable_layernorm=cfg.enable_layernorm
        )

    def get_logits(self, x):
        node_state = self.model(x)
        pi_emb = node_state[x.PIs]
        po_emb = node_state[x.POs]
        index = x.batch[x.PIs]
        in_neg_logits = self.model.pred_in_neg(pi_emb).flatten()
        in_perm_logits = self.model.pred_in_perm(pi_emb).flatten()
        out_neg_logits = self.model.pred_out_neg(po_emb).flatten()
        return in_neg_logits, in_perm_logits, out_neg_logits, index

    @torch.no_grad()
    def act(self, x, group_size: int = 1, temp: float = 1.0):
        in_neg_logits, in_perm_logits, out_neg_logits, index = self.get_logits(x)
        in_neg, in_perm, out_neg = sample_npn(
            in_neg_logits, in_perm_logits, out_neg_logits, index,
            group_size=group_size, temp=temp
        )
        return in_neg, in_perm, out_neg

    def forward(self, x, in_neg, in_perm, out_neg, group_size: int = 1, temp: float = 1.0):
        in_neg_logit, in_perm_logit, out_neg_logit, index = self.get_logits(x)
        logprob, entropy = logprob_and_entropy_npn(
            in_neg_logit, in_perm_logit, out_neg_logit, index,
            in_neg, in_perm, out_neg,
            group_size=group_size, temp=temp
        )
        return logprob, entropy

@torch.inference_mode()
def run_inference(agent, dataset, device, batch_size: int):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_rollouts,
        num_workers=8
    )
    track_mem = torch.cuda.is_available() and torch.device(device).type == "cuda"
    if track_mem:
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    in_neg_logits_list, in_perm_logits_list, out_neg_logits_list = [], [], []
    for batch in tqdm(dataloader, desc="Running inference"):
        batch = batch.to(device)
        in_neg_logits, in_perm_logits, out_neg_logits, _ = agent.get_logits(batch)
        in_neg_logits_list.append(in_neg_logits)
        in_perm_logits_list.append(in_perm_logits)
        out_neg_logits_list.append(out_neg_logits)
    in_neg_logits_all = torch.cat(in_neg_logits_list, dim=0)
    in_perm_logits_all = torch.cat(in_perm_logits_list, dim=0)
    out_neg_logits_all = torch.cat(out_neg_logits_list, dim=0)
    if track_mem:
        peak_bytes = torch.cuda.max_memory_allocated(device)
        print(f"Max GPU memory allocated during inference: {peak_bytes / 1024 ** 2:.1f} MiB")
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    return in_neg_logits_all, in_perm_logits_all, out_neg_logits_all

if __name__ == "__main__":
    # Example usage of Agent
    class CFG:
        num_rounds = 1
        dim_hidden = 16
        enable_reverse = True
        enable_layernorm = True
    agent = Agent(CFG()).eval()
    print("Agent initialized successfully.")
    num_params = sum(p.numel() for p in agent.parameters())
    print(f"#parameters: {num_params}")

    import os
    from src.rollout import Rollout, collate_rollouts
    filenames = [
        "examples/example_6bit.aig",
        "examples/example_8bit.aig"
    ]
    rollouts = [Rollout(filename) for filename in filenames]
    print("Rollouts initialized successfully.")
    batch = collate_rollouts(rollouts)
    print("Batch created successfully.")
    print(batch)

    in_neg, in_perm, out_neg = agent.act(batch, group_size=2, temp=0.5)
    print("Agent action taken successfully.")
    print("Input negation:", in_neg)
    print("Permutation:", in_perm)
    print("Output negation:", out_neg)

    logprob, entropy = agent(batch, in_neg, in_perm, out_neg, group_size=2, temp=0.5)
    print("Agent forward pass successful.")
    print("Log probability:", logprob)
    print("Entropy:", entropy)
