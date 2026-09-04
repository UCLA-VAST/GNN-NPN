import os
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.action import sample_npn
from src.mockturtle import get_action_fn
from src.rollout import load_tt, map_actions, Rollout

class UnionFind:
    def __init__(self, init_tts):
        self.parent = {tt: tt for tt in init_tts}

    def find(self, tt):
        if tt not in self.parent:
            self.parent[tt] = tt
            return tt

        root = tt
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[tt] != root:
            self.parent[tt], tt = root, self.parent[tt]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        elif rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra
        return True

def collect_unique_truth_tables(dataset):
    start = time.perf_counter()
    first_index = {}
    for index in tqdm(range(len(dataset)), desc="Collecting unique truth tables"):
        filename = os.path.join(dataset.root, dataset.filenames[index])
        tt = load_tt(filename)
        if tt not in first_index:
            first_index[tt] = index
    elapsed = time.perf_counter() - start
    print(f"Found {len(first_index)} unique truth tables out of {len(dataset)} in {elapsed:.2f}s.")
    unique_tts, first_indices = zip(*sorted(first_index.items(), key=lambda kv: kv[1])) if first_index else ([], [])
    return list(unique_tts), list(first_indices)

def run_npn(
        unique_tts,
        in_neg_logits, in_perm_logits, out_neg_logits,
        trial, temp, post_proc
    ):
    n = len(unique_tts)
    uf = UnionFind(unique_tts)
    num_bit = in_neg_logits.shape[0] // n
    index = torch.arange(n).repeat_interleave(num_bit)
    post_proc_fn = get_action_fn(post_proc)
    rollouts = [Rollout.from_tt_only(tt, post_proc_fn=post_proc_fn) for tt in unique_tts]
    def reached_set(rollout, pairs):
        return {tt for pair in pairs for tt in pair}

    start = time.perf_counter()
    in_neg, in_perm, out_neg = sample_npn(
        in_neg_logits, in_perm_logits, out_neg_logits, index,
        trial, temp
    )
    elapsed = time.perf_counter() - start
    print(f"Sampled {len(out_neg)} NPN actions in {elapsed:.2f}s.")

    start = time.perf_counter()
    reached_per_tt = map_actions(
        rollouts, in_neg, in_perm, out_neg,
        num_pis=[num_bit] * n, reduce_fn=reached_set,
        desc="Building NPN classes"
    )
    for tt, reached in zip(unique_tts, reached_per_tt):
        for tt_other in reached:
            uf.union(tt, tt_other)
    elapsed = time.perf_counter() - start
    num_classes = len(set(uf.find(tt) for tt in unique_tts))
    print(f"Identified {num_classes} NPN classes from {n} unique truth tables in {elapsed:.2f}s.")
    return num_classes
