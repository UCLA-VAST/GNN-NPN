import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from src.deepgate import AigParser
from src.mockturtle import aig_to_tt, apply_to_tt

PARSER = AigParser()

def tt_to_float(tt: str) -> float:
    return int(tt, 16) / (16 ** len(tt))

def load_tt(aig_filename: str) -> str:
    aig_dir, basename = os.path.split(aig_filename)
    name, _ = os.path.splitext(basename)
    truth_path = os.path.join(os.path.dirname(aig_dir), "truth", name + ".txt")
    if os.path.isfile(truth_path):
        with open(truth_path) as f:
            return f.read().strip()
    return aig_to_tt(aig_filename)

def apply_action(
    tt: str,
    in_neg: list[bool],
    in_perm: list[int],
    out_neg: bool,
    post_proc_fn: Callable | None = None,
    post_cache: dict | None = None,
) -> tuple[str, str]:
    tt_new = apply_to_tt(tt, in_neg, in_perm, out_neg)
    if post_proc_fn is None:
        return tt_new, tt_new
    if post_cache is None:
        return tt_new, post_proc_fn(tt_new)[0]

    tt_post = post_cache.get(tt_new)
    if tt_post is None:
        tt_post = post_proc_fn(tt_new)[0]
        post_cache[tt_new] = tt_post
    return tt_new, tt_post

NUM_WORKERS = 1

def set_num_workers(num_workers: int | None) -> int:
    global NUM_WORKERS
    if num_workers and num_workers > 0:
        NUM_WORKERS = num_workers
    else:
        try:
            NUM_WORKERS = len(os.sched_getaffinity(0))
        except AttributeError:  # not Linux
            NUM_WORKERS = os.cpu_count() or 1
    return NUM_WORKERS

def map_actions(
    rollouts,
    in_neg,
    in_perm,
    out_neg,
    num_pis: list[int] | None = None,
    reduce_fn: Callable | None = None,
    num_workers: int | None = None,
    desc: str | None = None,
):
    group_size = in_neg.shape[0]
    in_neg_l = in_neg.reshape(group_size, -1).bool().tolist()
    in_perm_l = in_perm.reshape(group_size, -1).tolist()
    out_neg_l = out_neg.reshape(group_size, -1).bool().tolist()

    if num_pis is None:
        num_pis = [int(rollout.data.PIs.numel()) for rollout in rollouts]
    offsets = [0]
    for num_pi in num_pis:
        offsets.append(offsets[-1] + num_pi)

    post_caches = defaultdict(dict)
    def work(i):
        rollout = rollouts[i]
        tt, post_proc_fn = rollout.tt, rollout.post_proc_fn
        post_cache = post_caches[post_proc_fn] if post_proc_fn is not None else None
        lo, hi = offsets[i], offsets[i + 1]
        pairs = [
            apply_action(
                tt, in_neg_l[g][lo:hi], in_perm_l[g][lo:hi], out_neg_l[g][i],
                post_proc_fn, post_cache
            )
            for g in range(group_size)
        ]
        return pairs if reduce_fn is None else reduce_fn(rollout, pairs)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        results = pool.map(work, range(len(rollouts)))
        if desc is not None:
            results = tqdm(results, total=len(rollouts), desc=f"{desc} ({NUM_WORKERS} threads)")
        return list(results)

class Rollout(object):
    def __init__(self, filename, post_proc_fn: Callable | None = None):
        self.data = PARSER.read_aiger(filename)
        self.tt = load_tt(filename)
        self.post_proc_fn = post_proc_fn

    @classmethod
    def from_xdata(cls, x_data, edge_index, tt, post_proc_fn: Callable | None = None):
        rollout = cls.__new__(cls)
        rollout.data = PARSER.from_xdata(x_data, edge_index)
        rollout.tt = tt
        rollout.post_proc_fn = post_proc_fn
        return rollout

    @classmethod
    def from_tt_only(cls, tt, post_proc_fn: Callable | None = None):
        rollout = cls.__new__(cls)
        rollout.data = None
        rollout.tt = tt
        rollout.post_proc_fn = post_proc_fn
        return rollout

    def step(self, in_neg: list[bool], in_perm: list[int], out_neg: bool):
        tt_new, tt_post = apply_action(self.tt, in_neg, in_perm, out_neg, self.post_proc_fn)
        return self.rewards(tt_new, tt_post)

    def rewards(self, tt_new: str, tt_post: str):
        base = tt_to_float(self.tt)
        reward_raw = base - tt_to_float(tt_new)
        reward = reward_raw if tt_post == tt_new else base - tt_to_float(tt_post)
        return tt_new, tt_post, reward, reward_raw

def collate_rollouts(rollouts: list[Rollout]):
    data_list = [rollout.data for rollout in rollouts]
    return Batch.from_data_list(data_list)

if __name__ == "__main__":
    import random
    from src.mockturtle import exact_npn, sifting_npn

    # Example usage of Rollout
    rollout = Rollout("examples/example_6bit.aig", post_proc_fn=sifting_npn)
    tt = rollout.tt
    print("Initial truth table:", tt)
    print("Initial float value:", tt_to_float(tt))

    # step using random action
    in_neg = [random.random() > 0 for _ in range(6)]
    in_perm = list(range(6))
    random.shuffle(in_perm)
    out_neg = random.random() > 0
    tt_new, tt_post, reward, reward_raw = rollout.step(in_neg, in_perm, out_neg)
    print("Random in_neg:", in_neg)
    print("Random in_perm:", in_perm)
    print("Random out_neg:", out_neg)
    print("New truth table:", tt_new)
    print("Post-processed truth table:", tt_post)
    print("Reward:", reward)
    print("Reward raw:", reward_raw)

    # step using exact action
    tt_optim, in_neg, in_perm, out_neg = exact_npn(tt)
    tt_new, tt_post, reward, reward_raw = rollout.step(in_neg, in_perm, out_neg)
    print("Exact in_neg:", in_neg)
    print("Exact in_perm:", in_perm)
    print("Exact out_neg:", out_neg)
    print("Optimized truth table:", tt_optim)
    print("New truth table:", tt_new)
    print("Post-processed truth table:", tt_post)
    print("Reward:", reward)
    print("Reward raw:", reward_raw)
