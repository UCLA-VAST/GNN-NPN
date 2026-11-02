import os
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from src.mockturtle import (
    get_action_fn,
    generate_random_aig,
    generate_random_aig_rewriting,
)
from src.rollout import Rollout

BENCHMARKS = [
    "adder", "arbiter", "bar", "cavlc", "ctrl", "dec", "div", "hyp",
    "i2c", "int2float", "log2", "max", "mem_ctrl", "multiplier",
    "priority", "router", "sin", "sqrt", "square", "voter",
]

class RolloutDataset(Dataset):
    def __init__(self, rollouts: list[Rollout]):
        super().__init__()
        self.rollouts = rollouts

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, index) -> Rollout:
        return self.rollouts[index]

class EPFLDataset(Dataset):
    def __init__(
        self,
        root: str,
        num_bit: int,
        include_action: str,
        post_proc: str,
        rewriting: bool,
    ):
        super().__init__()
        self.root = root
        self.num_bit = num_bit
        self._action_fn = get_action_fn(include_action)
        self._post_proc_fn = get_action_fn(post_proc)
        self.rewriting = rewriting

        # Load AIG files
        aig_subdir = "aig_rewriting" if rewriting else "aig"
        self.filenames = []
        for benchmark in BENCHMARKS:
            aig_dir = os.path.join(self.root, benchmark, str(num_bit), aig_subdir)
            if not os.path.isdir(aig_dir):
                continue
            filelist = [
                os.path.join(benchmark, str(num_bit), aig_subdir, f)
                for f in sorted(os.listdir(aig_dir))
                if f.endswith(".aig")
            ]
            self.filenames.extend(filelist)
        print(f"EPFLDataset: {len(self.filenames)} AIG files found for {num_bit}-bit benchmarks.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index) -> Rollout:
        filename = os.path.join(self.root, self.filenames[index])
        rollout = Rollout(filename=filename, post_proc_fn=self._post_proc_fn)
        if self._action_fn is not None:
            _, in_neg, in_perm, out_neg = self._action_fn(rollout.tt)
            rollout.data.in_neg = torch.tensor(in_neg, dtype=torch.float)
            rollout.data.in_perm = torch.tensor(in_perm, dtype=torch.long)
            rollout.data.out_neg = torch.tensor(out_neg, dtype=torch.float)
        return rollout

class RandomAIGDataset(Dataset):
    def __init__(
        self,
        num_bit: int,
        num_train: int,
        num_eval: int,
        num_test: int,
        mean_nodes: float,
        std_nodes: float,
        include_action: str,
        post_proc: str,
        rewriting: bool,
        seed: int,
    ):
        super().__init__()

        self.num_bit = num_bit
        self.num_train = num_train
        self.num_eval = num_eval
        self.num_test = num_test
        self._mean_nodes = mean_nodes
        self._std_nodes = std_nodes
        self._action_fn = get_action_fn(include_action)
        self._post_proc_fn = get_action_fn(post_proc)
        self.rewriting = rewriting
        self._gen_fn = generate_random_aig_rewriting if rewriting else generate_random_aig
        self.seed = seed

        print(f"RandomAIGDataset: {num_train} train samples/epoch, "
              f"generating {num_eval} eval + {num_test} test rollouts...")
        self.val_rollouts: list[Rollout] = [
            self._make_rollout(seed + i)
            for i in tqdm(range(num_eval), desc="Generating val AIGs")
        ]
        self.test_rollouts: list[Rollout] = [
            self._make_rollout(seed + num_eval + i)
            for i in tqdm(range(num_test), desc="Generating test AIGs")
        ]

    def _make_rollout(self, seed: int) -> Rollout:
        x_data, edge_index, tt = self._gen_fn(
            self.num_bit,
            self._mean_nodes, self._std_nodes,
            seed=seed,
        )
        rollout = Rollout.from_xdata(x_data, edge_index, tt, post_proc_fn=self._post_proc_fn)
        if self._action_fn is not None:
            _, in_neg, in_perm, out_neg = self._action_fn(rollout.tt)
            rollout.data.in_neg = torch.tensor(in_neg, dtype=torch.float)
            rollout.data.in_perm = torch.tensor(in_perm, dtype=torch.long)
            rollout.data.out_neg = torch.tensor(out_neg, dtype=torch.float)
        return rollout

    def __len__(self) -> int:
        return self.num_train

    def __getitem__(self, index) -> Rollout:
        return self._make_rollout(self.seed + self.num_eval + self.num_test + index)

if __name__ == "__main__":
    # Example usage of EPFLDataset
    dataset = EPFLDataset(
        root="benchmarks", num_bit=6, rewriting=True,
        include_action="exact_npn", post_proc="sifting_npn"
    )
    sample_rollout = dataset[0]
    print(f"Sample rollout TT: {sample_rollout.tt}")

    # Example usage of RandomAIGDataset
    random_dataset = RandomAIGDataset(
        num_bit=6, rewriting=True,
        num_train=10, num_eval=5, num_test=5, seed=42,
        mean_nodes=7.0, std_nodes=2.0,
        include_action="exact_npn", post_proc="sifting_npn"
    )
    random_sample_rollout = random_dataset[0]
    print(f"Random sample rollout TT: {random_sample_rollout.tt}")