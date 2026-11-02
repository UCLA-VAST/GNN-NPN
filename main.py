from omegaconf import OmegaConf
import os
import random
import sys
import warnings

import torch
from torch.utils.data import random_split, Subset, ConcatDataset

from src.agent import Agent, run_inference
from src.dataset import RolloutDataset, EPFLDataset, RandomAIGDataset
from src.npn import collect_unique_truth_tables, run_npn
from src.reinforcement import eval_reinforcement, train_grpo
from src.rollout import set_num_workers
from src.supervised import eval_supervised, train_supervised

MISSING_KEY = object()

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <command> <config.yaml> [key=value ...]")
        sys.exit(1)

    cmd = sys.argv[1]
    cfg = OmegaConf.load(sys.argv[2])
    for override in sys.argv[3:]:
        key, value = override.split("=", 1)
        if OmegaConf.select(cfg, key, default=MISSING_KEY) is MISSING_KEY:
            warnings.warn(
                f"Override key {key!r} is not present in {sys.argv[2]}; "
                "check for a typo (it will be added as a new key)."
            )
        value = OmegaConf.from_dotlist([f"value={value}"]).value
        OmegaConf.update(cfg, key, value, merge=True)
    print(f"Running command {cmd} with config: {cfg}")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    set_num_workers(cfg.get("num_workers", None))

    # run-npn uses saved inference results, no dataset or agent is needed
    if cmd == "run-npn":
        with open(cfg.npn.truth_table_path) as f:
            unique_tts = [line.strip() for line in f if line.strip()]
        results = torch.load(cfg.npn.inference_results_path)
        run_npn(
            unique_tts,
            results["in_neg_logits"],
            results["in_perm_logits"],
            results["out_neg_logits"],
            cfg.npn.trial,
            cfg.npn.temp,
            cfg.npn.post_proc,
        )
        return

    full_datasets = []
    train_datasets, val_datasets, test_datasets = [], [], []
    for dataset_cfg in cfg.dataset:
        name = dataset_cfg.name
        if dataset_cfg.cls == "epfl":
            dataset = EPFLDataset(
                root=dataset_cfg.root,
                num_bit=dataset_cfg.num_bit,
                include_action=dataset_cfg.action,
                post_proc=cfg.reinforcement.post_proc,
                rewriting=dataset_cfg.rewriting,
            )
            train_split, val_split, test_split = random_split(
                dataset,
                [dataset_cfg.train_split, dataset_cfg.val_split,
                1 - dataset_cfg.train_split - dataset_cfg.val_split],
            )
            full_datasets.append(dataset)
            train_datasets.append(train_split)
            val_datasets.append((name, val_split))
            test_datasets.append((name, test_split))
        elif dataset_cfg.cls == "random":
            dataset = RandomAIGDataset(
                num_bit=dataset_cfg.num_bit,
                num_train=dataset_cfg.num_train,
                num_eval=dataset_cfg.num_eval,
                num_test=dataset_cfg.num_test,
                mean_nodes=dataset_cfg.mean_nodes,
                std_nodes=dataset_cfg.std_nodes,
                include_action=dataset_cfg.action,
                post_proc=cfg.reinforcement.post_proc,
                rewriting=dataset_cfg.rewriting,
                seed=cfg.seed,
            )
            full_datasets.append(dataset)
            train_datasets.append(dataset)
            val_datasets.append((name, RolloutDataset(dataset.val_rollouts)))
            test_datasets.append((name, RolloutDataset(dataset.test_rollouts)))
        else:
            raise ValueError(f"Unknown dataset class: {dataset_cfg.cls}")
    full_dataset = ConcatDataset(full_datasets)
    train_dataset = ConcatDataset(train_datasets)

    # uniquify-truth-tables uses the dataset to collect unique truth tables, no agent is needed
    if cmd == "uniquify-truth-tables":
        assert len(full_datasets) == 1, "uniquify-truth-tables only supports a single dataset"
        unique_tts, unique_indices = collect_unique_truth_tables(full_datasets[0])
        with open(cfg.npn.truth_table_path, "w") as f:
            for tt in unique_tts:
                f.write(tt + "\n")
        with open(cfg.npn.indices_path, "w") as f:
            for index in unique_indices:
                f.write(f"{index}\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(cfg.agent)
    load_path = getattr(cfg, "load", None)
    if load_path:
        agent.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Loaded checkpoint: {load_path}")
    agent.to(device)

    # supervised learning
    if cmd == "test-supervised":
        for name, test_dataset in test_datasets:
            print(f"Test supervised '{name}':")
            eval_supervised(agent, test_dataset, device, cfg.supervised)
    elif cmd == "train-supervised":
        os.makedirs(cfg.supervised.dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.supervised.dir, "config.yaml"))
        train_supervised(agent, train_dataset, val_datasets, device, cfg.supervised)

    # reinforcement learning
    elif cmd == "test-reinforcement":
        for name, test_dataset in test_datasets:
            print(f"Test reinforcement '{name}':")
            eval_reinforcement(agent, test_dataset, device, cfg.reinforcement)
    elif cmd == "train-grpo":
        os.makedirs(cfg.reinforcement.grpo.dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.reinforcement.grpo.dir, "config.yaml"))
        train_grpo(agent, train_dataset, val_datasets, device, cfg.reinforcement)

    # inference
    elif cmd == "run-inference":
        with open(cfg.npn.indices_path) as f:
            unique_indices = [int(line) for line in f if line.strip()]
        inference_dataset = Subset(full_dataset, unique_indices)
        in_neg_logits, in_perm_logits, out_neg_logits = run_inference(agent, inference_dataset, device, cfg.npn.inference_batch_size)
        torch.save({
            "in_neg_logits": in_neg_logits.cpu(),
            "in_perm_logits": in_perm_logits.cpu(),
            "out_neg_logits": out_neg_logits.cpu(),
        }, cfg.npn.inference_results_path)
    else:
        raise ValueError(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
