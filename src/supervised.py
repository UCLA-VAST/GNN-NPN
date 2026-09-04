import os

from tqdm import tqdm

import torch
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogits
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import scatter, group_argsort

from src.rollout import collate_rollouts
from src.utils import MetricAggregator, group_triu_indices

def supervised_metrics(
    in_neg_logits, in_perm_logits, out_neg_logits, index,
    in_neg, in_perm, out_neg
):
    # --- negation: one BCE per PI/PO ---
    in_neg_loss = BCEWithLogits(in_neg_logits, in_neg, reduction="none")
    in_neg_correct = (in_neg_logits > 0).long() == in_neg.long()
    out_neg_loss = BCEWithLogits(out_neg_logits, out_neg, reduction="none")
    out_neg_correct = (out_neg_logits > 0).long() == out_neg.long()

    # --- permutation: pairwise ordering problems over PI pairs ---
    i_idx, j_idx = group_triu_indices(index)
    in_perm_pair_logits = in_perm_logits[j_idx] - in_perm_logits[i_idx]
    in_perm_pair_target = (in_perm[i_idx] < in_perm[j_idx]).float()
    in_perm_loss = BCEWithLogits(in_perm_pair_logits, in_perm_pair_target, reduction="none")
    in_perm_pair_correct = (in_perm_pair_logits > 0) == (in_perm_pair_target > 0.5)
    in_perm_correct = scatter(in_perm_pair_correct.float(), index[i_idx], dim=0, reduce="min") > 0

    return {
        "in_neg_loss": in_neg_loss,
        "in_neg_acc": in_neg_correct,
        "in_perm_loss": in_perm_loss,
        "in_perm_acc_pair": in_perm_pair_correct,
        "in_perm_acc_exact": in_perm_correct,
        "out_neg_loss": out_neg_loss,
        "out_neg_acc": out_neg_correct,
    }

@torch.inference_mode()
def eval_supervised(agent, dataset, device, cfg):
    was_training = agent.training
    agent.eval()

    aggregator = MetricAggregator()
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=list)
    for rollouts in tqdm(dataloader, desc="Evaluating"):
        batch = collate_rollouts(rollouts)
        batch = batch.to(device)
        in_neg_logits, in_perm_logits, out_neg_logits, index = agent.get_logits(batch)
        metrics = supervised_metrics(
            in_neg_logits, in_perm_logits, out_neg_logits, index,
            batch.in_neg, batch.in_perm, batch.out_neg
        )
        aggregator.update(metrics)

    averaged = aggregator.average()
    averaged["loss"] = aggregator.pooled(["in_neg_loss", "in_perm_loss", "out_neg_loss"])
    for name, value in averaged.items():
        tqdm.write(f"{name}: {value:.4f}")

    agent.train(was_training)
    return averaged

def train_supervised(agent, dataset, val_datasets, device, cfg):
    agent.train()
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    writer = SummaryWriter(log_dir=cfg.dir)
    os.makedirs(cfg.dir, exist_ok=True)
    best_val_loss = float("inf")

    def log_eval(step):
        nonlocal best_val_loss
        per_dataset_loss = []
        for name, val_dataset in val_datasets:
            tqdm.write(f"Evaluating on validation set '{name}'...")
            averaged = eval_supervised(agent, val_dataset, device, cfg)
            for metric, value in averaged.items():
                writer.add_scalar(f"val/{name}/{metric}", value, step)
            per_dataset_loss.append(averaged["loss"])
        val_loss = sum(per_dataset_loss) / len(per_dataset_loss)
        writer.add_scalar("val/loss", val_loss, step)

        torch.save(agent.state_dict(), os.path.join(cfg.dir, "last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(agent.state_dict(), os.path.join(cfg.dir, "best.pt"))
            tqdm.write(f"New best validation loss: {val_loss:.4f}")
        return val_loss

    log_eval(0)
    aggregator = MetricAggregator()
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=list)
    def step_stream():
        while True:
            yield from dataloader

    for i_update, rollouts in tqdm(
        enumerate(step_stream()), desc="Train Supervised", total=cfg.train_steps
    ):
        if i_update >= cfg.train_steps:
            break
        batch = collate_rollouts(rollouts)
        batch = batch.to(device)
        in_neg_logits, in_perm_logits, out_neg_logits, index = agent.get_logits(batch)
        metrics = supervised_metrics(
            in_neg_logits, in_perm_logits, out_neg_logits, index,
            batch.in_neg, batch.in_perm, batch.out_neg,
        )
        losses = [metrics["in_neg_loss"], metrics["in_perm_loss"], metrics["out_neg_loss"]]
        total = sum(loss.sum() for loss in losses)
        count = sum(loss.numel() for loss in losses)
        loss = total / count

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), i_update)
        writer.add_scalar("train/grad_norm", grad_norm.item(), i_update)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], i_update)

        aggregator.update(metrics)
        if (i_update + 1) % cfg.log_interval == 0:
            averaged = aggregator.average()
            for name, value in averaged.items():
                tqdm.write(f"{name}: {value:.4f}")
                writer.add_scalar(f"train/{name}", value, i_update)
            aggregator.clear()

        if (i_update + 1) % cfg.eval_interval == 0:
            log_eval(i_update)

    if cfg.train_steps % cfg.eval_interval != 0:
        log_eval(cfg.train_steps)

    writer.close()

if __name__ == "__main__":
    # Example usage of supervised metrics
    logits = torch.tensor([2.0, -2.0, 1.0, 0.0, -1.0])
    index = torch.tensor([0, 0, 1, 1, 1])
    neg = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    perm = torch.tensor([1, 0, 2, 0, 1])

    metrics = supervised_metrics(
        logits, logits, logits, index,
        neg, perm, neg
    )
    for name, value in metrics.items():
        print(name, "\n", value)
