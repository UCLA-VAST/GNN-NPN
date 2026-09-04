import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.rollout import collate_rollouts, map_actions

def compute_rewards(rollouts, in_neg, in_perm, out_neg):
    def per_rollout_rewards(rollout, pairs):
        return [rollout.rewards(*pair)[2:] for pair in pairs]

    # [num_rollouts][group_size] of (reward, reward_raw).
    per_rollout = map_actions(
        rollouts, in_neg, in_perm, out_neg,
        reduce_fn=per_rollout_rewards
    )
    rewards = torch.tensor(per_rollout).permute(2, 1, 0)
    return rewards[0].contiguous(), rewards[1].contiguous()

@torch.inference_mode()
def eval_reinforcement(agent, dataset, device, cfg, temp=None):
    if temp is None:
        temp = cfg.temp
    was_training = agent.training
    agent.eval()

    sums = {"r_mean": 0.0, "r_best": 0.0, "r_raw_mean": 0.0, "r_raw_best": 0.0}
    n_rollouts = 0

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=list)
    for rollouts in tqdm(dataloader, desc="Evaluate Reinforcement"):
        batch = collate_rollouts(rollouts)
        batch = batch.to(device)
        in_neg, in_perm, out_neg = agent.act(batch, group_size=cfg.group_size, temp=temp)
        rewards, rewards_raw = compute_rewards(
            rollouts, in_neg, in_perm, out_neg
        )
        sums["r_mean"] += rewards.mean(dim=0).sum().item()
        sums["r_best"] += rewards.max(dim=0).values.sum().item()
        sums["r_raw_mean"] += rewards_raw.mean(dim=0).sum().item()
        sums["r_raw_best"] += rewards_raw.max(dim=0).values.sum().item()
        n_rollouts += len(rollouts)

    averaged = {name: total / n_rollouts for name, total in sums.items()}
    for name, value in averaged.items():
        tqdm.write(f"{name}: {value:.4f}")

    agent.train(was_training)
    return averaged

def train_grpo(agent, dataset, val_datasets, device, cfg):
    grpo_cfg = cfg.grpo
    agent.train()
    optimizer = torch.optim.Adam(agent.parameters(), lr=grpo_cfg.lr)
    os.makedirs(grpo_cfg.dir, exist_ok=True)
    writer = SummaryWriter(log_dir=grpo_cfg.dir)
    best_val_reward = -float("inf")
    val_reward_key = "r_raw_mean" if grpo_cfg.raw_reward else "r_mean"

    def log_eval(step):
        nonlocal best_val_reward
        per_dataset_reward = []
        for name, val_dataset in val_datasets:
            tqdm.write(f"Evaluating on validation set '{name}'...")
            averaged = eval_reinforcement(agent, val_dataset, device, cfg)
            for metric, value in averaged.items():
                writer.add_scalar(f"val/{name}/{metric}", value, step)

            # Deterministic (greedy, temp=0) evaluation, under val_det/<name>/.
            tqdm.write(f"Evaluating on validation set '{name}' (deterministic, temp=0)...")
            averaged_det = eval_reinforcement(agent, val_dataset, device, cfg, temp=0.0)
            for metric, value in averaged_det.items():
                writer.add_scalar(f"val_det/{name}/{metric}", value, step)

            per_dataset_reward.append(averaged[val_reward_key])

        val_reward = sum(per_dataset_reward) / len(per_dataset_reward)
        writer.add_scalar("val/r", val_reward, step)

        torch.save(agent.state_dict(), os.path.join(grpo_cfg.dir, "last.pt"))
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            torch.save(agent.state_dict(), os.path.join(grpo_cfg.dir, "best.pt"))
            tqdm.write(f"New best validation reward: {val_reward:.4f}")
        return val_reward

    log_eval(0)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=list)
    def episode_stream():
        while True:
            yield from dataloader

    for i_update, rollouts in tqdm(
        enumerate(episode_stream()), desc="Training", total=grpo_cfg.train_steps
    ):
        if i_update >= grpo_cfg.train_steps:
            break

        batch = collate_rollouts(rollouts)
        batch = batch.to(device)
        in_neg, in_perm, out_neg = agent.act(batch, group_size=cfg.group_size, temp=cfg.temp)
        rewards, rewards_raw = compute_rewards(
            rollouts, in_neg, in_perm, out_neg
        )
        rewards = rewards.to(device)
        rewards_raw = rewards_raw.to(device)
        adv_reward = rewards_raw if grpo_cfg.raw_reward else rewards

        adv = adv_reward - adv_reward.mean(dim=0, keepdim=True)
        std = adv_reward.std(dim=0, keepdim=True)
        adv = adv / (std + 1e-8)

        # PPO-style epochs
        old_logprob = None
        n_skipped = 0
        grad_norm = torch.tensor(0.0, device=device)
        for _ in range(grpo_cfg.num_epoch):
            logprob, entropy = agent(
                batch, in_neg, in_perm, out_neg, group_size=cfg.group_size, temp=cfg.temp
            )
            if old_logprob is None:
                old_logprob = logprob.detach()

            ratio = torch.exp(logprob - old_logprob)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - grpo_cfg.clip_eps, 1.0 + grpo_cfg.clip_eps) * adv
            pg_loss = -torch.min(unclipped, clipped).mean()
            entropy_mean = entropy.mean()
            loss = pg_loss - grpo_cfg.ent_coef * entropy_mean

            # Guard against numerical divergence
            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), grpo_cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                n_skipped += 1
                continue
            optimizer.step()

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > grpo_cfg.clip_eps).float().mean()

        writer.add_scalar("train/loss", loss.item(), i_update)
        writer.add_scalar("train/pg_loss", pg_loss.item(), i_update)
        writer.add_scalar("train/entropy", entropy_mean.item(), i_update)
        writer.add_scalar("train/clip_frac", clip_frac.item(), i_update)
        writer.add_scalar("train/grad_norm", grad_norm.item(), i_update)
        writer.add_scalar("train/n_skipped", n_skipped, i_update)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], i_update)
        writer.add_scalar("train/r_mean", rewards.mean().item(), i_update)
        writer.add_scalar("train/r_best", rewards.max(dim=0).values.mean().item(), i_update)
        writer.add_scalar("train/r_raw_mean", rewards_raw.mean().item(), i_update)
        writer.add_scalar("train/r_raw_best", rewards_raw.max(dim=0).values.mean().item(), i_update)

        if (i_update + 1) % grpo_cfg.log_interval == 0:
            tqdm.write(
                f"step {i_update + 1}: loss {loss.item():.4f} "
                f"r_mean {rewards.mean().item():.4f} entropy {entropy_mean.item():.4f} "
                f"clip_frac {clip_frac.item():.4f}"
            )

        if (i_update + 1) % grpo_cfg.eval_interval == 0:
            log_eval(i_update)

    if grpo_cfg.train_steps % grpo_cfg.eval_interval != 0:
        log_eval(grpo_cfg.train_steps)

    writer.close()
