import torch

class MetricAggregator(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.sums = {}
        self.counts = {}

    def update(self, metrics):
        for name, value in metrics.items():
            value = value.float()
            self.sums[name] = self.sums.get(name, 0.0) + value.sum().item()
            self.counts[name] = self.counts.get(name, 0) + value.numel()

    def average(self):
        return {name: self.sums[name] / self.counts[name] for name in self.sums}

    def pooled(self, names):
        total = sum(self.sums[name] for name in names)
        count = sum(self.counts[name] for name in names)
        return total / count

def group_index(index: torch.Tensor, group_size: int = 1) -> torch.Tensor:
    # Replicate `index` group_size times and shift properly
    num_bit = index.numel()
    num_group = index.max().item() + 1
    offset = torch.arange(group_size, device=index.device).repeat_interleave(num_bit)
    return index.repeat(group_size) + offset * num_group

def group_start(index: torch.Tensor) -> torch.Tensor:
    # Assume that each group in index is contiguous
    is_start = torch.ones_like(index, dtype=torch.bool)
    is_start[1:] = index[1:] != index[:-1]
    positions = torch.arange(len(index), device=index.device)
    start_positions = torch.where(is_start, positions, 0)
    start = start_positions.cummax(dim=0).values
    return start

def group_gather(src: torch.Tensor, index: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    # Gather elements from src according to perm, while respecting the group structure defined by index
    start = group_start(index)
    return torch.gather(src, 0, perm + start)

def group_to_dense(src: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Scatter a flat grouped tensor into a padded [num_groups, max_width] matrix
    pos = torch.arange(len(index), device=index.device) - group_start(index)
    num_groups = int(index[-1]) + 1
    width = int(pos.max()) + 1
    dense = src.new_full((num_groups, width), float("-inf"))
    dense[index, pos] = src
    mask = torch.zeros(num_groups, width, dtype=torch.bool, device=index.device)
    mask[index, pos] = True
    return dense, mask

def group_triu_indices(index: torch.Tensor) -> torch.Tensor:
    # Upper-triangular (strictly not diagonal) indices for each group defined by index
    num_bit = index.numel()
    sizes = torch.bincount(index)
    group_ends = torch.repeat_interleave(sizes.cumsum(0), sizes)

    base_rows = torch.arange(num_bit, device=index.device)
    row_lengths = group_ends - base_rows - 1
    row = torch.repeat_interleave(base_rows, row_lengths)

    row_starts = row_lengths.cumsum(0) - row_lengths
    within_row = (
        torch.arange(row.numel(), device=index.device)
        - torch.repeat_interleave(row_starts, row_lengths)
    )
    col = row + within_row + 1

    return row, col

if __name__ == "__main__":
    # Example usage
    src = torch.tensor([1, 2, 3, 4, 5])
    index = torch.tensor([0, 0, 1, 1, 1])
    perm = torch.tensor([1, 0, 2, 1, 0])

    print("Grouped index:\n", group_index(index, group_size=3))
    print("Group start:\n", group_start(index))
    print("Group gather:\n", group_gather(src, index, perm))
    print("Group to dense:\n", group_to_dense(src.float(), index))
    print("Group upper-triangular indices:\n", group_triu_indices(index))
