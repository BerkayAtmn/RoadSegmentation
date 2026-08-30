"""Sliding-window inference shared between training-time validation (notebook.ipynb)
and tester.py, so both evaluate full-resolution tiles the same way: merge predictions
across overlapping windows first, then compute metrics on the merged result."""

import torch


def window_positions(size, window, stride):
    """Top-left offsets covering `size` with `window`-sized steps of `stride`,
    with a final offset snapped to the edge so the last window is flush."""
    if size <= window:
        return [0]
    positions = list(range(0, size - window + 1, stride))
    if positions[-1] != size - window:
        positions.append(size - window)
    return positions


@torch.no_grad()
def sliding_window_predict(model, images, window=512, stride=384, max_batch=32, amp_dtype=torch.bfloat16):
    """Tile `images` (B,C,H,W) into overlapping windows, run the model on each,
    and stitch the probabilities back onto a full-size canvas, averaging where
    windows overlap. Returns (B,1,H,W) probabilities at full resolution."""
    B, C, H, W = images.shape
    coords = [
        (y, x)
        for y in window_positions(H, window, stride)
        for x in window_positions(W, window, stride)
    ]

    # (num_windows * B, C, window, window), grouped window-major
    batched = torch.cat([images[:, :, y:y + window, x:x + window] for y, x in coords], dim=0)

    probs_chunks = []
    for i in range(0, batched.size(0), max_batch):
        chunk = batched[i:i + max_batch]
        if images.is_cuda:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(chunk)
        else:
            logits = model(chunk)
        probs_chunks.append(torch.sigmoid(logits.float()))
    probs = torch.cat(probs_chunks, dim=0).view(len(coords), B, 1, window, window)

    prob_sum = torch.zeros((B, 1, H, W), device=images.device)
    count = torch.zeros((B, 1, H, W), device=images.device)
    for i, (y, x) in enumerate(coords):
        prob_sum[:, :, y:y + window, x:x + window] += probs[i]
        count[:, :, y:y + window, x:x + window] += 1

    return prob_sum / count.clamp(min=1)
