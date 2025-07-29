
import os
from PIL import Image
import numpy as np
import torch

# Hook into the worker
import modules.async_worker as worker
from modules import config

# Backup original function
original_execute = worker.execute

def grid_execute(*args, **kwargs):
    # Call original to get the images list
    results = original_execute(*args, **kwargs)

    # Check if we should force grid
    if not getattr(config.args, 'force_grid', False):
        return results

    print("[Force Grid] Stitching images into grid...")

    images = [r['image'] for r in results if 'image' in r]
    if len(images) < 1:
        return results

    # Determine grid size: closest square
    n = len(images)
    cols = rows = int(np.ceil(np.sqrt(n)))

    # If not enough images, pad with last one
    while len(images) < (cols * rows):
        images.append(images[-1])

    # Get size of images (assume all same size)
    w, h = images[0].size
    grid_img = Image.new('RGB', (w * cols, h * rows))

    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid_img.paste(img, (x, y))

    # Save only the grid
    output_dir = config.path_outputs
    basename = "grid_output"
    grid_path = os.path.join(output_dir, f"{basename}.png")
    counter = 1
    while os.path.exists(grid_path):
        grid_path = os.path.join(output_dir, f"{basename}_{counter:04d}.png")
        counter += 1

    grid_img.save(grid_path)
    print(f"[Force Grid] Saved grid to {grid_path}")

    # Clear individual outputs
    return [{'image': grid_img, 'type': 'output'}]


# Patch worker only once
if not hasattr(worker, '_grid_patched'):
    worker.execute = grid_execute
    worker._grid_patched = True
