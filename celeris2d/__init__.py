import os

# GPU memory: expandable allocator segments let freed blocks serve tensors of other sizes (a box-256 refinement ran
# out with 1.6 GB reserved but unusable); read when CUDA starts, so it is set on import. A user's own setting wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
