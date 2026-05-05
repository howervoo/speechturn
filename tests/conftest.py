"""Keep offline CPU checks deterministic and inexpensive."""

import torch

torch.set_num_threads(1)
