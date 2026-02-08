import torch
import torchvision
import importlib

# Disable MPS backend if available to avoid potential issues on macOS
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
