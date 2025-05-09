import time
from multiprocessing import Pool, cpu_count
import numpy as np
import torch


def eval_model(model, loader, config, device):
    model.eval()
    pass