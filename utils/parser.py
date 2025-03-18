import re
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from utils import MODEL_SIZE, TASK_SET


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config. Mostly for convenience.
    """

    # Logic
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v == None:
                v = True
        except:
            pass

    # Convenience
    # cfg.work_dir = Path(hydra.utils.get_original_cwd()) / \
    #     'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
    # cfg.task_title = cfg.task.replace("-", " ").title()

    return cfg
