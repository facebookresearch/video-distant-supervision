import torch
import torch.utils.data
import lib.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Dummy_howto100m(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, num_retries=20):
        logger.info("Constructing Dummy_HowTo100M {}...".format(mode))
        self.cfg = cfg

    def __getitem__(self, index):
        # Placeholder for a set of self.cfg.MODEL.NUM_SEG video clips from a single video.
        # Clips should be ~in order~
        video_clips = torch.rand((3, self.cfg.MODEL.NUM_SEG, 8, 224, 224))
        label = 1
        text = {
            "label": torch.tensor([1]),
            "emb": torch.rand((self.cfg.MODEL.NUM_SEG, 768))
        }
        return video_clips, label, index, text

    def __len__(self):
        return 1000