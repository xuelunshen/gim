import torch

from .superpoint import SuperPoint
from .models.matchers.lightglue import LightGlue


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """

    # noinspection PyDefaultArgument
    def __init__(self, config={}):
        super().__init__()
        self.detector = SuperPoint({
            'max_num_keypoints': 2048,
            'force_num_keypoints': True,
            'detection_threshold': 0.0,
            'nms_radius': 3,
            'trainable': False,
        })
        self.model = LightGlue({
            'filter_threshold': 0.1,
            'flash': False,
            'checkpointed': True,
        })

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        pred.update({k + '0': v for k, v in self.detector({
            "image": data["gray0"],
            "image_size": data["size0"],
        }).items()})
        pred.update({k + '1': v for k, v in self.detector({
            "image": data["gray1"],
            "image_size": data["size1"],
        }).items()})

        pred.update(self.model({
            **pred, **{
                'resize0': data['size0'],
                'resize1': data['size1']
            }
        }))

        return pred
