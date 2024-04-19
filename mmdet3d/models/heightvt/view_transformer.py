from mmcv.runner import BaseModule
from mmdet.models import NECKS


@NECKS.register_module()
class HeightVT(BaseModule):

    def __init__(self,
                 down_sample,
                 **kwargs):
        super(HeightVT, self).__init__(**kwargs)

        self.down_sample = down_sample

    def forward(self, mlvl_feats, cam_params=None):
        """
            mlvl_feats:
                (24, 64, 64, 176), (24, 64, 32, 88), (24, 64, 16, 44)
            cam_params:
                rots:
                trans:
                intrinsics:
                post_rots:
                post_trans:
                bda:
        """

        return mlvl_feats
