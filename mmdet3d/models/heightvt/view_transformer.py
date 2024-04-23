import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule
from mmdet.models.backbones.resnet import BasicBlock
from torch.cuda.amp.autocast_mode import autocast

from .height_net import Mlp, SELayer, ASPP, HeightNet
from ..builder import NECKS


class SABlock(nn.Module):
    """ Spatial attention block """

    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                       nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x, y):
        return torch.mul(self.conv(x), self.attention(y))


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """

    def __init__(self, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.depth2sem = SABlock(channels, channels)
        self.sem2depth = SABlock(channels, channels)

    def forward(self, depth, sem):
        depth_new = depth + self.sem2depth(sem, depth)
        sem_new = sem + self.depth2sem(depth, sem)
        return depth_new, sem_new


class TaskHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(TaskHead, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.decoder = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, return_feat=True):
        if return_feat:
            feat = self.decoder(feat)
            return self.head(feat), feat
        return self.head(self.decoder(feat))


class TaskFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TaskFPN, self).__init__()
        self.reduce_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.self_attention = SABlock(out_channels, out_channels)

    def forward(self, feat0, feat1):
        feat0 = self.reduce_conv(F.interpolate(feat0, scale_factor=2, mode='bilinear'))
        feat0_new = feat0 + self.self_attention(feat1, feat0)
        return feat0_new


class MSCThead(nn.Module):
    def __init__(self,
                 in_channels=[512, 512],
                 mid_channels=[512, 256],
                 depth_channels=118,
                 semantic_channels=2,
                 context_channels=80,
                 ):
        super(MSCThead, self).__init__()
        # preprocess
        self.reduce_conv0 = nn.Sequential(
            nn.Conv2d(in_channels[0], mid_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[0]), nn.ReLU(inplace=True))
        self.reduce_conv1 = nn.Sequential(
            nn.Conv2d(in_channels[1], mid_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[1]), nn.ReLU(inplace=True))
        self.bn = nn.BatchNorm1d(27)
        self.scale0_mlp = Mlp(27, mid_channels[0], mid_channels[0])
        self.scale1_mlp = Mlp(27, mid_channels[1], mid_channels[1])
        self.scale0_se = SELayer(mid_channels[0])
        self.scale1_se = SELayer(mid_channels[1])
        self.aspp = ASPP(mid_channels[0], mid_channels[0])
        # stage one
        self.depth_head0 = TaskHead(mid_channels[0], mid_channels[0], depth_channels)
        self.semantic_head0 = TaskHead(mid_channels[0], mid_channels[0], semantic_channels)
        self.context_conv0 = nn.Sequential(
            nn.Conv2d(mid_channels[0], mid_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[0]),
            nn.ReLU(inplace=True)
        )
        # combine information
        self.mtd = MultiTaskDistillationModule(mid_channels[0])
        self.depth_fpn = TaskFPN(mid_channels[0], mid_channels[1])
        self.semantic_fpn = TaskFPN(mid_channels[0], mid_channels[1])
        self.context_fpn = TaskFPN(mid_channels[0], mid_channels[1])
        # stage two
        self.depth_head1 = TaskHead(mid_channels[1], mid_channels[1], depth_channels)
        self.semantic_head1 = TaskHead(mid_channels[1], mid_channels[1], semantic_channels)
        self.context_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels[1], mid_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels[1], context_channels, kernel_size=1, stride=1, padding=0)
        )

    @autocast(False)
    def forward(self, x, mlp_input):
        # preprocess
        B, N, C, H, W = x[0].shape
        scale0_feat = x[0].view(B * N, C, H, W).float()
        B, N, C, H, W = x[1].shape
        scale1_feat = x[1].view(B * N, C, H, W).float()
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        scale0_feat = self.reduce_conv0(scale0_feat)
        scale1_feat = self.reduce_conv1(scale1_feat)
        scale0_se = self.scale0_mlp(mlp_input)[..., None, None]
        scale1_se = self.scale1_mlp(mlp_input)[..., None, None]
        scale0_feat = self.scale0_se(scale0_feat, scale0_se)
        scale1_feat = self.scale1_se(scale1_feat, scale1_se)
        scale0_feat = self.aspp(scale0_feat)

        # stage one
        depth0, depth_feat = self.depth_head0(scale0_feat)
        semantic0, semantic_feat = self.semantic_head0(scale0_feat)
        context_feat = self.context_conv0(scale0_feat)

        # combine information
        depth_feat, semantic_feat = self.mtd(depth_feat, semantic_feat)
        depth_feat = self.depth_fpn(depth_feat, scale1_feat)
        semantic_feat = self.semantic_fpn(semantic_feat, scale1_feat)
        context_feat = self.context_fpn(context_feat, scale1_feat)

        # stage two
        depth1 = self.depth_head1(depth_feat, return_feat=False)
        semantic1 = self.semantic_head1(semantic_feat, return_feat=False)
        context1 = self.context_conv1(context_feat)

        return depth1, semantic1, context1, depth0, semantic0


@NECKS.register_module()
class HeightVT(BaseModule):

    def __init__(self,
                 in_channels=512,
                 out_channels=64,
                 loss_semantic_weight=25,
                 depthnet_cfg=dict(),
                 **kwargs):
        super(HeightVT, self).__init__(**kwargs)

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.height_net = HeightNet(self.in_channels, self.in_channels,
                                    self.out_channels, 2, **depthnet_cfg)

        self.loss_semantic_weight = loss_semantic_weight

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        B, N, _, _ = rot.shape  # 4, 6, 3
        bda = bda.view(B, 1, 4, 4).repeat(1, N, 1, 1)  # (4, 6, 4, 4)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 0, 3],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 1, 3],
            bda[:, :, 2, 2],
            bda[:, :, 2, 3],
        ],
            dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                               dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def get_downsampled_gt_depth_and_semantic(self, gt_semantics, down_sample):

        B, N, H, W = gt_semantics.shape
        gt_semantics = gt_semantics.view(
            B * N,
            H // down_sample,
            down_sample,
            W // down_sample,
            down_sample,
            1,
        )
        gt_semantics = gt_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantics = gt_semantics.view(-1, down_sample * down_sample)
        gt_semantics = torch.max(gt_semantics, dim=-1).values
        gt_semantics = gt_semantics.view(B * N, H // down_sample,
                                         W // down_sample)
        gt_semantics = F.one_hot(gt_semantics.long(),
                                 num_classes=2).view(-1, 2).float()

        return gt_semantics

    @force_fp32()
    def get_depth_and_semantic_loss(self, semantic_labels, semantic_preds):
        semantic_preds = semantic_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        with autocast(enabled=False):
            pred = semantic_preds
            target = semantic_labels
            alpha = 0.25
            gamma = 2
            pt = (1 - pred) * target + pred * (1 - target)
            focal_weight = (alpha * target + (1 - alpha) *
                            (1 - target)) * pt.pow(gamma)
            semantic_loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
            semantic_loss = semantic_loss.sum() / max(1, len(semantic_loss))
        return self.loss_semantic_weight * semantic_loss

    def forward(self, mlvl_feats, img_metas):

        cam_params = [torch.stack(params) for params in zip(*[meta['cam_params'] for meta in img_metas])]
        mlp_input = self.get_mlp_input(*cam_params).to(device=mlvl_feats[0].device, dtype=torch.float)

        semantics, semantic_masks = [], []
        for lvl, feats in enumerate(mlvl_feats):
            semantic, context = self.height_net(feats, mlp_input)
            semantic = semantic.softmax(dim=1)

            mlvl_feats[lvl] = context

            semantic_mask = (semantic[:, 1:2] >= 0.5)  # 前景
            semantic_masks.append(semantic_mask)  # 语义掩码
            semantics.append(semantic)  # 语义监督

        return mlvl_feats, semantic_masks, semantics

    @force_fp32(apply_to='img_preds')
    def get_loss(self, img_preds, gt_depth, gt_semantic):
        loss_semantic = dict()
        for i, semantic in enumerate(img_preds):
            down_sample = gt_semantic.shape[-1] / semantic.shape[-1]
            depth_labels, semantic_labels = self.get_downsampled_gt_depth_and_semantic(gt_semantic, down_sample)
            loss_semantic = self.get_depth_and_semantic_loss(depth_labels, semantic_labels, semantic)
            loss_semantic['loss_semantic_{}'.format(str(i))] = loss_semantic
        return loss_semantic


@NECKS.register_module()
class SABEVPoolwithMSCT(HeightVT):

    def __init__(self,
                 head_in_channels,
                 head_mid_channels,
                 loss_depth_weight=3.0,
                 loss_semantic_weight=25,
                 depth_threshold=1,
                 semantic_threshold=0.25,
                 **kwargs):
        super(SABEVPoolwithMSCT, self).__init__(**kwargs)
        self.depth_net = MSCThead(head_in_channels, head_mid_channels,
                                  self.D, 2, self.out_channels)

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input, paste_idx, bda_paste) = input[:10]

        out = self.depth_net(x, mlp_input)
        depth = out[0].softmax(dim=1)  # (24, 118, 32, 88)
        semantic = out[1].softmax(dim=1)  # (24, 2, 32, 88)
        tran_feat = out[2]  # (24, 80, 32, 88)
        kept = (depth >= self.depth_threshold) * (semantic[:, 1:2] >= self.semantic_threshold)  # (24, 118, 32, 88)
        return self.view_transform(input[0][1].shape, input, depth, tran_feat, kept, paste_idx, bda_paste), \
            (out[3], out[4], out[0], out[1])

    def get_loss(self, img_preds, gt_depth, gt_semantic):
        depth0 = F.interpolate(img_preds[0], scale_factor=2, mode='bilinear').softmax(1)
        semantic0 = F.interpolate(img_preds[1], scale_factor=2, mode='bilinear').softmax(1)
        depth1 = img_preds[2].softmax(1)
        semantic1 = img_preds[3].softmax(1)
        depth_labels, semantic_labels = \
            self.get_downsampled_gt_depth_and_semantic(gt_depth, gt_semantic)
        loss_depth0, loss_semantic0 = \
            self.get_depth_and_semantic_loss(depth_labels, depth0, semantic_labels, semantic0)
        loss_depth1, loss_semantic1 = \
            self.get_depth_and_semantic_loss(depth_labels, depth1, semantic_labels, semantic1)
        loss_depth = (loss_depth0 + loss_depth1) / 2
        loss_semantic = (loss_semantic0 + loss_semantic1) / 2
        return loss_depth, loss_semantic
