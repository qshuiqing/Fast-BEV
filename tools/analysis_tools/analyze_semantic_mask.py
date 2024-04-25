# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path
from os import path as osp

import cv2
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint

from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--samples', default=2000, type=int, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=10, help='interval of logging')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    if osp.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # benchmark with several samples and take the average
    for sample_id, data in enumerate(data_loader):
        if not os.path.exists('vis/{}'.format(sample_id)):
            os.mkdir('vis/{}'.format(sample_id))
        data = scatter(data, [0])[0]
        with torch.no_grad():
            img, img_metas, canvas = data['img'], data['img_metas'], data['canvas']
            _, semantic = model.module.extract_feat(img=img, img_metas=img_metas)

            semantic_mask = (semantic[:, 1:2] >= 0.25)  # (6, 1, 16, 44)
            stride = int(img.shape[-1] / semantic_mask.shape[-1])  # 16
            semantic_mask = F.interpolate(semantic_mask.float(), scale_factor=stride, mode='nearest'). \
                bool().permute(0, 2, 3, 1)  # (6, 256, 704, 1)

            _canvas = torch.stack(canvas, dim=0).squeeze(1)
            for img_id, canvas_img in enumerate(_canvas):
                cv2.imwrite('vis/{}/{}_o.jpg'.format(sample_id, img_id), canvas_img.cpu().numpy())

            _canvas = _canvas * semantic_mask

            bboxes = data['gt_bboxes_3d'][0]
            lidar2imgs = img_metas[0]['lidar2img']['extrinsic']
            for img_id in range(len(_canvas)):
                new_img = draw_lidar_bbox3d_on_img(bboxes, _canvas[img_id].cpu().numpy(), lidar2imgs[img_id],
                                                   dict())
                cv2.imwrite('vis/{}/{}.jpg'.format(sample_id, img_id), new_img)


if __name__ == '__main__':
    main()
