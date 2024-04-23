# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

import cv2
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint

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
    dataset = build_dataset(cfg.data.test)
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
    for i, data in enumerate(data_loader):
        data = scatter(data, [0])[0]
        with torch.no_grad():
            img, img_metas, canvas = data['img'], data['img_metas'], data['canvas']
            _, semantic = model.module.extract_feat(img=img, img_metas=img_metas)

            semantic = semantic.softmax(dim=1)
            semantic_mask = (semantic[:, 1:2] >= 0.25)  # (6, 1, 16, 44)
            semantic_mask = F.interpolate(semantic_mask.float(), scale_factor=16, mode='nearest'). \
                bool().permute(0, 2, 3, 1).repeat(1, 1, 1, 3)  # (6, 256, 704, 3)

            canvas = torch.stack(canvas, dim=0).squeeze(1)
            for img_id, canvas_img in enumerate(canvas):
                cv2.imwrite('o_{}.jpg'.format(str(img_id)), canvas_img.cpu().numpy())

            canvas = canvas * semantic_mask

            for img_id, canvas_img in enumerate(canvas):
                cv2.imwrite('{}.jpg'.format(str(img_id)), canvas_img.cpu().numpy())

        break


if __name__ == '__main__':
    main()
