import argparse
import os
import os.path as osp
import shutil
import tempfile
import json
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from tools.cityPerson.eval_demo import validate


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('checkpoint_start', type=int, default=1)
    parser.add_argument('checkpoint_end', type=int, default=100)
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.json', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    for i in range(args.checkpoint_start, args.checkpoint_end):
        cfg = mmcv.Config.fromfile(args.config)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            shuffle=False)

        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if not args.mean_teacher:
            while not osp.exists(args.checkpoint + str(i) + '.pth'):
                time.sleep(5)
            while i+1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i+1) + '.pth'):
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth', map_location='cpu')
            model.CLASSES = dataset.CLASSES
        else:
            while not osp.exists(args.checkpoint + str(i) + '.pth.stu'):
                time.sleep(5)
            while i+1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i+1) + '.pth.stu'):
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth.stu', map_location='cpu')
            checkpoint['meta'] = dict()
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES        

        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)

        res = []
        for id, boxes in enumerate(outputs):
            boxes=boxes[0]
            if type(boxes) == list:
                boxes = boxes[0]
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            if len(boxes) > 0:
                for box in boxes:
                    bbox_score = float(box[4])
                    if bbox_score < 0.3:
                        continue
                    temp = dict()
                    temp['image_id'] = id+1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist()
                    temp['score'] = bbox_score
                    res.append(temp)

        with open(args.out, 'w') as f:
            json.dump(res, f)

        MRs = validate('datasets/CityPersons/val_gt.json', args.out)
        print('Checkpoint %d: [Reasonable: %.2f%%], [Reasonable_Small: %.2f%%], [Heavy: %.2f%%], [All: %.2f%%]'
              % (i, MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))


if __name__ == '__main__':
    main()
