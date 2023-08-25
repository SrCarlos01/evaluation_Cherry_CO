# Copyright (c) OpenMMLab. All rights reserved.

import io
import os
import logging
import argparse

import itertools
import contextlib
import os.path

import numpy as np
from collections import OrderedDict
from terminaltables import AsciiTable

import mmcv
from mmcv import Config, DictAction
from mmcv.utils import print_log

from mmdet.datasets import build_dataset
from mmdet.datasets.api_wrappers import COCOeval
from mmdet.utils import update_data_root

from yolo_utils_eval import load_bbox_file_dicts

from mmdet.datasets import CocoDataset
from mmdet.datasets import DATASETS

@DATASETS.register_module
class SingleClassDataset(CocoDataset):
    CLASSES = ('target',)


@DATASETS.register_module
class RipenessDataset(CocoDataset):
    CLASSES = ('ripe','unripe','green',)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                                 'results saved in pkl format or a folder with .txt labels')
    parser.add_argument('--config', help='Config of the model')
    parser.add_argument('--pkl-results', help='Results in pickle format',default='')
    parser.add_argument('--folder-results', help='Results in yolo like format',default='')


    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='Evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')

    parser.add_argument(
        '--coco-format',
        action='store_true',
        help='Use a .pkl file to parse output predictions if True. A yolo like format will used otherwise')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')

    args = parser.parse_args()
    if args.coco_format:
        assert (os.path.isfile(args.pkl_results)), ('Path to pickle file is invalid for file:\n {:s}'.format(
            args.pkl_results))
    else:
        assert (os.path.isdir(args.folder_results)), ('Path to folder is invalid for path:\n {:s}'.format(
            args.folder_results))
    return args

def evaluate_results(dataset,
                     path_bbox,
                     # results,
                 metric='bbox',
                 logger=None,
                 # jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
    metrics = metric if isinstance(metric, list) else [metric]
    allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
    for metric in metrics:
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

    coco_gt = dataset.coco
    dataset.cat_ids = coco_gt.get_cat_ids(cat_names=dataset.CLASSES)

    # result_files, tmp_dir = dataset.format_results(results, jsonfile_prefix)

    eval_results = evaluate_det_segm(dataset, path_bbox, coco_gt,
                                          metrics, logger, classwise,
                                          proposal_nums, iou_thrs,
                                          metric_items)

    return eval_results


def evaluate_det_segm(dataset,
                      path_bbox_pred,
                     coco_gt,
                     metrics,
                     logger=None,
                     classwise=False,
                     proposal_nums=(100, 300, 1000),
                     iou_thrs=None,
                     metric_items=None):

    if iou_thrs is None:
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    if metric_items is not None:
        if not isinstance(metric_items, list):
            metric_items = [metric_items]

    eval_results = OrderedDict()
    for metric in metrics:
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        # if metric == 'proposal_fast':
        #     if isinstance(results[0], tuple):
        #         raise KeyError('proposal_fast is not supported for '
        #                        'instance segmentation result.')
        #     ar = dataset.fast_eval_recall(
        #         results, proposal_nums, iou_thrs, logger='silent')
        #     log_msg = []
        #     for i, num in enumerate(proposal_nums):
        #         eval_results[f'AR@{num}'] = ar[i]
        #         log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
        #     log_msg = ''.join(log_msg)
        #     print_log(log_msg, logger=logger)
        #     continue

        iou_type = 'bbox' if metric == 'proposal' else metric
        if metric not in ['bbox', 'proposal']:
            raise KeyError(f'{metric} is not in results')

        try:
            predictions = load_bbox_file_dicts(dataset.coco.imgs, path_bbox_pred)
            # if iou_type == 'segm':
            #     # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
            #     # When evaluating mask AP, if the results contain bbox,
            #     # cocoapi will use the box area instead of the mask area
            #     # for calculating the instance area. Though the overall AP
            #     # is not affected, this leads to different
            #     # small/medium/large mask AP results.
            #     for x in predictions:
            #         x.pop('bbox')
            #     warnings.simplefilter('once')
            #     warnings.warn(
            #         'The key "bbox" is deleted for more accurate mask AP '
            #         'of small/medium/large instances since v2.12.0. This '
            #         'does not change the overall mAP calculation.',
            #         UserWarning)
            coco_det = coco_gt.loadRes(predictions)
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger,
                level=logging.ERROR)
            break

        cocoEval = COCOeval(coco_gt, coco_det, iou_type)
        cocoEval.params.catIds = dataset.cat_ids
        cocoEval.params.imgIds = dataset.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')

        if metric == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()

            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                    'AR_m@1000', 'AR_l@1000'
                ]

            for item in metric_items:
                val = float(
                    f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()

            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(dataset.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(dataset.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = dataset.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

    return eval_results


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)

    kwargs = {}
    if args.coco_format:
        outputs = mmcv.load(args.pkl_results)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))
    else:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))

        print(evaluate_results(dataset, args.folder_results, **eval_kwargs))


if __name__ == '__main__':
    main()
