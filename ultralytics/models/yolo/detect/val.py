# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from custom_folder.visual import draw_rotated_boxes_from_tensor
import config
from ultralytics.utils.metrics import ap_per_class, Metric
from collections import defaultdict
import json
from math import cos, sin
from scipy.optimize import least_squares
from scipy.optimize import root_scalar

class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.is_coco = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt")  # is COCO
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
        if config.split_metrics:
            self.stats_all_split = {}

        if config.split_distance_metrics:
            self.stats_all_for_nmf = {}
            self.stats_split_nmf = {'near':dict(tp=[], conf=[], pred_cls=[], target_cls=[]), 'middle':dict(tp=[], conf=[], pred_cls=[], target_cls=[]), 'far':dict(tp=[], conf=[], pred_cls=[], target_cls=[])}

        if config.split_seen_un_seen:
            self.stats_all_for_sus = {}

        if config.location_eval:
            self.stats_positional_error = {}

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        if config.fisheye8k_eval:
            return ops.non_max_suppression_fekv9(
                preds,
                0.5,
                0.5,
                labels=self.lb,
                multi_label=True,
                agnostic=self.args.single_cls,
                max_det=self.args.max_det,
            )
        else:
            return ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=True,
                agnostic=self.args.single_cls,
                max_det=self.args.max_det,
            )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)

                    if config.split_metrics:  #按场景划分
                        split_key = batch["im_file"][si].split("/")[-1].rsplit('.', 1)[0].rsplit('_', 1)[0]  # e. High_activity
                        if split_key not in self.stats_all_split:
                            self.stats_all_split[split_key] = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
                        for k in self.stats.keys():
                            self.stats_all_split[split_key][k].append(stat[k])

                    if config.split_distance_metrics:  #按与相机水平距离划分
                        matched_gt_idx = np.full(0, fill_value=-1, dtype=int)

                        img_file_name = batch["im_file"][si].split('/')[-1]
                        split_name = batch["im_file"][si].split('/')[-2]
                        self.stats_all_for_nmf[img_file_name] = {
                            'matched_gt_idx': matched_gt_idx,
                            'predn': np.full((0,7), fill_value=0, dtype=float),
                            'bbox': bbox.cpu().numpy(),
                            'cls': cls.cpu().numpy(),
                            'stat': stat
                        }
                        self.stats_all_for_nmf['split_name'] = split_name

                    if config.split_seen_un_seen:  #按见过和没见过划分
                        img_file_name = batch["im_file"][si].split('/')[-1]
                        split_name = batch["im_file"][si].split('/')[-2]
                        self.stats_all_for_sus[img_file_name] = {
                            'stat': stat
                        }
                        self.stats_all_for_sus['split_name'] = split_name

                # self.pred_to_json(None, batch["im_file"][si], batch["ori_shape"][si], batch["resized_shape"][si], batch["ratio_pad"][si])
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                if config.location_eval:
                    stat["tp"], matched_gt_idx = self._process_batch(predn, bbox, cls)
                    match_pred_obb = predn[stat["tp"][:,0]]
                    match_label_obb = bbox[matched_gt_idx][matched_gt_idx>-1]

                    img_file_name = batch["im_file"][si].split('/')[-1]
                    split_name = batch["im_file"][si].split('/')[-2]

                    self.stats_positional_error[img_file_name] = {
                        "matched_pred_obb": match_pred_obb.cpu().numpy(),
                        "matched_label_obb": match_label_obb.cpu().numpy(),
                    }
                    self.stats_positional_error['split_name'] = split_name
                elif config.split_distance_metrics: #按与相机水平距离划分
                    stat["tp"], matched_gt_idx  = self._process_batch(predn, bbox, cls)

                    img_file_name = batch["im_file"][si].split('/')[-1]
                    split_name = batch["im_file"][si].split('/')[-2]
                    self.stats_all_for_nmf[img_file_name] = {
                        'matched_gt_idx': matched_gt_idx,
                        'predn': predn.cpu().numpy(),
                        'bbox': bbox.cpu().numpy(),
                        'cls': cls.cpu().numpy(),
                        'stat': stat
                    }
                    self.stats_all_for_nmf['split_name'] = split_name
                else:
                    stat["tp"] = self._process_batch(predn, bbox, cls)

                if config.save_all_val_results:
                    draw_rotated_boxes_from_tensor(torch.cat([predn[:, :4], predn[:, -1:]], dim=-1), bbox, path=batch['im_file'][si],
                                                   save_dir=self.save_dir, conf=predn[:, 4])

                if config.training==False and sum(stat['tp'][:,0])/bbox.shape[0] < 0.75 and config.val_plot_low:  #for box   #####################modified###############
                    if config.val_plot_low_limit is not None:
                        if sum(stat['tp'][:,0])/bbox.shape[0] < config.val_plot_low_limit:
                            draw_rotated_boxes_from_tensor(torch.cat([predn[:, :4], predn[:, -1:]], dim=-1), bbox, path=batch['im_file'][si],
                                                           save_dir=self.save_dir, conf=predn[:, 4])
                    else:
                        draw_rotated_boxes_from_tensor(torch.cat([predn[:, :4], predn[:, -1:]], dim=-1), bbox,  path = batch['im_file'][si], save_dir = self.save_dir, conf = predn[:,4])

                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            if config.split_metrics: #按场景划分
                split_key = batch["im_file"][si].split("/")[-2].rsplit('.', 1)[0].rsplit('_', 1)[0]  # e. High_activity
                if split_key not in self.stats_all_split:
                    self.stats_all_split[split_key] = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
                for k in self.stats.keys():
                    self.stats_all_split[split_key][k].append(stat[k])

            if config.split_seen_un_seen: #按见过和没见过划分
                img_file_name = batch["im_file"][si].split('/')[-1]
                split_name = batch["im_file"][si].split('/')[-2]
                self.stats_all_for_sus[img_file_name] = {
                    'stat': stat
                }
                self.stats_all_for_sus['split_name'] = split_name

            # Save
            if self.args.save_json:  ##############################modified############################3
                self.pred_to_json(predn, batch["im_file"][si], batch["ori_shape"][si], batch["resized_shape"][si], batch["ratio_pad"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.nc
        )  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def split_metrics(self):
        box = Metric()
        split_mean_mp, split_mean_mr, split_mean_map50, split_mean_map, split_mean_map75 = 0,0,0,0,0
        cnt_split = 0
        split_name = []

        for split, split_value in self.stats_all_split.items():
            cnt_split+=1
            split_name.append(split)
            split_stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in split_value.items()}  # to numpy
            if len(split_stats) and split_stats["tp"].any():
                split_mp, split_mr, split_map50, split_map, split_map75 = self.ap_per_class(box = box, **split_stats)
                split_mean_mp += split_mp
                split_mean_mr += split_mr
                split_mean_map50 += split_map50
                split_mean_map += split_map
                split_mean_map75 += split_map75

        split_mean_mp/= cnt_split
        split_mean_mr/= cnt_split
        split_mean_map50/= cnt_split
        split_mean_map/= cnt_split
        split_mean_map75/= cnt_split

        F1 = 2 * split_mean_mp * split_mean_mr / (split_mean_mp + split_mean_mr + 1e-16)
        fitness = split_mean_map50*0.9 + F1*0.1 #主要关注map50

        print("note !!!!! under two lines are the real Metrics !!! Split Metrics with scenes, follow ##RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images## method.")
        print("################################################real metrics############################################")
        print(f"Split Metrics Mean, name: {', '.join(split_name)}")
        print(f"Mean Precision: {split_mean_mp:.3f}, Mean Recall: {split_mean_mr:.3f}, Mean F-score: {2*split_mean_mp*split_mean_mr/(split_mean_mp+split_mean_mr):.3f}, Mean map50: {split_mean_map50:.3f}, Mean map: {split_mean_map:.3f}, Mean map75: {split_mean_map75:.3f}") if cnt_split > 0 else print("No split metrics available.")
        print("################################################real metrics############################################")
        return fitness

    def split_distance_metrics(self):
        label_json_path_root = 'datasets/LOAF_YOLO_xyxy_from_rbox/annotations/resolution_1k'
        split_name = self.stats_all_for_nmf['split_name']
        label_json_name = f'instances_{split_name}.json'
        label_json_path = label_json_path_root + '/' + label_json_name

        imgid2anns = defaultdict(list)

        with open(label_json_path, 'r') as f:
            label_json = json.load(f)
            for ann in label_json['annotations']:
                imgid2anns[ann['image_id']].append(ann)

            for img in label_json['images']:
                if img['file_name'] in self.stats_all_for_nmf:
                    self.stats_all_for_nmf[img['file_name']].update({'anns': imgid2anns[img['id']]})  #得到每张图像的annotations

        coeffs_json_path = "custom_folder/calibrate/fisheye_fit_results.json"
        with open(coeffs_json_path, 'r') as f:
            coeffs_data = json.load(f)
        coeffs = np.array(coeffs_data['14']["coeffs"], dtype=np.float64)  #相机畸变参数

        for file_name, data in self.stats_all_for_nmf.items():  #依据与相机的距离分类，分别计算AP
            if file_name == 'split_name':
                continue

            bbox = data['bbox']
            feet_point_label = []
            for i_box in range(data['bbox'].shape[0]):
                feet_point = self.find_feet_point(data['bbox'][i_box])
                feet_point_label.append(feet_point)

            pred_box = data['predn']
            feet_point_pred = []
            for i_pred_box in range(pred_box.shape[0]):
                feet_point = self.find_feet_point(pred_box[i_pred_box][[0,1,2,3,6]])
                feet_point_pred.append(feet_point)

            ann = data['anns']

            label_nmf = np.zeros(len(feet_point_label))  # 1-near, 2-middle, 3-far

            for i_feet_point in range(len(feet_point_label)):
                label_person_pixel_location = feet_point_label[i_feet_point]  # feet point in pixel coordinates
                camera_height = ann[0]['camera_height']  # camera height
                R, fai = self.get_world_location_R_from_pixel(label_person_pixel_location, coeffs, camera_height)  #label_person_world_location_polar
                R = R/100 # cm to m

                if R<=10: #near
                    label_nmf[i_feet_point] = 1
                elif R<=20: #middle
                    label_nmf[i_feet_point] = 2
                else: #far
                    label_nmf[i_feet_point] = 3

            detected_nmf = np.zeros(len(feet_point_pred))  # 1-near, 2-middle, 3-far
            matched_gt_idx = data['matched_gt_idx']

            for i_feet_point in range(len(feet_point_pred)):
                if matched_gt_idx[i_feet_point] != -1:  # 匹配上，按gt划分
                    detected_nmf[i_feet_point] = label_nmf[matched_gt_idx[i_feet_point]]
                else:
                    detected_person_pixel_location = feet_point_pred[i_feet_point]
                    camera_height = ann[0]['camera_height']  # camera height
                    R, fai = self.get_world_location_R_from_pixel(detected_person_pixel_location, coeffs, camera_height)  #label_person_world_location_polar
                    R = R / 100  # cm to m

                    if R<=10: #near
                        detected_nmf[i_feet_point] = 1
                    elif R<=20: #middle
                        detected_nmf[i_feet_point] = 2
                    else: #far
                        detected_nmf[i_feet_point] = 3

            stat_near = {'conf':[], 'pred_cls':[], 'tp':[], 'target_cls':[]}
            stat_middle = {'conf':[], 'pred_cls':[], 'tp':[], 'target_cls':[]}
            stat_far = {'conf':[], 'pred_cls':[], 'tp':[], 'target_cls':[]}

            for i_label in range(len(label_nmf)):
                if label_nmf[i_label] == 1:
                    stat_near['target_cls'].append(0)
                elif label_nmf[i_label] == 2:
                    stat_middle['target_cls'].append(0)
                elif label_nmf[i_label] == 3:
                    stat_far['target_cls'].append(0)

            for i_detect in range(len(detected_nmf)):
                if detected_nmf[i_detect] == 1:
                    stat_near['conf'].append(data['stat']['conf'][i_detect].cpu().numpy())
                    stat_near['pred_cls'].append(0)
                    stat_near['tp'].append(data['stat']['tp'][i_detect].cpu().numpy())
                elif detected_nmf[i_detect] == 2:
                    stat_middle['conf'].append(data['stat']['conf'][i_detect].cpu().numpy())
                    stat_middle['pred_cls'].append(0)
                    stat_middle['tp'].append(data['stat']['tp'][i_detect].cpu().numpy())
                elif detected_nmf[i_detect] == 3:
                    stat_far['conf'].append(data['stat']['conf'][i_detect].cpu().numpy())
                    stat_far['pred_cls'].append(0)
                    stat_far['tp'].append(data['stat']['tp'][i_detect].cpu().numpy())

            for key, value in stat_near.items():
                if len(value)==0:
                    value = np.array([], dtype=np.float32).reshape(0, self.niou) if key=='tp' else np.array([], dtype=np.float32)
                self.stats_split_nmf['near'][key].append(np.array(value))

            for key, value in stat_middle.items():
                if len(value)==0:
                    value = np.array([], dtype=np.float32).reshape(0, self.niou) if key=='tp' else np.array([], dtype=np.float32)
                self.stats_split_nmf['middle'][key].append(np.array(value))

            for key, value in stat_far.items():
                if len(value)==0:
                    value = np.array([], dtype=np.float32).reshape(0, self.niou) if key=='tp' else np.array([], dtype=np.float32)
                self.stats_split_nmf['far'][key].append(np.array(value))

        box = Metric()
        split_ap_n, split_ap_m, split_ap_f = 0, 0, 0

        split_stats = {k: np.concatenate(v, 0) for k, v in self.stats_split_nmf['near'].items()}  # to numpy
        split_mp, split_mr, split_map50, split_map, split_map75 = self.ap_per_class(box=box, **split_stats)
        split_ap_n = split_map
        print(f"Near: Precision: {split_mp:.3f}, Recall: {split_mr:.3f}, map50: {split_map50:.3f}, map: {split_map:.3f}, map75: {split_map75:.3f}")

        split_stats = {k: np.concatenate(v, 0) for k, v in self.stats_split_nmf['middle'].items()}  # to numpy
        split_mp, split_mr, split_map50, split_map, split_map75 = self.ap_per_class(box=box, **split_stats)
        split_ap_m = split_map
        print(f"Middle: Precision: {split_mp:.3f}, Recall: {split_mr:.3f}, map50: {split_map50:.3f}, map: {split_map:.3f}, map75: {split_map75:.3f}")

        split_stats = {k: np.concatenate(v, 0) for k, v in self.stats_split_nmf['far'].items()}  # to numpy
        split_mp, split_mr, split_map50, split_map, split_map75 = self.ap_per_class(box=box, **split_stats)
        split_ap_f = split_map
        print(f"Far: Precision: {split_mp:.3f}, Recall: {split_mr:.3f}, map50: {split_map50:.3f}, map: {split_map:.3f}, map75: {split_map75:.3f}")

        print(f"Distance-based Metrics, name: {split_name}")

        print(f"map_near: {split_ap_n:.3f}, map_middle: {split_ap_m:.3f}, map_far: {split_ap_f:.3f}")
        return split_ap_n, split_ap_m, split_ap_f

    def split_seen_un_seen(self):
        label_json_path_root = 'datasets/LOAF_YOLO_xyxy_from_rbox/annotations/resolution_1k'  #这里给个宏吧，或者在config里面定义
        split_name = self.stats_all_for_sus['split_name']
        label_json_name_seen = f'instances_{split_name}-seen.json'
        label_json_path_seen = label_json_path_root + '/' + label_json_name_seen
        label_json_name_unseen = f'instances_{split_name}-unseen.json'
        label_json_path_unseen = label_json_path_root + '/' + label_json_name_unseen

        img_set = {'seen': set(), 'unseen': set()}

        with open(label_json_path_seen, 'r') as f:
            label_json = json.load(f)
            for img in label_json['images']:
                img_set['seen'].add(img['file_name'])

        with open(label_json_path_unseen, 'r') as f:
            label_json = json.load(f)
            for img in label_json['images']:
                img_set['unseen'].add(img['file_name'])

        stat_seen = {'conf':[], 'pred_cls':[], 'tp':[], 'target_cls':[]}
        stat_unseen = {'conf':[], 'pred_cls':[], 'tp':[], 'target_cls':[]}
        for file_name, data in self.stats_all_for_sus.items():
            if file_name == 'split_name':
                continue

            if file_name in img_set['seen']:
                for key, value in data['stat'].items():
                    stat_seen[key].append(value)
            elif file_name in img_set['unseen']:
                for key, value in data['stat'].items():
                    stat_unseen[key].append(value)

        box = Metric()
        split_stats_seen = {k: torch.cat(v, 0).cpu().numpy() for k, v in stat_seen.items()}  # to numpy
        split_mp, split_mr, split_map50, split_map, split_map75 = self.ap_per_class(box=box, **split_stats_seen)
        print(f"Seen: Precision: {split_mp:.3f}, Recall: {split_mr:.3f}, map50: {split_map50:.3f}, map: {split_map:.3f}, map75: {split_map75:.3f}")
        split_stats_unseen = {k: torch.cat(v, 0).cpu().numpy() for k, v in stat_unseen.items()}  # to numpy
        split_mp, split_mr, split_map50, split_map, split_map75 = self.ap_per_class(box=box, **split_stats_unseen)
        print(f"Unseen: Precision: {split_mp:.3f}, Recall: {split_mr:.3f}, map50: {split_map50:.3f}, map: {split_map:.3f}, map75: {split_map75:.3f}")
        print(f"Seen/Unseen Metrics, name: {split_name}")


    def find_feet_point(self, box, image_center=(512, 512)):
        """
        给定旋转框（OBB），返回距离图像中心最近的边中点，作为“脚下位置”

        Args:
            box: [x_ctr, y_ctr, w, h, angle] 形式
            image_center: (u0, v0)，图像中心像素坐标

        Returns:
            min_distance_point: np.ndarray, 形如 [x, y]
        """
        x_ctr, y_ctr, w, h, angle = box
        ctr = np.array([x_ctr, y_ctr])

        cos_a, sin_a = cos(angle), sin(angle)
        vec1 = np.array([w / 2 * cos_a, w / 2 * sin_a])
        vec2 = np.array([-h / 2 * sin_a, h / 2 * cos_a])

        # 计算四个角点
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2

        # 四条边的中点
        midpoints = np.array([
            (pt1 + pt2) / 2,
            (pt2 + pt3) / 2,
            (pt3 + pt4) / 2,
            (pt4 + pt1) / 2
        ])  # shape (4, 2)

        # 计算每个中点到图像中心的距离
        distances = np.linalg.norm(midpoints - np.array(image_center), axis=1)

        # 找到最小距离的那个点
        min_idx = np.argmin(distances)
        return midpoints[min_idx]

    def get_world_location_R_from_pixel(self, pixel_location, coeffs, camera_height, image_center=(512,512)):
        r = np.linalg.norm(pixel_location - np.array(image_center))  # 计算像素坐标到图像中心的距离

        def theta_from_r(r_value, coeffs, theta_max=np.pi / 2):
            """
            根据 r 和畸变系数 coeffs，求解对应的 theta。
            参数:
                r_value: float，输入的像素半径 r
                coeffs: list 或 ndarray，拟合得到的畸变模型系数 [k1, k2, ..., kn]
                theta_max: 最大可能的 θ 值，默认 π/2（可根据视场调大）

            返回:
                theta: float, 入射角 θ（单位：弧度）
            """

            def distortion_function(theta):
                return sum(coeffs[i] * theta ** (2 * i + 1) for i in range(len(coeffs))) - r_value

            # 数值解方程 f(θ) = 0，初始区间 [0, theta_max]
            # sol = root_scalar(distortion_function, bracket=[0, theta_max], method='brentq')

            try:
                sol = root_scalar(distortion_function, bracket=[0, theta_max], method='brentq')

            except ValueError:
                print(f"ValueError: f(a) and f(b) must have different signs for r = {r_value}. Trying a larger theta_max.")

            if sol.converged:
                return sol.root
            else:
                raise RuntimeError(f"Failed to solve θ for r = {r_value}")

        theta = theta_from_r(r, coeffs)
        R = camera_height*np.tan(theta)

        def compute_fai(u, v, u0=512, v0=512):
            du = u - u0
            dv = v - v0
            fai = np.arctan2(-dv, du)  # 注意是 -dv，确保 Y 轴向下变为向上

            fai_deg = np.degrees(fai)
            if fai_deg < 0:
                fai_deg += 360  # 转为 [0, 360) 度范围
            return fai_deg  # 单位是角度，逆时针为正，右方为0

        fai = compute_fai(pixel_location[0], pixel_location[1], image_center[0], image_center[1])

        return R,fai  # 返回世界坐标系下的 R 值

    def polar_distance(self, polar1, polar2):
        """
        计算两个极坐标点在笛卡尔空间下的距离（单位：米）

        Args:
            polar1: [R1_cm, theta1_deg]
            polar2: [R2_cm, theta2_deg]

        Returns:
            distance_m: float，单位为米
        """
        # 解包
        R1_cm, theta1_deg = polar1
        R2_cm, theta2_deg = polar2

        # 单位换算：cm → m
        R1 = R1_cm / 100.0
        R2 = R2_cm / 100.0

        # 角度 → 弧度
        theta1 = np.radians(theta1_deg)
        theta2 = np.radians(theta2_deg)

        # 极坐标 → 笛卡尔坐标
        x1, y1 = R1 * np.cos(theta1), R1 * np.sin(theta1)
        x2, y2 = R2 * np.cos(theta2), R2 * np.sin(theta2)

        # 欧几里得距离
        return np.linalg.norm([x1 - x2, y1 - y2])

    def ap_per_class(self, box, tp, conf, pred_cls, target_cls):
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            names = self.names,
        )[2:]

        box.nc = 1
        box.update(results)

        # split_mp, split_mr, split_map50, split_map, split_map75 =

        return box.mp, box.mr, box.map50, box.map, box.map75

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def pred_to_json(self, predn, filename, *args):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, "bbox")
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
