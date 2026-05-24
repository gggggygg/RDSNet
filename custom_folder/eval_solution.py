import os
import numpy as np
from pycocotools import cocoeval
from pycocotools import mask as maskUtils
from collections import defaultdict
from ultralytics.utils.metrics import batch_probiou
from custom_folder.iou_solution import cal_iou
import torch

class YOLOeval(cocoeval.COCOeval):
    def __init__(self, gt_json, dt_json, img0_path, save_dir, iouType='bbox', iou_solution = 'LOAF'):
        # super().__init__()
        assert iouType == 'bbox', 'Only support (rotated) bbox iou type'
        self.gt_json = gt_json
        self.dt_json = dt_json
        self.img_dir = img0_path.split('/')[:-1]
        self.img_dir = '/' + os.path.join(*self.img_dir) + '/'
        self.save_dir = save_dir
        self.iou_solution = iou_solution
        self.now_img_id = ''
        self.save_cnt = 0
        self._preprocess_dt_gt()
        self.params = cocoeval.Params(iouType=iouType)
        self.params.imgIds = sorted([img['id'] for img in self.gt_json['images']])
        self.params.catIds = sorted([cat['id'] for cat in self.gt_json['categories']])
        # Initialize some variables which will be modified later
        self.evalImgs = defaultdict(list)   # per-image per-category eval results
        self.eval     = {}                  # accumulated evaluation results

    def _preprocess_dt_gt(self):
        # We are not using 'id' in ground truth annotations because it's useless.
        # However, COCOeval API requires 'id' in both detections and ground truth.
        # So, add id to each dt and gt in the dt_json and gt_json.
        for i, gt in enumerate(self.gt_json['annotations']):
            gt['id'] = gt.get('id', i+1)
        for i, dt in enumerate(self.dt_json):
            dt['id'] = dt.get('id', i+1)
            # Calculate the areas of detections if there is not. category_id
            dt['area'] = dt.get('area', dt['bbox'][2]*dt['bbox'][3])
            dt['category_id'] = dt.get('category_id', 1)
        # A dictionary mapping from image id to image information
        self.imgId_to_info = {img['id']:img for img in self.gt_json['images']}

    def _prepare(self):
        p = self.params
        gts = [ann for ann in self.gt_json['annotations']]
        dts = self.dt_json

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt.get('ignore', False) or gt.get('iscrowd', False)
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)

    def computeIoU(self, imgId, catId):
        self.now_img_id = imgId
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            raise NotImplementedError()
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]

            # get image width and height
            img = self.imgId_to_info[imgId]
            img_size = (img['height'], img['width'])

            if self.iou_solution == 'YOLO':
                iou_func = lambda x,y: batch_probiou(x,y).cpu().numpy()
            elif self.iou_solution == 'LOAF':
                iou_func = lambda x, y: cal_iou(x, y)
            elif self.iou_solution == 'coco':
                iou_func = lambda x, y: self.coco_iou_rle(x, y, img_size=img_size)

        else:
            raise Exception('unknown iouType for iou computation')

        ious = self.vary_iou(d, g, iou_func, type=self.iou_solution)

        return ious


    def vary_iou(self, d, g, iou_func, type):
        if type == 'YOLO':  #YOLO
            d = np.array(d)
            g = np.array(g)
            if d.shape[0] == 0 or g.shape[0] == 0:
                return np.zeros((d.shape[0], g.shape[0]))
            ious = iou_func(d, g)

        elif type == 'LOAF':  #LOAF
            assert isinstance(d, list) and isinstance(g, list)
            # convert bounding boxes to torch.tensor
            boxes1 = np.array(d).reshape(-1, 5)
            boxes2 = np.array(g).reshape(-1, 5)
            if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
                return np.zeros((boxes1.shape[0], boxes2.shape[0]))

            xyxy1 = xywhr2xyxyxyxy4cocoapi(boxes1).reshape(-1, 8)
            xyxy2 = xywhr2xyxyxyxy4cocoapi(boxes2).reshape(-1, 8)

            d = torch.from_numpy(xyxy1).cuda().to(torch.float32)
            g = torch.from_numpy(xyxy2).cuda().to(torch.float32)

            if d.shape[0] == 0 or g.shape[0] == 0:
                return np.zeros((d.shape[0], g.shape[0]))

            ious, corners1, corners2, u = iou_func(d, g)
            ious = ious.cpu().numpy()

        else:  #RAPID
            if len(d)==0 or len(g) == 0:
                return np.zeros((len(d) , len(g) ))
            ious = iou_func(d, g)

        return ious

    def coco_iou_rle(self, boxes1, boxes2, img_size=2048):
        '''
        Use mask and Run Length Encoding to calculate IOU between rotated bboxes.

        NOTE: rotated bounding boxes format is [cx, cy, w, h, degree].

        Args:
            boxes1: list[list[float]], shape[M,5], 5=(cx, cy, w, h, degree)
            boxes2: list[list[float]], shape[N,5], 5=(cx, cy, w, h, degree)
            img_size: int or list, (height, width)

        Return:
            ious: np.array[M,N], ious of all bounding box pairs
        '''
        assert isinstance(boxes1, list) and isinstance(boxes2, list)
        # convert bounding boxes to torch.tensor
        boxes1 = np.array(boxes1).reshape(-1, 5)
        boxes2 = np.array(boxes2).reshape(-1, 5)
        if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]))

        xyxy1 = xywhr2xyxyxyxy4cocoapi(boxes1).reshape(-1,8).tolist()
        xyxy2 = xywhr2xyxyxyxy4cocoapi(boxes2).reshape(-1,8).tolist()

        h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
        b1 = maskUtils.frPyObjects(xyxy1, h, w)
        b2 = maskUtils.frPyObjects(xyxy2, h, w)
        ious = maskUtils.iou(b1, b2, [0 for _ in b2])

        return ious


def xywhr2xyxyxyxy4cocoapi(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 - vec2
    pt2 = ctr + vec1 + vec2
    pt3 = ctr - vec1 + vec2
    pt4 = ctr - vec1 - vec2
    return stack([pt1, pt2, pt3, pt4], -2)

from .iou_mask import iou_mask, iou_rle

class custom_MWeval():
    """
    Custom evaluation tools for rotated bounding boxes.
    The class name 'MWeval' might be misleading; it doesn't have any relationship with \
        the Mirror Worlds (MW) dataset.
    """

    def __init__(self, gt_json, iou_method='LOAF'):
        """
        Args:
            gt_path (str): path point to the ground truth JSON file
            iou_method (str): method to calculate rotated bbox IoU. 'rle' is recommended.
        """
        assert torch.__version__.startswith('1')
        self.maxDet = 100  # max number of detections per image
        self.iou_thres = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
        self.rec_thres = torch.linspace(0, 1, steps=101)
        self.iou_method = iou_method

        if iou_method == 'mask':
            self.iou_func = iou_mask
        elif iou_method == 'coco':
            self.iou_func = iou_rle
        elif iou_method == 'LOAF':
            self.iou_func = lambda x, y: cal_iou(x, y)
        else:
            raise Exception('unknown IoU method')

        self._prepare(gt_json)

    def _prepare(self, json_data):
        '''
        Load and prepare the ground truth data
        '''
        self.num_gt = len(json_data['annotations'])
        self.img_ids = []
        self.imgid2info = dict()
        self.gts = defaultdict(list)
        for img in json_data['images']:
            self.img_ids.append(img['id'])
            self.imgid2info[img['id']] = img
        # assign annotations to the corresponding image
        for ann in json_data['annotations']:
            self.gts[ann['image_id']].append(ann)

    def evaluate_dtList(self, dt_json, metric='AP', **kwargs):
        """
        Args:
            dt_json: list of dict
            metric: str, could be 'AP', 'F', or 'counting'
            debug (default: False): if True, plot the P-R curve

        Returns:
            eval_str: str
        """
        self.dts = defaultdict(list)
        # load detections
        for dt in dt_json:
            self.dts[dt['image_id']].append(dt)

        if metric == 'AP':
            # traverse all the images, calculate TP and FP in each image
            self._evaluateAll()
            # accumulate and create P-R curve
            self._accumulate(**kwargs)
            if kwargs.get('debug', False):
                import matplotlib.pyplot as plt
                p = self.PRcurve[0, :].numpy()
                r = self.rec_thres.numpy()
                plt.plot(r, p)
                plt.title(f'P-R curve at IoU=0.5. AP_50={self._getAP(0.5):.4f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.show()
                debug = 1
            eval_str = self._summary()
        elif metric == 'F':
            # traverse all the images, calculate TP and FP in each image
            self._evaluateAll()
            # under the criterion of IoU = 0.5, compute TP, FP, precision, recall
            eval_str = self._summaryTPFP()
        elif metric == 'counting':
            eval_str = self._summaryCounting()
        else:
            raise Exception('Unkonwn evaluation metric')

        return eval_str

    def _evaluateAll(self):
        '''
        Traverse all the images and determine the TP and FP in each image
        '''
        tps = []  # list of tensors
        scores = []  # list of tensors
        num_gt = 0  # number of gts in the whole dataset
        for img_id in self.img_ids:
            dts_info, gts_info = self.dts[img_id], self.gts[img_id]

            dts = torch.zeros(len(dts_info), 6)
            for i, dt in enumerate(dts_info):
                dts[i, 0:5] = torch.Tensor(dt['bbox'])
                dts[i, 5] = dt['score']
            # sort the dts by confidence score
            sort_idx = torch.argsort(dts[:, 5], descending=True)
            dts = dts[sort_idx, :]

            num_gt += len(gts_info)
            gts = torch.empty(len(gts_info), 5)
            for i, gt in enumerate(gts_info):
                gts[i, :] = torch.Tensor(gt['bbox'])

            # limit the number of detections in one image
            if len(dts) > self.maxDet:
                dts = dts[0:self.maxDet, :]
            # calculate each detection is TP or FP in every IoU threshold
            img_tp = self._match(dts, gts)
            # self._visualize(img_id, f'../../../COSSY/{self.video_name}', dts, gts)
            # record into the total tps
            tps.append(img_tp)
            scores.append(dts[:, 5])  # confidence

        assert num_gt == self.num_gt
        self.tps, self.scores = tps, scores

    def vary_iou(self, d, g, iou_func, type):
        if type == 'YOLO':  # YOLO
            d = np.array(d)
            g = np.array(g)
            if d.shape[0] == 0 or g.shape[0] == 0:
                return np.zeros((d.shape[0], g.shape[0]))
            ious = iou_func(d, g)
        elif type == 'LOAF':  # LOAF
            d = d.cpu().numpy()
            g = g.cpu().numpy()
            # convert bounding boxes to torch.tensor
            boxes1 = np.array(d).reshape(-1, 5)
            boxes2 = np.array(g).reshape(-1, 5)
            if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
                return np.zeros((boxes1.shape[0], boxes2.shape[0]))

            from ultralytics.utils import ops

            xyxy1 = xywhr2xyxyxyxy4cocoapi(boxes1).reshape(-1, 8)
            xyxy2 = xywhr2xyxyxyxy4cocoapi(boxes2).reshape(-1, 8)

            # if self.save_cnt == 0:
            #     from custom_folder.visual import draw_rotated_boxes
            #     img_path = self.img_dir + self.now_img_id + '.jpg'
            #     draw_rotated_boxes(img_path, xyxy1.copy().tolist(), xyxy2.copy().tolist(), save_dir=self.save_dir)
            #     self.save_cnt += 1

            d = torch.from_numpy(xyxy1).cuda().to(torch.float32)
            g = torch.from_numpy(xyxy2).cuda().to(torch.float32)

            from ultralytics.utils import ops

            if d.shape[0] == 0 or g.shape[0] == 0:
                return np.zeros((d.shape[0], g.shape[0]))

            ious, corners1, corners2, u = iou_func(d, g)
            ious = ious.to(torch.float32)

        else:  # RAPID
            if len(d) == 0 or len(g) == 0:
                return np.zeros((len(d), len(g)))
            ious = iou_func(d, g)
        return ious

    def _match(self, dts, gts):
        '''
        Match dts and gts in one image. This is a bipartite matching based a \
            greedy algorithm. The algorithm is adapted from COCO.

        Args:
            dts: tensor, shape[N,6], rows [x,y,w,h,a,conf]
            gts: tensor, shape[M,5], rows [x,y,w,h,a]
        '''
        assert dts.dim() == 2 and dts.shape[1] == 6
        assert gts.dim() == 2 and gts.shape[1] == 5
        # make sure detections are sorted by confidence
        score_sorted, _ = torch.sort(dts[:, 5], descending=True)
        assert torch.equal(dts[:, 5], score_sorted)

        T = len(self.iou_thres)  # number of IoU thresholds
        dtTP = torch.zeros(T, len(dts), dtype=torch.bool)
        gtmatched = torch.zeros(T, len(gts), dtype=torch.bool)

        if len(dts) == 0 or len(gts) == 0:
            return dtTP
        # compute IoU between prediction and ground truth
        ious = self.vary_iou(dts[:, 0:5], gts, self.iou_func, type=self.iou_method)
        assert ious.shape[0] == len(dts) and ious.shape[1] == len(gts)

        # compute tp, fp for every IoU threshold
        for tidx, t in enumerate(self.iou_thres):
            for dtidx, dt in enumerate(dts):
                # for each detection, traverse all the gt
                best_iou = t  # initial best IoU is the IoU threshold
                best_gt = -1  # -1 -> unmatched
                for gtidx, gt in enumerate(gts):
                    # if this gt is already matched, continue to the next gt
                    if gtmatched[tidx, gtidx]:
                        continue
                    # if the IoU is not better than before, continue
                    if ious[dtidx, gtidx] < best_iou:
                        continue
                    # the IoU is better than before:
                    best_iou = ious[dtidx, gtidx]
                    best_gt = gtidx
                # if this dt matches a gt, it is a TP
                if best_gt >= 0:
                    dtTP[tidx, dtidx] = 1  # the dtidx'th dt is a TP
                    gtmatched[tidx, best_gt] = 1  # the best_gt'th gt is matched
        return dtTP

    def _accumulate(self, **kwargs):
        '''
        Accumulate the TP/FP in all images to create the P-R curve
        '''
        print('accumulating results')
        num_gt = self.num_gt
        tps = torch.cat(self.tps, dim=1)
        # sort all the tps in descending order of score
        scores = torch.cat(self.scores, dim=0)
        scores, sortidx = torch.sort(scores, dim=0, descending=True)
        tps = tps[:, sortidx]
        # False Positive = NOT True Positive
        fps = ~tps
        num_dt = tps.shape[1]
        assert tps.dim() == 2 and tps.shape[0] == len(self.iou_thres)

        # accumulate
        tps, fps = tps.float(), fps.float()
        tp_sum = torch.cumsum(tps, dim=1)
        fp_sum = torch.cumsum(fps, dim=1)
        assert ((tp_sum[:, -1] + fp_sum[:, -1]) == num_dt).all()
        # calculate precision and recall
        precision = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / num_gt
        f1 = 2 * (precision * recall) / (precision + recall)
        # print('p', precision[0,:100])
        # print('r', recall[0,:100])
        if kwargs.get('debug', False):
            import matplotlib.pyplot as plt
            p = precision[0, :].numpy()
            r = recall[0, :].numpy()
            plt.plot(r, p)
            plt.title(f'P-R curve at IoU=0.5 before smoothing')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
            debug = 1

        # initialize the approximate P-R curve for all IoU thresholds
        PRcurve = torch.zeros(len(self.iou_thres), len(self.rec_thres))
        # there is no searchsorted() in pytorch so convert recall to numpy
        recall = recall.numpy()
        for ti, (prec_T, rc_T) in enumerate(zip(precision, recall)):
            assert prec_T.shape[0] == rc_T.shape[0] == num_dt

            # make the Precision monotonically decreasing
            for i in range(num_dt - 1, 0, -1):
                if prec_T[i] > prec_T[i - 1]:
                    prec_T[i - 1] = prec_T[i]
            # find the 101 recall points
            idxs = np.searchsorted(rc_T, self.rec_thres, side='left')
            # fill in the P-R curve
            for ri, pi in enumerate(idxs):
                if pi >= len(prec_T):
                    # reach the upper bound of Recall
                    break
                PRcurve[ti, ri] = prec_T[pi]

        self.PRcurve = PRcurve
        self.APs = self.PRcurve.mean(dim=1)
        self.best_thres = scores[torch.argmax(f1, dim=1)]

    def _summary(self):
        '''
        P-R curve to string for AP
        '''
        if not torch.is_tensor(self.PRcurve):
            raise Exception('Please run accumulate first')

        Template = ' {} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        title = 'Average Precision' if True else 'Average Recall'
        abb = '(AP)' if True else '(AR)'

        s = ''
        s += Template.format(title, abb, '0.5:0.95', 'all', 100, self._getAP()) + '\n'
        s += Template.format(title, abb, '0.5', 'all', 100, self._getAP(0.5)) + '\n'
        s += Template.format(title, abb, '0.75', 'all', 100, self._getAP(0.75))

        l = [f'{self._getAP(iouT):.3f}' for iouT in self.iou_thres]
        l = [float(s) for s in l]
        s += f'\n AP for IoU=0.5, 0.55, 0.6, ..., 0.95: {l}'
        s += f'\n Best confidence thresholds: {self.best_thres}'
        return s

    def _getAP(self, iouThr=None):
        if iouThr:
            idx = self.iou_thres.index(iouThr)
            ap = self.APs[idx]
        else:
            ap = self.APs.mean()
        return ap.item()

    def _summaryTPFP(self):
        '''
        P-R curve to string for TP/FP/F-score
        '''
        print('computing TP, FP, FN, Precision, Recall, and F1 score')
        tps = torch.cat(self.tps, dim=1)
        num_dt = tps.shape[1]
        num_gt = self.num_gt

        tp_num = torch.sum(tps, dim=1).float()
        fp_num = num_dt - tp_num
        fn_num = num_gt - tp_num
        precision = tp_num / num_dt
        recall = tp_num / num_gt
        f1 = 2 * (precision * recall) / (precision + recall)

        assert (fn_num >= 0).all()
        assert f1.shape[0] == len(self.iou_thres)

        s = f'[IoU=0.5] TP={tp_num[0]}, FP={fp_num[0]}, FN={fn_num[0]}, ' \
            f'Precision={precision[0]}, Recall={recall[0]}, F1={f1[0]}'
        return s

    def _summaryCounting(self):
        '''
        P-R curve to string for object (people) counting
        '''
        error = 0
        overcounts = 0
        undercounts = 0
        over_num = 0
        under_num = 0
        gt_num = 0
        for img_id in self.img_ids:
            img_dt = self.dts[img_id]
            img_gt = self.gts[img_id]
            nd = len(img_dt)
            ng = len(img_gt)

            error += abs(nd - ng)
            if nd > ng:
                overcounts += nd - ng
                over_num += 1
            elif nd < ng:
                undercounts += ng - nd
                under_num += 1
            gt_num += ng

        img_num = len(self.img_ids)
        mae_img = error / img_num
        # over_img = overcounts / img_num
        # under_img = undercounts / img_num
        mae_person = error / gt_num
        s = f'[Image num]: {img_num}, [correct num]: {img_num - over_num - under_num} ' \
            f'[over num]: {over_num}, [under num]: {under_num}\n'
        s += f'[MAE/img]: {mae_img:.4f}, [overcount/img]: {overcounts / img_num:.4f}' \
             f', [undercount/img] {undercounts / img_num:.4f}\n'
        s += f'[MAE/p]: {mae_person:.4f}, [overcount/p]: {overcounts / gt_num:.4f}' \
             f', [undercount/p] {undercounts / gt_num:.4f}'
        return s

    def _visualize(self, img_id, img_dir, detections, labels):
        '''
        For debugging
        '''
        import os
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.transforms import Affine2D
        img_name = self.imgid2info[img_id]['file_name']
        img_path = os.path.join(img_dir, img_name)
        imgnp = plt.imread(img_path)

        _, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(imgnp)
        ax.set_axis_off()

        # draw labels
        assert labels.dim() == 2 and labels.shape[1] == 5
        for i, (x, y, w, h, a) in enumerate(labels.cpu()):
            x, y, w, h, a = x.item(), y.item(), w.item(), h.item(), a.item()
            x1, y1 = x - w / 2, y - h / 2
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                     edgecolor='g', facecolor='none')
            t = Affine2D().rotate_deg_around(x, y, degrees=a) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
            ax.text(x, y, f'{i}', color='w', size=6,
                    backgroundcolor='blue')

        # draw detections
        for i, (x, y, w, h, a, conf) in enumerate(detections.cpu()):
            x, y, w, h, a = x.item(), y.item(), w.item(), h.item(), a.item()
            score = conf.item()  # objectness score
            x1, y1 = x - w / 2, y - h / 2
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, linestyle='--',
                                     edgecolor='r', facecolor='none')
            t = Affine2D().rotate_deg_around(x, y, degrees=a) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
            ax.text(x1, y1, f'{conf:.2f}', color='w', size=16, backgroundcolor='none')
            ax.text(x1, y1, f'{i}', color='w', size=6, backgroundcolor='red')

        plt.show()
