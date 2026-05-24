import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from ultralytics.data.augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

import os

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"

"""
Dataset4YoloObb(img_dir, json_path, img_size=608, augmentation=True, only_person=True, debug_mode=False)

dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )

"""



class Dataset4YoloObb(BaseDataset):
    def __init__(self, *args, data=None, **kwargs):
        self.use_obb = True
        self.data = data
        self.max_labels = 50

        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path("./labels.cache").

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages

        print(f"{self.prefix}Scanning {path.parent / path.stem}...")

        ###################from json get labels###################

        imgid2path = dict()
        imgid2anns = defaultdict(list)
        imgid2label = defaultdict(list)
        catids = []
        only_person = True
        img_ids = []

        self.coco = False
        print(f'Loading annotations {self.label_files} into memory...')
        with open(self.label_files, 'r') as f:
            json_data = json.load(f)
        for ann in json_data['annotations']:
            img_id = ann['image_id']
            # get width and height
            if len(ann['bbox']) == 4:
                # using COCO dataset. 4 = [x1,y1,w,h]
                self.coco = True
                # convert COCO format: x1,y1,w,h to x,y,w,h
                ann['bbox'][0] = ann['bbox'][0] + ann['bbox'][2] / 2
                ann['bbox'][1] = ann['bbox'][1] + ann['bbox'][3] / 2
                ann['bbox'].append(1/2*np.pi)
            else:
                # using rotated bounding box datasets. 5 = [cx,cy,w,h,angle]
                assert len(ann['bbox']) == 5, 'Unknown bbox format' # x,y,w,h,a
            ann['bbox'] = torch.Tensor(ann['bbox'])
            imgid2anns[img_id].append(ann)
        for img in json_data['images']:
            img_id = img['id']
            assert img_id not in imgid2path
            anns = imgid2anns[img_id]
            # if there is crowd gt, skip this image
            if self.coco and any(ann['iscrowd'] for ann in anns):
                continue
            # if only for person detection
            if only_person:
                # select the images which contain at least one person
                if not any(ann['category_id']==1 for ann in anns):
                    continue
                # and ignore all other categories
                imgid2anns[img_id] = [a for a in anns if a['category_id']==1]
            img_ids.append(img_id)
            imgid2path[img_id] = os.path.join(self.img_path, img['file_name'])
            # self.imgid2info[img['id']] = img
        catids = [cat['id'] for cat in json_data['categories']]
        if self.coco:
            print('Training on perspective images; adding angle to BBs')
        else:
            assert only_person


        ############################process labels#################################


        for img_id in img_ids:
            img_path = imgid2path[img_id]
            self.coco = True if 'COCO' in img_path else False
            img_shape = self.get_shape(imgid2path[img_id]) #hw

            # load unnormalized annotation
            annotations = imgid2anns[img_id]
            gt_num = len(annotations)
            # labels shape(50, 5), 5 = [x, y, w, h, angle]
            labels = torch.zeros(self.max_labels, 5)
            categories = torch.zeros(self.max_labels, dtype=torch.int64)
            li = 0
            for ann in annotations:
                if only_person and ann['category_id'] != 1:
                    continue
                area = ann['bbox'][2]*ann['bbox'][3] / img_shape[1] / img_shape[0]
                if only_person and self.coco and area <= 0.001:
                    # import matplotlib.pyplot as plt
                    # plt.imshow(np.array(img))
                    # plt.show()
                    continue
                # assert ann['category_id'] == 1, 'only support person object'
                if li >= 50:
                    print(only_person)
                    print(categories)
                    break
                labels[li,:] = ann['bbox']
                categories[li] = catids.index(ann['category_id'])
                li += 1
            if only_person:
                assert (categories == 0).all()
            gt_num = li

            labels[:gt_num] = normalize_bbox(labels[:gt_num], img_shape[1], img_shape[0])

            imgid2label[img_id] = [labels, categories, gt_num]

            # x,y,w,h: 0~1, angle: pi/2

        self.im_files = []

        for img_id in img_ids:
            x['labels'].append({
                "im_file": imgid2path[img_id],
                "shape": self.get_shape(imgid2path[img_id]),
                "cls": imgid2label[img_id][1][:imgid2label[img_id][2]],  # n, 1
                "bboxes": imgid2label[img_id][0][:imgid2label[img_id][2]],  # n, 5
                "segments": [],
                "keypoints": False,
                "normalized": True,
                "bbox_format": "xywhr",
            })

            self.im_files.append(imgid2path[img_id])

        print('Scanning end')

        #############################################################
        if nf :=len(x['labels']) == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash([self.label_files + self.img_path])
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_shape(self, im_file):
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        return shape

    def img2label_paths(self, im_files):
        if 'COCO2017' in im_files:
            return '/ric_nas/COCO2017/annotations/instances_train2017.json'
        return './images/tiny_val/one.json'

    def get_img_files(self, img_dir):
        """Returns list of image files."""
        if 'COCO2017' in img_dir:
            return '/data2t2/dataset/COCO2017/train2017'
        return './images/tiny_val/one'

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = self.img2label_paths(self.im_files)
        if 'COCO2017' in self.label_files:
            cache_path = 'dataset/cache/coco/train.cache'
        else:
            cache_path = 'dataset/cache/coco/val.cache'

        # cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([self.label_files + self.img_path])  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def normalize_bbox(xywha, w, h, max_angle=1):
    '''
    Normalize bounding boxes to 0~1 range

    Args:
        xywha: torch.tensor, bounding boxes, shape(...,5)
        w: image width
        h: image height
        max_angle: the angle will be divided by max_angle
    '''
    assert torch.is_tensor(xywha)

    if xywha.dim() == 1:
        assert xywha.shape[0] == 5
        xywha[0] /= w
        xywha[1] /= h
        xywha[2] /= w
        xywha[3] /= h
        xywha[4] /= max_angle
    elif xywha.dim() == 2:
        assert xywha.shape[1] == 5
        xywha[:, 0] /= w
        xywha[:, 1] /= h
        xywha[:, 2] /= w
        xywha[:, 3] /= h
        xywha[:, 4] /= max_angle
    else:
        raise Exception('unkown bbox format')

    return xywha