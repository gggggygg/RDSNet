import os
from pathlib import Path

img_path = '/home/weiph/code/ultralytics/Overhead_fisheye/COCO2017_tiny_test/images/train'
label_path = '/home/weiph/code/ultralytics/Overhead_fisheye/COCO2017_tiny_test/labels/train'

img_list = list(Path(os.path.join(img_path)).glob('*.images'))
label_list = list(Path(os.path.join(label_path)).glob('*.txt'))
x=1