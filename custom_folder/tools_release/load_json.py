from pycocotools.coco import COCO

loaf = COCO('../annotations/resolution_1k/instances_train.json')

img_ids = loaf.getImgIds()
print('total {} images'.format(len(img_ids)))
for img_id in img_ids:
    img_info = loaf.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    ann_ids = loaf.getAnnIds(imgIds=img_id)
    anns = loaf.loadAnns(ann_ids)
    print(file_name,'has {} annotations'.format(len(anns)))
    break