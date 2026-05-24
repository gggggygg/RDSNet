CUSTOM_OPTIMIZER_SCHEDULER = False
CUSTOM_PLOT_NUMS = False
TRAIN_YMAL = "data/coco.yaml"
custom_LOAFeval_from_cocoeval = False
rapid_loaf: False
val_plot_cnt = 0
val_plot_mod = 1000
val_imgsz = -1
rapid_train_strategy = False
rapid_train_data = 'COCO'
rapid_batsize = 4
training = False
rapid_dark53_bottleneck_kernel = False
rapid_transforms = False
loaf_transforms = False
coco = False
HCM = False
limit_whr = False
keep_label = False
custom_trainer = False
custom_scale = 0.5
multiscale_interval = 10
imgsz = 608
random_imgsz = 608
angle_loss = False
penalty_mask = False
area_thr = 0
mosaic_fish = False
CADC = False
cut_scale_up = False
limit_loss_wh = 0
radial_distortion = False
radial_distortion_k = 1e-6
angle_offset = False
angle_offset_loss = False
keep_label_loaf = False
split_metrics = False
Conv_Sparse_sampling = False  #推理加速用
coco_pretrain_augment = False
Conv_Sparse_sampling_traing = False
fitness_focus_ap50 = False
loaf_transforms_ori = False
location_eval = False
close_degrees = False
close_degrees_epoch = 50
fekv9_transforms = False
nw = None
fisheye8k_eval = None
split_distance_metrics: False
split_seen_un_seen: False
save_all_val_results: False
Conv_sampling_only_inner: False

val_plot_low = False
val_plot_low_limit = None

coco_eval = False
coco_metric = 'AP'
iou_solution = 'LOAF'

cache_path = None