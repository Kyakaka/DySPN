seed = 7240
n_device = 8
# dataset
KITTI_PATH = '/media/lin/dataset/kitti_depth_completion'
split_train = "dataset/index_files/kitti_dc_s_train.txt"
split_val = "dataset/index_files/kitti_dc_s_val.txt"
test_only = False
augment = True
num_sample = 0
patch_height = 240
patch_width = 1216
top_crop = 100
# base network
norm_depth = [0.0, 100.0]
basemodel = "v1"
resnet = "res34"
sto_depth = True
pretrain_weight = None
val_output = True
resume_weight = None
# pretrain_weight = "lightning_logs/version_118/checkpoints/last.ckpt"
# resume_weight = "weight/version_48/checkpoints/epoch=38-RMSE=0.7239.ckpt"
# SPN
spn_enable = True
spn_module = 'dyspn'
num_neighbor = 5
iteration = 6
assert iteration == 6
# loss
w_1 = 1.0
w_2 = 1.0
# dataloader
n_thread = 8
n_batch = 4
# met
eval_range = None
# optimizer
learning_rates = 5e-4
w_weight_decay = 0
epochs = 100
step = 40
if test_only:
    n_device = 1
    top_crop = 0
    pretrain_weight = "weight/version_48/checkpoints/epoch=38-RMSE=0.7239.ckpt"
    resume_weight = None
