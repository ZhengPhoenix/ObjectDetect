import os
from sys import argv

nums = argv[2]
gpu = argv[1]
cmd1 = 'python3 DataAugment/augmentData.py Trainset_full_0408_0519 image_{}w_t0408_f0519 {}0000 '.format(nums, nums)
cmd2 = 'CUDA_VISIBLE_DEVICES={} python3 train.py model_dataset_{}w_t0408_f0519 image_{}w_t0408_f0519/multipleBackgrounds'.format(gpu, nums, nums)

os.system('{} && {}'.format(cmd1, cmd2))
