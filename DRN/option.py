import DRN.utility as utility
import numpy as np

class parse_args():
    on_server = False
    n_threads = 6
    cpu = False
    n_GPUs = 1
    seed=1
    data_dir='data_path'
    data_train='DF2K'
    data_test='Set5'
    data_range='1-800/801-810'
    scale=8
    patch_size=384
    rgb_range=255
    n_colors=3
    no_augment='store_true'
    model='DRN-S'
    pre_train='.'
    pre_train_dual='.'
    n_blocks=30
    n_feats=16
    negval=0.2
    test_every=1000
    epochs=1000
    batch_size=32
    self_ensemble=False
    test_only=False
    lr=1e-4
    eta_min=1e-7
    beta1=0.9
    beta2=0.999
    epsilon=1e-8
    weight_decay=0
    loss='1*L1'
    skip_threshold='1e6'
    dual_weight=0.1
    save='./experiment/test/'
    print_every=100
    save_results=False

args = parse_args()

# utility.init_model(args)
#
# # scale = [2,4] for 4x SR to load data
# # scale = [2,4,8] for 8x SR to load data
# args.scale = [pow(2, s+1) for s in range(int(np.log2(args.scale)))]
#
# for arg in vars(args):
#     if vars(args)[arg] == 'True':
#         vars(args)[arg] = True
#     elif vars(args)[arg] == 'False':
#         vars(args)[arg] = False

