import DRN.utility as utility
import DRN.data as data
import DRN.model as model
import DRN.loss as loss
from DRN.option import args
from DRN.checkpoint import Checkpoint
from DRN.trainer import Trainer
import numpy as np

# args.on_server = True
# args.cpu = True
args.data_dir = './'
args.save = './DRN/experiments'
args.data_test = 'Set5'
args.scale = 4
args.pre_train = './DRN/pretrained_models/DRNS{}x.pt'.format(str(args.scale)) #8x max is 256, 4x max is 512
args.test_only = True
args.save_results = True

utility.init_model(args)

# scale = [2,4] for 4x SR to load data
# scale = [2,4,8] for 8x SR to load data
args.scale = [pow(2, s+1) for s in range(int(np.log2(args.scale)))]

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

utility.set_seed(args.seed)
checkpoint = Checkpoint(args)


loader = data.Data(args)
model = model.Model(args, checkpoint)
loss = loss.Loss(args, checkpoint) if not args.test_only else None
t = Trainer(args, loader, model, loss, checkpoint)

def super_resolution(mediaId, image_type):
    error_info = t.server_test_split_time_data(mediaId, image_type)
    return error_info
    # if error_info is not None:
    #     print(error_info)
    # else:
    #     print('is None')





