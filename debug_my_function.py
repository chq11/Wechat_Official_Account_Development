import DRN.utility as utility
import DRN.data as data
import DRN.model as model
import DRN.loss as loss
from DRN.option import args
from DRN.checkpoint import Checkpoint
from DRN.trainer import Trainer

# args.on_server = True
# args.cpu = True
args.data_dir = './'
args.save = './DRN/experiments'
args.data_test = 'Set5'
# args.scale = 4
args.pre_train = './DRN/pretrained_models/DRNS8x.pt'
args.test_only = True
args.save_results = True


utility.set_seed(args.seed)
checkpoint = Checkpoint(args)


loader = data.Data(args)
model = model.Model(args, checkpoint)
loss = loss.Loss(args, checkpoint) if not args.test_only else None
t = Trainer(args, loader, model, loss, checkpoint)
error_info = t.server_test('b')
if error_info is not None:
	print(error_info)
else:
	print('is None')



