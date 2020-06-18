import DRN.utility as utility
import data
import model
import DRN.loss as loss
from DRN.option import args
from DRN.checkpoint import Checkpoint
from DRN.trainer import Trainer

utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()

