import torch
import DRN.utility as utility
import cv2,imageio,os
import numpy as np
import math,time
from decimal import Decimal
from tqdm import tqdm


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_models = self.model.dual_models
        self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
            sr = self.model(lr[0])
            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

            # compute primary loss
            loss_primary = self.loss(sr[-1], hr)
            for i in range(1, len(sr)):
                loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

            # compute dual loss
            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])

            # compute total loss
            loss = loss_primary + self.opt.dual_weight * loss_dual
            
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()                
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            scale = max(self.scale)
            # print(scale)
            for si, s in enumerate([scale]):
                # print(si,s)
                eval_psnr = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    print(lr[0].shape)
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                        # print(lr, len(lr))
                    else:
                        lr, = self.prepare(lr)
                        # print(lr, lr.shape)

                    print(lr[0])

                    sr = self.model(lr[0])
                    if isinstance(sr, list): sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)

                    if not no_eval:
                        eval_psnr += utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)

                self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def server_test(self, mediaId, image_type):
        with torch.no_grad():
            # b, g, r = cv2.split(cv2.imread('/home/haiquan/deep_learning/DRN-master-function/benchmark/Set5/LR_bicubic/X4/babyx4.png'))
            # lr = cv2.merge([r,g,b])
            # lr = np.transpose(lr,[2,0,1])
            ipath = './DRN/received_image/{}.{}'.format(mediaId, image_type)
            fsize = os.path.getsize(ipath)
            f_kb = fsize / float(1024)
            f_mb = f_kb / float(1024)
            if f_mb < 1:
                lr = imageio.imread(ipath)
                lr = np.transpose(lr, [2, 0, 1])
                lr = lr[np.newaxis,:,:,:]
                lrshape = lr.shape
                if lrshape[2] <= 512 and lrshape[3] <= 512:
                    lr = torch.tensor(lr).float() #torch.Size([1, 3, 128, 128])
                    lr, = self.prepare(lr) #torch.Size([1, 3, 128, 128])
                    lr = lr[0].view([lrshape[0], lrshape[1], lrshape[2], lrshape[3]])
                    sr = self.model(lr)
                    if isinstance(sr, list): sr = sr[-1]
                    sr = utility.quantize(sr, self.opt.rgb_range)

                    normalized = sr[0].data.mul(255 / self.opt.rgb_range)
                    ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                    imageio.imwrite('./DRN/received_image/{}1.jpg'.format(mediaId), ndarr)
                else:
                    return 'image resolution should be less than 512*512!'
            else:
                return 'image size should be less than 2M!'

            # sr = sr.cpu()[0]
            # sr = np.transpose(sr, [1, 2, 0])
            # print(sr.shape)
            #
            # # save test results
            # imageio.imsave('./debug/aaa.png', sr)

    def server_test_split_data(self, mediaId, image_type):
        with torch.no_grad():
            ipath = './DRN/received_image/{}.{}'.format(mediaId, image_type)
            scale = self.scale[-1]
            max_shape = 256
            lr = imageio.imread(ipath)
            lr = np.transpose(lr, [2, 0, 1])  # [3, 256, 256]
            _, lr_h, lr_w = lr.shape
            h_number = math.ceil(lr_h / max_shape)
            w_number = math.ceil(lr_w / max_shape)
            s_h = int(lr_h / h_number)
            s_w = int(lr_w / w_number)
            split_array = np.zeros([1, h_number * w_number, 3, s_h, s_w])
            for i in range(h_number):
                for j in range(w_number):
                    split_array[0, i*w_number+j] = lr[:, i * s_h:(i + 1) * s_h, j * s_w:(j + 1) * s_w]

            split_array = torch.tensor(split_array).float() #torch.Size([1, 9, 3, 85, 85])
            split_array, = self.prepare(split_array) #torch.Size([1, 9, 3, 85, 85])
            if isinstance(split_array, list): split_array = split_array[-1] #torch.Size([9, 3, 85, 85])

            start_t = time.time()
            sr_array = self.model(split_array) #torch.Size([1, 9, 3, 680, 680])
            end_t = time.time()
            print('used time: {}s'.format(str(end_t-start_t)))

            if isinstance(sr_array, list): sr_array = sr_array[-1] #torch.Size([9, 3, 680, 680])
            sr_array = utility.quantize(sr_array, self.opt.rgb_range) #torch.Size([9, 3, 680, 680])

            normalized = sr_array.data.mul(255 / self.opt.rgb_range) #torch.Size([9, 3, 680, 680])
            ndarr = normalized.byte().permute(0, 2, 3, 1).cpu().numpy() #torch.Size([9, 680, 680, 3])

            sr = np.zeros([s_h*h_number*scale, s_w*w_number*scale, 3], dtype=np.uint8)
            for i in range(h_number):
                for j in range(w_number):
                    sr[i * s_h*scale:(i + 1) * s_h*scale, j * s_w*scale:(j + 1) * s_w*scale, :] = ndarr[i*w_number+j]

            imageio.imwrite('./DRN/received_image/{}1.jpg'.format(mediaId), sr)

    def server_test_split_time_data(self, mediaId, image_type):
        with torch.no_grad():
            ipath = './DRN/received_image/{}.{}'.format(mediaId, image_type)
            scale = self.scale[-1]
            unit_max_shape = 768 #4X: fit 768, best 832, try 896
            lr = imageio.imread(ipath)
            lr = np.transpose(lr, [2, 0, 1])  # [3, 960, 960]
            _, lr_h, lr_w = lr.shape
            unit_h_number = math.ceil(lr_h / unit_max_shape) #2
            unit_w_number = math.ceil(lr_w / unit_max_shape) #2
            unit_h = int(lr_h / unit_h_number) #480
            unit_w = int(lr_w / unit_w_number) #480

            per_max_shape = 768 #128
            per_h_number = math.ceil(unit_h / per_max_shape) #4
            per_w_number = math.ceil(unit_w / per_max_shape) #4
            per_h = int(unit_h / per_h_number) #120
            per_w = int(unit_w / per_w_number) #120

            sr = np.zeros([per_h * per_h_number * unit_h_number * scale, per_w * per_w_number * unit_w_number * scale, 3], dtype=np.uint8)
            for i in range(unit_h_number):
                for j in range(unit_w_number):
                    unit_lr = lr[:, i * unit_h:(i + 1) * unit_h, j * unit_w:(j + 1) * unit_w]
                    split_array = np.zeros([1, per_h_number * per_w_number, 3, per_h, per_w])
                    for i1 in range(per_h_number):
                        for j1 in range(per_w_number):
                            split_array[0, i1*per_w_number+j1] = unit_lr[:, i1 * per_h:(i1 + 1) * per_h, j1 * per_w:(j1 + 1) * per_w]

                    split_array = torch.tensor(split_array).float() #torch.Size([1, 16, 3, 120, 120])
                    split_array, = self.prepare(split_array) #torch.Size([1, 16, 3, 120, 120])
                    if isinstance(split_array, list): split_array = split_array[-1] #torch.Size([16, 3, 120, 120])

                    # start_t = time.time()
                    sr_array = self.model(split_array) #torch.Size([1, 16, 3, 960, 960])
                    # end_t = time.time()
                    # print('used time: {}s'.format(str(end_t-start_t)))

                    if isinstance(sr_array, list): sr_array = sr_array[-1] #torch.Size([16, 3, 960, 960])
                    sr_array = utility.quantize(sr_array, self.opt.rgb_range) #torch.Size([16, 3, 960, 960])

                    normalized = sr_array.data.mul(255 / self.opt.rgb_range) #torch.Size([16, 3, 960, 960])
                    ndarr = normalized.byte().permute(0, 2, 3, 1).cpu().numpy() #torch.Size([16, 960, 960, 3])

                    for ii in range(per_h_number):
                        for jj in range(per_w_number):
                            sr[(i*unit_h + ii * per_h)*scale:(i*unit_h + (ii + 1) * per_h)*scale,(j*unit_w + jj * per_w)*scale:(j*unit_w + (jj + 1) * per_w)*scale, :] = ndarr[ii*per_w_number+jj]

            imageio.imwrite('./DRN/received_image/{}1.jpg'.format(mediaId), sr)

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args)>1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]], 

    def terminate(self):
        if self.opt.test_only:
            if self.opt.on_server:
                self.server_test()
            else:
                self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
