import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from collections import OrderedDict
###from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from models import FFDNet
# from networks.network_ffdnet import FFDNet
from networks.network_usrnet import USRNet
# from networks.network_discriminator import Discriminator_PatchGAN,Discriminator_UNet,\
#	 Discriminator_VGG_96,Discriminator_VGG_128,Discriminator_VGG_192,Discriminator_VGG_128_SN

from networks.network_discriminator import Discriminator_VGG_128_SN
from networks.network_dncnn import DnCNN
from networks.network_dpsr import SRResNet
from networks.network_faceenhancer import FullGenerator
from networks.network_feature import VGGFeatureExtractor
from networks.network_imdn import IMDN
from networks.network_msrresnet import MSRResNet1
from networks.network_rrdb import RRDB
from networks.network_rrdbnet import RRDBNet
from networks.network_srmd import SRMD
from networks.network_swinir import SwinIR
from networks.network_unet import UNetRes
from networks.network_vrt import VRT
import json

# from network_usrnet import USRNet
from dataset import Dataset  # 从dataset.py获取Dataset数据集
from utils import weights_init_kaiming, batch_psnr, batch_ssim, init_logger, \
    svd_orthogonalization
# import pytorch_ssim
import random
import torch.utils.data as data
import utilss.utils_image as util
from utilss import utils_deblur
from utilss import utils_sisr
import os
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.io import loadmat

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def savetrainimg(j, data):
    # every batch get 5 images,save to hardware
    for i in range(5):
        # img = cv2.imread(data[i, :, :, :] )*255
        # saveimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        saveimg = data[i, :, :,:] * 255
        saveimg = saveimg.numpy()  # (1, 40, 40)

        # saveimg = saveimg[0]

        saveimg = saveimg.astype('uint8')  # （40，40）       [[148 146 154 ......]]    uint8      #<class 'numpy.ndarray'>
        # print("---------saveimg------", saveimg.shape,saveimg,saveimg.dtype,type(saveimg))
        saveimg = np.transpose(saveimg, (1, 2, 0))  # (40,20,1)

        dirs = 'saveimg/{}epoch/savetrainimg'.format(j + 1)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        # cv2.imwrite('dirs/mypic{}.jpg'.format(i+1), saveimg)
        cv2.imwrite(os.path.join(dirs, 'trainimg{}.jpg'.format(i + 1)), saveimg)


def saveimgn(j, data):
    # every batch get 5 images,add noise, save to hardware
    for i in range(5):
        saveimg = data[i, :, :,
                  :] * 255  # torch.Size([1, 40, 40])     //////////   tensor([[[132., 129., 135., ........
        saveimg = saveimg.numpy()
                #print("-----saveimgn-------",saveimg)       #[[[209.05566  135.8353    16.266031 ...
        saveimg = saveimg.astype('uint8')  # （40，40）       [[148 146 154 ......]]    uint8      #<class 'numpy.ndarray'>
        # print("---------saveimg------", saveimg.shape,saveimg,saveimg.dtype,type(saveimg))
        saveimg = np.transpose(saveimg, (1, 2, 0))  # (40,20,1)

        dirs = 'saveimg/{}epoch/imgnoise'.format(j + 1)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        # cv2.imwrite('./saveimgn/mypic{}.jpg'.format(i), saveimg)
        # cv2.imwrite('dirs/imgno{}.jpg'.format(i), saveimg)
        cv2.imwrite(os.path.join(dirs, 'imgno{}.jpg'.format(i + 1)), saveimg)


def saveoutimg(j, data):
    # every batch get 5 denoised images,save to hardware
    for i in range(5):

        saveimg = data[i, :, :,
                  :] * 255  # torch.Size([1, 40, 40])     //////////   tensor([[[132., 129., 135., ........
        saveimg = saveimg.detach().numpy()

        saveimg = saveimg.astype('uint8')  # （40，40）       [[148 146 154 ......]]    uint8      #<class 'numpy.ndarray'>
        # print("---------saveimg------", saveimg.shape,saveimg,saveimg.dtype,type(saveimg))
        saveimg = np.transpose(saveimg, (1, 2, 0))  # (40,20,1)

        dirs = 'saveimg/{}epoch/imgout'.format(j + 1)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        # cv2.imwrite('dirs/mypic{}.jpg'.format(i), saveimg)
        cv2.imwrite(os.path.join(dirs, 'outimg{}.jpg'.format(i + 1)), saveimg)


def main(args):
    r"""Performs the main training loop
    """
    # Load dataset
    print('> Loading dataset ...')  # 从dataset.py获取Dataset数据集
    dataset_train = Dataset(train=True, gray_mode=args.gray, shuffle=True)
    dataset_val = Dataset(train=False, gray_mode=args.gray, shuffle=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    print("\t# of training samples: %d\n" % int(len(dataset_train)))

    # Init loggers
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)    #logs save to /logs,name events.out.tfevents.167.......
    logger = init_logger(args)
    # Create model
    if not args.gray:
        in_ch = 3
    else:
        in_ch = 1
    # 选择model
    if args.model_name == "FFDNet":
        net = FFDNet(num_input_channels=in_ch)
        print("IN FFDNet")
    elif args.model_name == "USRNet":
        print("IN USRNet")
        # f = open('options/train_usrnet.json', "r")  # JSON file
        # opt = json.loads(f.read())  # Reading from file


        # ----------------------------------------
        json_str = ''
        with open('options/train_usrnet.json', 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line

        # ----------------------------------------
        # initialize opt
        # ----------------------------------------
        opt = json.loads(json_str, object_pairs_hook=OrderedDict)

        opt_net = opt['netG']
        batchsize = opt['datasets']
        batchsize2 = batchsize['train']
        batchsizes = batchsize2['dataloader_batch_size']
        # scales = opt['scales'] if opt['scales'] is not None else [1, 2, 3, 4]
        scales = [1, 2, 3, 4]
        net = USRNet(n_iter=opt_net['n_iter'],
                     h_nc=opt_net['h_nc'],
                     in_nc=opt_net['in_nc'],
                     out_nc=opt_net['out_nc'],
                     nc=opt_net['nc'],
                     nb=opt_net['nb'],
                     act_mode=opt_net['act_mode'],
                     downsample_mode=opt_net['downsample_mode'],
                     upsample_mode=opt_net['upsample_mode']
                     )
    elif args.model_name == "Discriminator_VGG_128_SN":
        net = Discriminator_VGG_128_SN(num_input_channels=in_ch)
    elif args.model_name == "DnCNN":
        print("IN DnCNN")
        net = DnCNN(num_input_channels=in_ch)  # from-models

    elif args.model_name == "SRResNet":
        net = SRResNet(num_input_channels=in_ch)
    elif args.model_name == "FullGenerator":
        net = FullGenerator(num_input_channels=in_ch)
    elif args.model_name == "VGGFeatureExtractor":
        net = VGGFeatureExtractor(num_input_channels=in_ch)
    elif args.model_name == "IMDN":
        net = IMDN(num_input_channels=in_ch)
    elif args.model_name == "MSRResNet1":
        net = MSRResNet1(num_input_channels=in_ch)
    elif args.model_name == "RRDB":
        net = RRDB(num_input_channels=in_ch)
    elif args.model_name == "RRDBNet":
        net = RRDBNet(num_input_channels=in_ch)
    elif args.model_name == "SRMD":
        net = SRMD(num_input_channels=in_ch)
    elif args.model_name == "SwinIR":
        net = SwinIR(num_input_channels=in_ch)
    elif args.model_name == "UNetRes":
        net = UNetRes(num_input_channels=in_ch)
    elif args.model_name == "VRT":
        net = VRT(num_input_channels=in_ch)

    net.apply(weights_init_kaiming)
    # Define loss
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()  # loss

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  #optimialize

    # Resume training or start a new.
    if args.resume_training:
        resumef = os.path.join(args.log_dir, 'ffdnet_grayckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)  #  reload model
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = args.epochs
            new_milestone = args.milestone
            current_lr = args.lr
            args = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            args.epochs = new_epoch
            args.milestone = new_milestone
            args.lr = current_lr
            print("=> loaded checkpoint '{}' (epoch {})" \
                  .format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = vars(checkpoint['args'])
            print("==> checkpoint['args']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))

            args.resume_training = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}". \
                            format(resumef))
    else:
        start_epoch = 0
        training_params = {}  #  define dict
        training_params['step'] = 0
        training_params['current_lr'] = 0
        training_params['no_orthog'] = args.no_orthog

    # Training
    for epoch in range(start_epoch, args.epochs):

        if epoch > args.milestone[1]:
            current_lr = args.lr / 1000.
            training_params['no_orthog'] = True
        elif epoch > args.milestone[0]:
            current_lr = args.lr / 10.
        else:
            current_lr = args.lr
        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train,0):
            if i == len(loader_train)//10:
                break
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data


            # saveimage
            if i == 0:
                savetrainimg(epoch, data)

            # inputs: noise and noisy image
            noise = torch.zeros(img_train.size())

            stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1], \
                                     size=noise.size()[0])
            for nx in range(noise.size()[0]):
                sizen = noise[0, :, :, :].size()

                noise[nx, :, :, :] = torch.FloatTensor(sizen). \
                    normal_(mean=0, std=stdn[nx])

            imgn_train = img_train + noise


            if i == 0:
                saveimgn(epoch, imgn_train)

            # Create input Variables

            img_train = Variable(img_train.cuda())

            imgn_train = Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            stdn_var = Variable(torch.cuda.FloatTensor(stdn))  # torch.Size([128])


            if args.model_name == "DnCNN":
                out_train = model(imgn_train, stdn_var)
            # out_train = model(imgn_train)
            elif args.model_name == "USRNet":

                #  ------USRNet kernel ------------------------------
                y = np.zeros([25, 25], dtype=float)
                r_value = random.randint(0, 7)
                for i in range(batchsizes - 1):
                    if r_value > 3:
                        k = utils_deblur.blurkernel_synthesis(h=25)  # motion blur
                    else:
                        sf_k = random.choice(scales)
                        k = utils_sisr.gen_kernel(scale_factor=np.array([sf_k, sf_k]))  # Gaussian blur
                        mode_k = random.randint(0, 7)
                        k = util.augment_img(k, mode=mode_k)

                    y = np.hstack([y, k])
                print("yyyyyyyyyyyyyyy----shape--Y", y.shape)  # shape....(25,1200)

                k = util.single2tensor3(np.expand_dims(np.float32(y), axis=2))  # to tensor
                print("XXXXXXXXXX1111----k.shape:", k.shape)  # torch.Size([1, 25, 1200])

                k = k.reshape(batchsizes, 1, 25, 25)  # to dim=4
                print("XXXXXXXXXX1111----k.reshape:", k.shape)  # k.reshape: torch.Size([48, 1, 25, 25])
                # print("XXXXXXXXXX1111---hou-k:", k)
                # ----------------kernel---end

                out_train = model(imgn_train, k, 3, stdn_var)
            elif args.model_name == "FFDNet":
                out_train = model(imgn_train, stdn_var)

            #print("XXXXXXXXout_train_shape", out_train.shape)
            # print("XXXXXXXXnoise_shape", noise.shape)

            loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
            loss.backward()
            optimizer.step()

            # Results
            model.eval()
            out_train = torch.clamp(imgn_train - model(imgn_train, stdn_var), 0., 1.)
            # out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
            elif args.model_name == "USRNet":
                out_train = torch.clamp(imgn_train - model(imgn_train, k, 3, stdn_var), 0., 1.)
            # print("shape:",out_train.size())
            # print("shape1:", img_train.size())
            elif args.model_name == "FFDNet":
                out_train = torch.clamp(imgn_train - model(imgn_train, stdn_var), 0., 1.)

            # saveimage2
            # print("-----out_train-----",out_train.shape)
            if i == 0:
                out_trainsave = out_train.cpu()
                saveoutimg(epoch, out_trainsave)  # torch.Size([128, 1, 40, 40])

            psnr_train = batch_psnr(out_train, img_train, 1.)

            ssim_train = batch_ssim(out_train, img_train, args.batch_size)

            # PyTorch v0.4.0: loss.data[0] --> loss.item()

            if training_params['step'] % args.save_every == 0:
                # Apply regularization by orthogonalizing filters
                if not training_params['no_orthog']:
                    model.apply(
                        svd_orthogonalization)
                writer.add_scalar('loss', loss.item(), training_params['step'])
                writer.add_scalar('PSNR on training data', psnr_train, training_params['step'])
                writer.add_scalar('SSIM on training data', ssim_train, training_params['step'])
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f ssim_train: %.4f" % \
                      (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train, ssim_train))
            training_params['step'] += 1
        # The end of each epoch
        model.eval()

        # Validation
        psnr_val = 0
        for valimg in dataset_val:

            img_val = torch.unsqueeze(valimg, 0)

            noise = torch.FloatTensor(img_val.size()). \
                normal_(mean=0, std=args.val_noiseL)
            imgn_val = img_val + noise

            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            sigma_noise = Variable(torch.cuda.FloatTensor([args.val_noiseL]))

            if args.model_name == "DnCNN":
                out_val = torch.clamp(imgn_val - model(imgn_val, sigma_noise), 0., 1.)
            # out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
            elif args.model_name == "USRNet":
                out_val = torch.clamp(imgn_val - model(imgn_val, k, 3, stdn_var), 0., 1.)
            elif args.model_name == "FFDNet":
                out_val = torch.clamp(imgn_val - model(imgn_val, sigma_noise), 0., 1.)
            psnr_val += batch_psnr(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        ssim_val = batch_ssim(out_val, img_val, epoch + 1)
        print("\n[epoch %d] SSIM_val: %.4f" % (epoch + 1, ssim_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('Learning rate', current_lr, epoch)
        writer.add_scalar('SSIM on validation data', ssim_val, epoch)

        # Log val images
        try:
            if epoch == 0:
                # Log graph of the model
                writer.add_graph(model, (imgn_val, sigma_noise), )
                # Log validation images
                for idx in range(2):
                    imclean = utils.make_grid(img_val.data[idx].clamp(0., 1.), \
                                              nrow=2, normalize=False, scale_each=False)
                    imnsy = utils.make_grid(imgn_val.data[idx].clamp(0., 1.), \
                                            nrow=2, normalize=False, scale_each=False)
                    # print("-----imclean------",imclean)     #tensor([[[0.4980, 0.4824, 0.4902,...
                    writer.add_image('Clean validation image {}'.format(idx), imclean, epoch)
                    writer.add_image('Noisy validation image {}'.format(idx), imnsy, epoch)
            for idx in range(2):
                imrecons = utils.make_grid(out_val.data[idx].clamp(0., 1.), \
                                           nrow=2, normalize=False, scale_each=False)
                writer.add_image('Reconstructed validation image {}'.format(idx), \
                                 imrecons, epoch)
            # Log training images
            imclean = utils.make_grid(img_train.data, nrow=8, normalize=True, \
                                      scale_each=True)
            writer.add_image('Training patches', imclean, epoch)


        except Exception as e:
            logger.error("Couldn't log results: {}".format(e))
        writer.close()

        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'ffdnet_gray.pth'))  # 保存权重值
        save_dict = { \
            'state_dict': model.state_dict(), \
            'optimizer': optimizer.state_dict(), \
            'training_params': training_params, \
            'args': args \
            }
        torch.save(save_dict,
                   os.path.join(args.log_dir, 'ffdnet_grayckpt.pth'))  # 保存state_dict,optimizer,training_params,args的值
        if epoch % args.save_every_epochs == 0:
            torch.save(save_dict, os.path.join(args.log_dir, \
                                               'ffdnet_graysckpt_e{}.pth'.format(epoch + 1)))
        del save_dict


if __name__ == "__main__":

    ### parser = argparse.ArgumentParser(description="FFDNet")
    ### parser = argparse.ArgumentParser(description="USRNet")
    parser = argparse.ArgumentParser(description="choosemodel")
    parser.add_argument("--model_name", type=str, default="FFDNet", \
                        help='choose network')


    parser.add_argument("--gray", action='store_true', \
                        help='train grayscale image denoising instead of RGB')
    parser.add_argument("--log_dir", type=str, default="logs", \
                        help='path of log files')
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, \
                        help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=80, \
                        help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true', \
                        help="resume training from a previous checkpoint")
    parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-3, \
                        help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true', \
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=10, \
                        help="Number of training steps to log psnr and perform \
						orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=5, \
                        help="Number of training epochs to save state")
    # nargs参数：表示该指令接收值的个数
    parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0, 75], \
                        help="Noise training interval")
    parser.add_argument("--val_noiseL", type=float, default=25, \
                        help='noise level used on validation set')
    argspar = parser.parse_args()
    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.noiseIntL[0] /= 255.
    argspar.noiseIntL[1] /= 255.

    ### print("\n### Training FFDNet model ###")
    print("\n### Training models ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(argspar)
