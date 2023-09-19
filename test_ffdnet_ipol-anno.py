
import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import FFDNet
from networks.network_dncnn import DnCNN
from utils import batch_psnr, normalize, init_logger_ipol, \
    variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
import imutils
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from skimage import filters, img_as_ubyte
from thop import profile  # 测试FLOPs

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_ffdnet(**args):
    r"""Denoises an input image with UMDNet
    """
    # Init logger

    logger = init_logger_ipol()


    try:
        rgb_den = is_rgb(args['input'])
    except:
        raise Exception('Could not open the input image')

    # Open image as a CxHxW torch.Tensor
    # CxHxW  input torch.Tensor
    if rgb_den:
        in_ch = 3
        # model_fn = 'models/dncnn_gray.pth'
        model_fn = 'logs/ffdnet_rgb.pth'
        imorig = cv2.imread(args['input'])

        grayA = cv2.cvtColor(imorig, cv2.COLOR_BGR2GRAY)  # ssim

        imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0,1)

    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        in_ch = 1
        if args['model_name'] == "DnCNN":
            model_fn = 'logs/dncnn_gray.pth'
            print("----load--dncnn_gray.pth")
        elif args['model_name'] == "FFDNet":
            model_fn = 'logs/ffdnet_gray.pth'
            print("----load--ffdnet_gray.pth")
        imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
        grayA = imorig
        imorig = np.expand_dims(imorig, 0)  # 给二维数据增加一个维度
    imorig = np.expand_dims(imorig, 0)  # 给三维增加y


    expanded_h = False
    expanded_w = False
    sh_im = imorig.shape
    if sh_im[2] % 2 == 1:  # thrid dims is odd
        expanded_h = True
        imorig = np.concatenate((imorig, \
                                 imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

    if sh_im[3] % 2 == 1:   # Fourth dims is odd
        expanded_w = True

        imorig = np.concatenate((imorig, \
                                 imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)


    imorig = normalize(imorig)
    imorig = torch.Tensor(imorig)


    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                            model_fn)

    # Create model

    print('Loading model ...\n')
    if args['model_name'] == "DnCNN":
        net = DnCNN(num_input_channels=in_ch)
        print("----load--dncnn_model")
    elif args['model_name'] == "FFDNet":
        net = FFDNet(num_input_channels=in_ch)
        print("----load--ffdnet_model")

    # Load saved weights
    # load GPU
    if args['cuda']:
        state_dict = torch.load(model_fn)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()

    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        # CPU mode: remove the DataParallel wrapper
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)  # load dict


    model.eval()

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    print("p2 before add_noise")


    if args['add_noise']:
        noise = torch.FloatTensor(imorig.size()). \
            normal_(mean=0, std=args['noise_sigma'])


        imnoisy = imorig + noise
    else:
        imnoisy = imorig.clone()

    # Test mode
    # dtype = torch.cuda.FloatTensor or torch.FloatTensor
    with torch.no_grad():  # PyTorch v0.4.0
        imorig, imnoisy = Variable(imorig.type(dtype)), \
            Variable(imnoisy.type(dtype))
        nsigma = Variable(
            torch.FloatTensor([args['noise_sigma']]).type(dtype))
        print("----nsigma----",nsigma)

    start_t = time.time()
    # Estimate noise and subtract it to the input image
    im_noise_estim = model(imnoisy, nsigma)
    outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)
    stop_t = time.time()
    print("denoisetime	: {}".format(stop_t - start_t))
    # -------Measure runtime & model-output--------

    # -----------FLOPs-------
    net = model
    # inputs = torch.randn(1, 3, 224, 224)
    inputs = imorig
    # print("inputs------------:",inputs.shape)
    flops, params = profile(net, inputs=(inputs, nsigma))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
    # -----------FLOPs-------


    if expanded_h:
        imorig = imorig[:, :, :-1, :]
        outim = outim[:, :, :-1, :]
        imnoisy = imnoisy[:, :, :-1, :]

        im_noise_estim = im_noise_estim[:, :, :-1, :]

    if expanded_w:
        imorig = imorig[:, :, :, :-1]
        outim = outim[:, :, :, :-1]
        imnoisy = imnoisy[:, :, :, :-1]

        im_noise_estim = im_noise_estim[:, :, :, :-1]

    # Compute PSNR and log it

    if rgb_den:
        logger.info("### RGB denoising ###")
    else:
        logger.info("### Grayscale denoising ###")
    if args['add_noise']:
        psnr = batch_psnr(outim, imorig, 1.)
        psnr_noisy = batch_psnr(imnoisy, imorig, 1.)

        logger.info("\tPSNR noisy {0:0.2f}dB".format(psnr_noisy))
        logger.info("\tPSNR denoised {0:0.2f}dB".format(psnr))
    else:
        psnr = batch_psnr(outim, imnoisy, 1.)
        logger.info("\tPSNR denoised {0:0.2f}dB".format(psnr))

        logger.info("\tNo noise was added, cannot compute PSNR")
    logger.info("\tRuntime {0:0.4f}s".format(stop_t - start_t))
    print("Compute PSNR and log it")

    # Compute difference
    im_noise_estim = 2 * im_noise_estim + .5
    diffout = 2 * (outim - imorig) + .5
    diffnoise = 2 * (imnoisy - imorig) + .5

    # Save images
    if not args['dont_save_results']:
        noisyimg = variable_to_cv2_image(imnoisy)
        outimg = variable_to_cv2_image(outim)
        im_noise_estim = variable_to_cv2_image(im_noise_estim)
        cv2.imwrite("estim_noise.png", im_noise_estim)

        print("shuchuweidu:{}".format(outimg.ndim))


        if outimg.ndim == 3:
            grayB = cv2.cvtColor(outimg, cv2.COLOR_BGR2GRAY)
        else:
            grayB = outimg
        (score, diff) = structural_similarity(grayA, grayB, win_size=101, full=True)
        # diff = (diff * 255).astype("uint8")
        # cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
        # cv2.imshow("diff", diff)
        print("SSIM:{}".format(score))
        print("save")
        cv2.imwrite("noisy.png", noisyimg)
        cv2.imwrite("ffdnet.png", outimg)
        if args['add_noise']:
            cv2.imwrite("noisy_diff.png",
                        variable_to_cv2_image(diffnoise))
            cv2.imwrite("ffdnet_diff.png", variable_to_cv2_image(diffout))
    # show originalimg   noiseimg(if add noise)  denoiseimg
    if args['add_noise']:
        imgyuantu = cv2.imread(args['input'])
        addnoisetu = cv2.imread("noisy.png")
        quzaohoutu = cv2.imread("ffdnet.png")
        plt.figure('denoiseimg', figsize=(16, 14))
        plt.subplot(131), plt.imshow(imgyuantu[:, :, ::-1]), plt.title('cleanimg')
        plt.subplot(132), plt.imshow(addnoisetu[:, :, ::-1]), plt.title('noiseimg')
        plt.subplot(133), plt.imshow(quzaohoutu[:, :, ::-1]), plt.title('denoiseimg')
        plt.show()
    else:
        imgyuantu = cv2.imread(args['input'])
        quzaohoutu = cv2.imread("ffdnet.png")
        plt.figure('denoiseimg', figsize=(16, 14))
        plt.subplot(121), plt.imshow(imgyuantu[:, :, ::-1]), plt.title('noiseimg')
        plt.subplot(122), plt.imshow(quzaohoutu[:, :, ::-1]), plt.title('denoiseimg')
        plt.show()


if __name__ == "__main__":
    # Parse arguments

    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument("--model_name", type=str, default="FFDNet", \
                        help='choose network')
    parser.add_argument('--add_noise', type=str, default="True")
    parser.add_argument("--input", type=str, default="", \
                        help='path to input image')
    parser.add_argument("--suffix", type=str, default="", \
                        help='suffix to add to output name')
    parser.add_argument("--noise_sigma", type=float, default=25, \
                        help='noise level used on test set')
    parser.add_argument("--dont_save_results", action='store_true', \
                        help="don't save output images")
    parser.add_argument("--no_gpu", action='store_true', \
                        help="run model on CPU")
    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]

    argspar.noise_sigma /= 255.

    # String to bool
    argspar.add_noise = (argspar.add_noise.lower() == 'true')


    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()


    print("\n### Testing  model ###")
    print("> Parameters:")

    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_ffdnet(**vars(argspar))
    print("test-end")
