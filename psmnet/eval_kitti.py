from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import cv2
from utils import preprocess 
from models import *
import imageio
from PIL import Image
from path import Path
import logging
from tqdm import tqdm
from collections import namedtuple
import pykitti
from scipy.linalg import expm

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='../00000000/',
                    help='select model')
parser.add_argument('--loadmodel', default='pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--sigma2', type=float, default=1.443)
parser.add_argument('--gamma2', type=float, default=13.82)
parser.add_argument('--ell', type=float, default=1.098, help='parameter ell for Kt')
parser.add_argument('--ell2', type=float, default=0.01, help='parameter ell for KR')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def RotDist(Ri, Rj):
    return np.sqrt(max(0.0, np.trace(np.eye(3) - (Ri.T).dot(Rj))))

def covfun(D, ell2):
    return np.exp(-np.square(D) / 2 / ell2)

def covfun_Matern(D, gamma2, ell):
    return gamma2 * (1 + math.sqrt(3) * D / ell) * np.exp(-math.sqrt(3) * D / ell)

def rotx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), np.sin(theta)],
                     [0, -np.sin(theta), np.cos(theta)]
                    ])

def roty(theta):
    return np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [0, 1, 0],
                     [np.sin(theta), 0, np.cos(theta)]
                    ])

def rotz(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0]
                    ])
def gyroD(xg, t):
    RR = np.zeros((3, 3, len(t)))
    R = np.eye(3)
    j = 0
    RR[:, :, j] = R
    j = j + 1
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        w = xg[i]
        R = expm(np.array([[0, w[2], -w[1]],
                           [-w[2], 0, w[0]],
                           [w[1], -w[0], 0]
                           ]) * dt).dot(R)

        RR[:, :, j] = R
        j = j + 1

    DR = np.zeros((len(t), len(t)))
    i = 0
    for ki in range(len(t)):
        Ri = RR[:, :, i]
        j = 0
        for kj in range(len(t)):
            Rj = RR[:, :, j]
            DR[i, j] = RotDist( Ri,  Rj)
            j += 1
        i += 1

    return DR

def genKt(t, gamma2, ell):
    Dt = np.zeros((len(t),len(t)))
    for ki in range(len(t)):
        for kj in range(len(t)):
            Dt[ki, kj] = np.abs(t[ki]-t[kj])
    covDt = gamma2 * (1 + math.sqrt(3) * Dt / ell) * np.exp(-math.sqrt(3) * Dt / ell)
    return covDt

def genDt(t):
    Dt = np.zeros((len(t),len(t)))
    for ki in range(len(t)):
        for kj in range(len(t)):
            Dt[ki, kj] = np.abs(t[ki]-t[kj])
    return Dt

def warp_with_disp(img, disp):
    h, w = img.shape[:2]
    i_range = np.tile(np.mgrid[0:h].reshape(h, 1), (1, w))
    j_range = np.tile(np.mgrid[0:w].reshape(1, w), (h, 1))

    map_x = (j_range - disp).astype(np.float32)
    map_y = i_range.astype(np.float32)

    projected = cv2.remap(np.asarray(img), map_x, map_y, cv2.INTER_LINEAR)
    return projected


def array2tensor(img):
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img.copy())


def SSIM(x, y):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    # (input, kernel, stride, padding)
    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return SSIM.mean()


def PSNR(x, y):
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def syn_accuracy(dispL, imL, imR):
    # synthetic imL with imR and disp
    h, w = imL.shape[:2]
    hp, wp = dispL.shape
    if hp != h or wp != w:
        dispL = cv2.resize(dispL, (h, w))

    warped_L = warp_with_disp(imR, dispL)
    mask = warped_L != 0
    mask = mask[:,:,0]
    mask = np.expand_dims(mask, 2)

    ssim = SSIM(array2tensor(imL*mask).float(), array2tensor(warped_L*mask).float())
    psnr = PSNR(imL*mask, warped_L*mask)
    #ssim = SSIM(array2tensor(imL).float(), array2tensor(warped_L).float())
    #psnr = PSNR(imL, warped_L)

    return ssim, psnr


def accuracy(dispL, dispL_gt):
    _, hg, wg = dispL_gt.shape
    _, hp, wp = dispL.shape
    if hp!=hg or wg!=wp:
        dispL = cv2.resize(dispL, (hg, wg))

    mask_disp = dispL_gt > 0
    disp_diff = abs(dispL_gt - dispL)
    # epe
    epe = disp_diff[mask_disp].mean()
    # d1
    mask1 = disp_diff[mask_disp] <= 3
    mask2 = (disp_diff[mask_disp] / dispL_gt[mask_disp]) <= 0.05
    pixels_good = (mask1 + mask2) > 0
    d1 = 100 - 100.0 * pixels_good.sum() / mask_disp.sum()
    return d1, epe

def accuracy2(dispL, dispL_gt):
    hg, wg = dispL_gt.shape
    hp, wp = dispL.shape
    if hp!=hg or wg!=wp:
        dispL = cv2.resize(dispL, (hg, wg))

    mask_disp = dispL_gt > 0
    disp_diff = abs(dispL_gt - dispL)
    # epe
    epe = disp_diff[mask_disp].mean()
    # d1
    mask1 = disp_diff[mask_disp] <= 3
    mask2 = (disp_diff[mask_disp] / dispL_gt[mask_disp]) <= 0.05
    pixels_good = (mask1 + mask2) > 0
    d1 = 100 - 100.0 * pixels_good.sum() / mask_disp.sum()
    return d1, epe

OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')


model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR, latent=None, pre=None, returnFlag=False):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        if latent is None and returnFlag:
            with torch.no_grad():
                latent, pre = model(imgL,imgR, None, None, returnFlag)
                return latent.cpu(), pre.cpu()
        else:
            with torch.no_grad():
                output = model(imgL,imgR, latent, pre, False)
                output = torch.squeeze(output)
                pred_disp = output.data.cpu().numpy()
                return pred_disp



def main():
    processed = preprocess.get_transform(augment=False)
    logging.basicConfig(filename="psmnet_kitti_Kcombined.log", level=logging.INFO)

    scene_name_list = os.listdir('/data/data_depth_annotated/val/')
    logging.info('use model %s' % args.loadmodel)
    logging.info('use sigma2 %f, gamma2 %f, ell %f, ell2 %f' % (args.sigma2, args.gamma2, args.ell, args.ell2))
    scene_gts_dict = {}
    for scene_name in scene_name_list:
        gts = sorted((Path('/data/data_depth_annotated/val') / scene_name / 'proj_depth' / 'groundtruth' / 'image_02/').files('*png'))
        if scene_name == '2011_09_26_drive_0036_sync':
           scene_gts_dict[scene_name] = [gts[:396], gts[396:]]
        elif scene_name == '2011_10_03_drive_0047_sync':
           scene_gts_dict[scene_name] = [gts[:413], gts[413:]]
        else:
           scene_gts_dict[scene_name] = [gts]

    all_gts = []
    all_preds = []

    all_ssim = []
    all_psnr = []

    for scene_name in scene_gts_dict.keys():
        for depth_gts in scene_gts_dict[scene_name]:

            year, month, day, _, drive, _ = scene_name.split('_')
            date = '_'.join([year, month, day])

            kitti_root = '/data/KITTI_raw'

            kitti_data = pykitti.raw(kitti_root, date, drive)

            left_images = []
            right_images = []
            oxts = []
            gyro_datas = []
            timestamps = []

            disps_gt = []

            raw_times = [d.timestamp() for d in kitti_data.timestamps]

            for gt in depth_gts:
                index = gt.split('/')[-1]  # %010.png
                left_images.append(Path(kitti_root) / date / scene_name / 'image_02' / 'data' / index)
                right_images.append(Path(kitti_root) / date / scene_name / 'image_03' / 'data' / index)

                oxt_fn = Path(kitti_root) / date / scene_name / 'oxts' / 'data' / index[:-4] + '.txt'
                oxts.append(oxt_fn)
                with open(oxt_fn, 'r') as f:
                    for line in f.read().splitlines():
                        line = line.split()
                        line[:-5] = [float(x) for x in line[:-5]]
                        line[-5:] = [int(float(x)) for x in line[-5:]]
                        packet = OxtsPacket(*line)
                        gyro_datas.append([packet.wx, packet.wy, packet.wz])
                        timestamps.append(raw_times[int(index[:-4])])


                baseline = kitti_data.calib.b_rgb
                focal_len = kitti_data.calib.K_cam2[0][0]
                depth = imageio.imread(gt).astype('float32')
                disparity = focal_len * baseline / depth * 256
                disparity[depth == 0] = 0
                disps_gt.append(disparity)
                all_gts.append(disparity)

            DR = gyroD(gyro_datas, timestamps)
            # if args.markov:
            logging.info('using markov now')
            d = np.diag(DR, -1)
            x = [0] + list(np.cumsum(d))
            Dm = np.abs(np.broadcast_to(x, (len(x), len(x))) - np.broadcast_to(x, (len(x), len(x))).transpose())

            #Kr = covfun(Dm, args.ell2)
            Kr = covfun_Matern(Dm, args.gamma2, args.ell)
            logging.info('using matern kgyro')
            Kt = genKt(timestamps, args.gamma2, args.ell)
            K = Kr * Kt
             
            assert len(left_images) == len(right_images)

            n = len(left_images)

            output = Path('preds')/scene_name/(args.loadmodel.split('.')[0])
            if not os.path.exists(output):
                   os.makedirs(output)

            latents = [] #saved for GP
            pres = [] #saved for GP

            preds = []
            with torch.no_grad():
                for i in tqdm(range(n)):
                    imgL_o = np.array((Image.open(left_images[i]).convert('RGB')))
                    imgR_o = np.array((Image.open(right_images[i]).convert('RGB')))


                    h_pad = 384 - imgL_o.shape[0]
                    w_pad = 1280 - imgL_o.shape[1]
                    top_pad = int(h_pad / 2)
                    left_pad = int(w_pad / 2)
                    bottom_pad = h_pad - top_pad
                    right_pad = w_pad - left_pad

                    imgL_o = np.lib.pad(imgL_o, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant',
                                       constant_values=0)
                    imgR_o = np.lib.pad(imgR_o, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant',
                                       constant_values=0)

                    imgL = processed(imgL_o).numpy()
                    imgR = processed(imgR_o).numpy()
                    imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
                    imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

                    latent, pre = test(imgL, imgR, returnFlag=True)

                    latents.append(latent)
                    pres.append(pre)

                Y = torch.stack(latents, dim=1).cpu()

                b, l, d, c, h, w = Y.size()
                Y = Y.view(b, l, -1).float()
                I = torch.eye(l).expand(b, l, l).float()
                K = torch.from_numpy(K).unsqueeze(0).float()
                X, _ = torch.solve(Y, K + args.sigma2 * I)
                Z = K.bmm(X)

                for i in tqdm(range(n)):
                    latent_after = Z[:,i].view(b,d, c,h,w).cuda()

                    imgL_o = np.array((Image.open(left_images[i]).convert('RGB')))
                    imgR_o = np.array((Image.open(right_images[i]).convert('RGB')))


                    h_pad = 384 - imgL_o.shape[0]
                    w_pad = 1280 - imgL_o.shape[1]
                    top_pad = int(h_pad / 2)
                    left_pad = int(w_pad / 2)
                    bottom_pad = h_pad - top_pad
                    right_pad = w_pad - left_pad

                    imgL_o = np.lib.pad(imgL_o, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant',
                                       constant_values=0)
                    imgR_o = np.lib.pad(imgR_o, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant',
                                       constant_values=0)

                    imgL = processed(imgL_o).numpy()
                    imgR = processed(imgR_o).numpy()
                    imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
                    imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

                    pred_disp = test(imgL, imgR, latent_after, pres[i].cuda(), returnFlag=False)
                    
                    if right_pad != 0:
                        disp_crop = pred_disp[top_pad:-bottom_pad, left_pad:-right_pad]
                    else:
                        disp_crop = pred_disp[top_pad:-bottom_pad, left_pad:]

                    preds.append(disp_crop)
                    all_preds.append(disp_crop)

                    imR = cv2.imread(right_images[i])
                    imL = cv2.imread(left_images[i])

                    ssim, psnr = syn_accuracy(disp_crop, imL, imR)
                    #warped_L = warp_with_disp(imR, disp_crop)
                    #cv2.imwrite('test_seq/%04d.png' % (i), warped_L)
                    #cv2.imwrite('test_seq/%04d-L.png' % (i), imL)
                    all_ssim.append(ssim)
                    all_psnr.append(psnr)

                    immy = np.squeeze(disp_crop) / 100 * 255  # modify with normalization for visualize
                    cv2.imwrite(output / '%04d.png' % (i), immy)
                    #np.save(output / '%04d.npy' % (i), disp_crop)

            d1, epe = accuracy(np.array(preds), np.array(disps_gt))
            print('scene:%s length: %d  D1: %04f EPE: %04f' % (scene_name, n, d1, epe))
            logging.info('scene:%s length: %d  D1: %04f EPE: %04f' % (scene_name, n, d1, epe))

    all_d1 = []
    all_epe = []

    for i in range(len(all_preds)):
        d1, epe = accuracy2(all_preds[i], all_gts[i])
        all_d1.append(d1)
        all_epe.append(epe)

    print('All D1: %04f, All EPE: %04f' % (np.mean(all_d1), np.mean(all_epe)))
    logging.info('All D1: %04f, All EPE: %04f' % (np.mean(all_d1), np.mean(all_epe)))
    print('All SSIM: %04f, All PSNR: %04f' % (np.mean(all_ssim), np.mean(all_psnr))) 
    logging.info('All SSIM: %04f, All PSNR: %04f' % (np.mean(all_ssim), np.mean(all_psnr)))  

if __name__ == '__main__':
   main()






