from torchvision import transforms as tr
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
from skimage.measure import regionprops, label
from skimage.filters import threshold_li
from skimage.exposure import adjust_gamma
from skimage.color import rgb2hsv
import os, os.path as osp

def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Training for Biomedical Image Classification')
    parser.add_argument('--im_path_in', type=str, default='../../datasets/justRAIGS/0/', help='path to training images')
    parser.add_argument('--im_path_out', type=str, default='data/images', help='path to training images')
        # parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=True, help='avoid saving anything')

    args = parser.parse_args()

    return args

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC


def get_fov(binary):
    binary = getLargestCC(binary)
    regions = regionprops(binary.astype(np.uint8))
    areas = [r.area for r in regions]
    largest_cc_idx = np.argmax(areas)
    fov = regions[largest_cc_idx]

    return fov


def crop_resize_im(im, tg_size=(512, 512), allowed_eccentricity=0.75):
    im = np.array(im)
    rsz = tr.Resize(tg_size)
    center_crop = tr.CenterCrop(tg_size)

    sat = rgb2hsv(im)[:, :, 1]
    t = threshold_li(sat)
    binary = sat > t
    fov = get_fov(binary)
    status = 'success'
    if fov.eccentricity > allowed_eccentricity:
        t = threshold_li(adjust_gamma(sat, 0.5))
        binary = adjust_gamma(sat, 0.5) > t
        fov = get_fov(binary)

    if fov.eccentricity > allowed_eccentricity:
        status = 'failed'

    cropped = im[fov.bbox[0]:fov.bbox[2], fov.bbox[1]: fov.bbox[3], :]
    cropped = Image.fromarray(cropped)

    return rsz(cropped), status

def main(args):
    im_path_in = args.im_path_in
    im_path_out = args.im_path_out
    os.makedirs(im_path_out, exist_ok=True)
    im_list = os.listdir(im_path_in)
    im_list = [osp.join(im_path_in, n) for n in im_list if n.endswith('.JPG')]

    if not im_path_in.endswith('/'):
        im_path_in += '/'
    if not im_path_out.endswith('/'):
        im_path_out += '/'
    for idx in tqdm(range(len(im_list))):
        n = im_list[idx]
        n_out = n.replace(im_path_in, im_path_out)
        try:
            im = Image.open(n)
            im_out, status = crop_resize_im(im)
            if status == 'failed':
                with open('log.txt', 'a') as f:
                    print('Failed after exposure correction at {}'.format(n), file=f)
            im_out.save(n_out)
        except:
            with open('log.txt', 'a') as f:
                print('Exception at {}'.format(n), file=f)


    # Start preprocessing
    start = time.time()



    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('done')

    print('Time Spent: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))
if __name__ == "__main__":
    args = get_args_parser()
    main(args)
