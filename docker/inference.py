import random
import os
from tqdm import tqdm
from PIL import Image
from skimage.filters import threshold_li
from skimage.exposure import adjust_gamma
from skimage.color import rgb2hsv
from skimage.measure import regionprops, label
import json
import torch
from torchvision.models import (resnet50, resnext50_32x4d, swin_t, efficientnet_b0, efficientnet_b1,
                                efficientnet_b2, efficientnet_b3, convnext_tiny)
from torchvision.models.convnext import LayerNorm2d
from torchvision import transforms as tr
from pathlib import Path
import tempfile
import sys
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.dataset import Dataset

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


def get_resnet50(n_classes):
    model = resnet50()
    num_ftrs = model.fc.in_features
    linear_cl = torch.nn.Linear(num_ftrs, n_classes)
    model.fc = linear_cl
    return model

def get_resnext50(n_classes):
    model = resnext50_32x4d()
    num_ftrs = model.fc.in_features
    linear_cl = torch.nn.Linear(num_ftrs, n_classes)
    model.fc = linear_cl
    return model

def get_swin(n_classes):
    model = swin_t()
    num_ftrs = model.head.in_features
    linear_cl = torch.nn.Linear(num_ftrs, n_classes)
    model.head = linear_cl
    return model

def get_effb0(n_classes):
    model = efficientnet_b0()
    num_ftrs = model.classifier[-1].in_features
    linear_cl = torch.nn.Linear(num_ftrs, n_classes)
    model.classifier = linear_cl
    return model

def get_effb1(n_classes):
    model = efficientnet_b1()
    num_ftrs = model.classifier[-1].in_features
    linear_cl = torch.nn.Linear(num_ftrs, n_classes)
    model.classifier = linear_cl
    return model

def get_effb2(n_classes):
    model = efficientnet_b2()
    num_ftrs = model.classifier[-1].in_features
    linear_cl = torch.nn.Linear(num_ftrs, n_classes)
    model.classifier = linear_cl
    return model

def get_effb3(n_classes):
    model = efficientnet_b3()
    num_ftrs = model.classifier[-1].in_features
    linear_cl = torch.nn.Linear(num_ftrs, n_classes)
    model.classifier = linear_cl
    return model

def get_convnext(n_classes):
    model = convnext_tiny()
    num_ftrs = model.classifier[-1].in_features
    model.classifier = torch.nn.Sequential(LayerNorm2d([num_ftrs, ], eps=1e-06, elementwise_affine=True),
                                           torch.nn.Flatten(start_dim=1, end_dim=-1),
                                           torch.nn.Linear(in_features=768, out_features=n_classes))
    return model

def load_justify_model_list(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model_name in ['r50', 'rx50', 'swin', 'effb0', 'effb1', 'effb2', 'effb3', 'convnext']
    n = 7
    if model_name == 'r50':
        get_model = get_resnet50
        n = 5
    elif model_name == 'rx50':
        get_model = get_resnext50
        n = 5
    elif model_name == 'swin': get_model = get_swin
    elif model_name == 'effb0': get_model = get_effb0
    elif model_name == 'effb1': get_model = get_effb1
    elif model_name == 'effb2': get_model = get_effb2
    elif model_name == 'effb3': get_model = get_effb3
    elif model_name == 'convnext': get_model = get_convnext

    path_model_justify_state_f1 = "weights/justify/F1/{}_bce_soft_mad_{}x3/best_model.pth".format(model_name, n)
    path_model_justify_state_f2 = "weights/justify/F2/{}_bce_soft_mad_{}x3/best_model.pth".format(model_name, n)
    path_model_justify_state_f3 = "weights/justify/F3/{}_bce_soft_mad_{}x3/best_model.pth".format(model_name, n)
    path_model_justify_state_f4 = "weights/justify/F4/{}_bce_soft_mad_{}x3/best_model.pth".format(model_name, n)
    path_model_justify_state_f5 = "weights/justify/F5/{}_bce_soft_mad_{}x3/best_model.pth".format(model_name, n)

    model_justify_f1 = get_model(n_classes=10).to(device)
    model_justify_f1.load_state_dict(torch.load(path_model_justify_state_f1, map_location=device))
    model_justify_f1.eval()

    model_justify_f2 = get_model(n_classes=10).to(device)
    model_justify_f2.load_state_dict(torch.load(path_model_justify_state_f2, map_location=device))
    model_justify_f2.eval()

    model_justify_f3 = get_model(n_classes=10).to(device)
    model_justify_f3.load_state_dict(torch.load(path_model_justify_state_f3, map_location=device))
    model_justify_f3.eval()

    model_justify_f4 = get_model(n_classes=10).to(device)
    model_justify_f4.load_state_dict(torch.load(path_model_justify_state_f4, map_location=device))
    model_justify_f4.eval()

    model_justify_f5 = get_model(n_classes=10).to(device)
    model_justify_f5.load_state_dict(torch.load(path_model_justify_state_f5, map_location=device))
    model_justify_f5.eval()

    justify_model_list = [model_justify_f1, model_justify_f2, model_justify_f3, model_justify_f4, model_justify_f5]

    return justify_model_list
def load_rg_model_list(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model_name in ['r50', 'rx50', 'effb2', 'effb3']
    if model_name == 'r50': get_model = get_resnet50
    elif model_name == 'rx50': get_model = get_resnext50
    elif model_name == 'effb2': get_model = get_effb2
    elif model_name == 'effb3': get_model = get_effb3

    if model_name == 'r50':
        get_model, n = get_resnet50, 5
    elif model_name == 'rx50':
        get_model, n = get_resnext50, 5
    elif model_name == 'effb2':
        get_model, n = get_effb2, 6
    elif model_name == 'effb3':
        get_model,n  = get_effb3, 6

    path_model_rg_state_f1 = "weights/rg/F1/bceSL_{}_bs8x1_512_{}x3/best_model.pth".format(model_name, n)
    path_model_rg_state_f2 = "weights/rg/F2/bceSL_{}_bs8x1_512_{}x3/best_model.pth".format(model_name, n)
    path_model_rg_state_f3 = "weights/rg/F3/bceSL_{}_bs8x1_512_{}x3/best_model.pth".format(model_name, n)
    path_model_rg_state_f4 = "weights/rg/F4/bceSL_{}_bs8x1_512_{}x3/best_model.pth".format(model_name, n)
    path_model_rg_state_f5 = "weights/rg/F5/bceSL_{}_bs8x1_512_{}x3/best_model.pth".format(model_name, n)

    model_rg_f1 = get_model(n_classes=1).to(device)
    model_rg_f1.load_state_dict(torch.load(path_model_rg_state_f1, map_location=device))
    model_rg_f1.eval()

    model_rg_f2 = get_model(n_classes=1).to(device)
    model_rg_f2.load_state_dict(torch.load(path_model_rg_state_f2, map_location=device))
    model_rg_f2.eval()

    model_rg_f3 = get_model(n_classes=1).to(device)
    model_rg_f3.load_state_dict(torch.load(path_model_rg_state_f3, map_location=device))
    model_rg_f3.eval()

    model_rg_f4 = get_model(n_classes=1).to(device)
    model_rg_f4.load_state_dict(torch.load(path_model_rg_state_f4, map_location=device))
    model_rg_f4.eval()

    model_rg_f5 = get_model(n_classes=1).to(device)
    model_rg_f5.load_state_dict(torch.load(path_model_rg_state_f5, map_location=device))
    model_rg_f5.eval()

    model_rg_list = [model_rg_f1, model_rg_f2, model_rg_f3, model_rg_f4, model_rg_f5]
    return model_rg_list

class TestDataset(Dataset):
    def __init__(self, im_list, transforms):
        self.im_list = im_list
        self.transforms = transforms

    def __getitem__(self, index):
        # load image
        img = Image.open(self.im_list[index])
        img, status = crop_resize_im(img)
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.im_list)

def run():
    GLAUCOMA_THRESH = 0.05
    FEATURES_OPT_THRESH = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    BATCH_SIZE, NUM_WORKERS = 32, os.cpu_count()
    TTA = 2

    justify_model_list_r50 = load_justify_model_list('r50')
    justify_model_list_effb0 = load_justify_model_list('effb0')
    justify_model_list_effb1 = load_justify_model_list('effb1')
    justify_model_list_effb2 = load_justify_model_list('effb2')


    justify_model_list = justify_model_list_r50+justify_model_list_effb0+justify_model_list_effb1+\
                         justify_model_list_effb2

    rg_model_list_effb2 = load_rg_model_list('effb2')
    rg_model_list_effb3 = load_rg_model_list('effb3')
    rg_model_list = rg_model_list_effb2+rg_model_list_effb3

    _show_torch_cuda_info()
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # resize = tr.Resize((512, 512))
    tensorize = tr.ToTensor()
    normalize = tr.Normalize(mean, std)
    # test_transforms = tr.Compose([resize, tensorize, normalize])
    test_transforms = tr.Compose([tensorize, normalize])

    input_files = [x for x in Path("/input").rglob("*") if x.is_file()]
    print('Found {} tiff files, unpacking:'.format(len( input_files)))

    glaucoma_likelihood_probs, glaucoma_features_probs = [], []
    for tiff_image_file_name in input_files:
        print(tiff_image_file_name)
        de_stacked_images = []
        with (tempfile.TemporaryDirectory() as temp_dir):
            with Image.open(tiff_image_file_name) as tiff_image:
                # Iterate through all pages
                for page_num in range(tiff_image.n_frames):
                    # Select the current page
                    tiff_image.seek(page_num)
                    # Define the output file path
                    output_path = Path(temp_dir) / f"image_{page_num + 1}.jpg"
                    tiff_image.save(output_path, "JPEG")
                    de_stacked_images.append(output_path)
                    print(f"De-Stacked {output_path}")

                test_ds = TestDataset(de_stacked_images, transforms=test_transforms)
                test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
                with torch.inference_mode():
                    for (i_batch, images) in enumerate(test_loader):
                        if TTA == 2:
                            images = torch.cat([images, torch.flip(images, dims=(1,))], dim=0)
                        if TTA == 4:
                            images = torch.cat([images, torch.flip(images, dims=(1,)),
                                                       torch.flip(images, dims=(2,)),
                                                       torch.flip(images, dims=(1, 2))], dim=0)

                        images = images.to(device)
                        per_model_glaucoma_likelihoods = []
                        for model in rg_model_list:
                            is_referable_glaucoma_likelihoods = model(images).cpu().sigmoid().flatten().numpy()
                            n_preds = is_referable_glaucoma_likelihoods.shape[0]
                            if TTA==2:
                                if n_preds == 2: # this was single element, although b_s higher
                                    is_referable_glaucoma_likelihoods = np.atleast_1d(is_referable_glaucoma_likelihoods.mean(axis=0))
                                else:
                                    is_referable_glaucoma_likelihoods = 0.5*is_referable_glaucoma_likelihoods[:n_preds//2] +\
                                                                        0.5*is_referable_glaucoma_likelihoods[n_preds//2:]
                            if TTA==4:
                                if n_preds == 4: # this was single element, although b_s higher
                                    is_referable_glaucoma_likelihoods = np.atleast_1d(is_referable_glaucoma_likelihoods.mean(axis=0))
                                else:
                                    is_referable_glaucoma_likelihoods = 0.25*is_referable_glaucoma_likelihoods[:n_preds//4] +\
                                                                        0.25*is_referable_glaucoma_likelihoods[n_preds//4:n_preds//2] + \
                                                                        0.25*is_referable_glaucoma_likelihoods[n_preds//2:3*n_preds//4]+\
                                                                        0.25*is_referable_glaucoma_likelihoods[3*n_preds//4:]
                            per_model_glaucoma_likelihoods.append(is_referable_glaucoma_likelihoods)
                        model_avg_glaucoma_likelihoods = np.mean(np.stack(per_model_glaucoma_likelihoods, axis=0), axis=0)
                        glaucoma_likelihood_probs.extend(list(model_avg_glaucoma_likelihoods))

                        # if any image was detected as glaucomatous proceed to make prediction on features
                        glaucoma_preds_this_batch = [p>GLAUCOMA_THRESH for p in list(model_avg_glaucoma_likelihoods)]
                        if True in glaucoma_preds_this_batch:
                            per_model_glaucoma_features = []
                            for model in justify_model_list:
                                glaucoma_features = model(images).cpu().sigmoid().numpy()
                                n_preds = glaucoma_features.shape[0]
                                if TTA==2:
                                    if glaucoma_features.shape[0] == 2:  # this was single element, although Bsize higher
                                        glaucoma_features = np.atleast_2d(glaucoma_features.mean(axis=0))
                                    else:
                                        glaucoma_features = 0.5 * glaucoma_features[:n_preds//2] + 0.5 * glaucoma_features[n_preds//2:]
                                elif TTA==4:
                                    if glaucoma_features.shape[0] == 4:  # this was single element, although Bsize higher
                                        glaucoma_features = np.atleast_2d(glaucoma_features.mean(axis=0))
                                    else:
                                        glaucoma_features = 0.25 * glaucoma_features[:n_preds//4] + 0.25 * glaucoma_features[n_preds//4:n_preds//2] + \
                                                            0.25 * glaucoma_features[n_preds//2:3*n_preds//4] + 0.25 * glaucoma_features[3*n_preds//4:]
                                per_model_glaucoma_features.append(glaucoma_features)
                            glaucoma_features = np.mean(np.stack(per_model_glaucoma_features, axis=0), axis=0)
                        else:
                            glaucoma_features = justify_model_list_effb0[0](images).cpu().sigmoid().numpy()
                            if TTA==2:
                                if glaucoma_features.shape[0] == 2:  # this was single element, although Bsize higher
                                    glaucoma_features = np.atleast_2d(glaucoma_features.mean(axis=0))
                                else:
                                    glaucoma_features = 0.5 * glaucoma_features[:n_preds//2] + 0.5 * glaucoma_features[n_preds//2:]
                            elif TTA == 4:
                                if glaucoma_features.shape[0] == 4:  # this was single element, although Bsize higher
                                    glaucoma_features = np.atleast_2d(glaucoma_features.mean(axis=0))
                                else:
                                    glaucoma_features = 0.25 * glaucoma_features[:n_preds//4] + 0.25 * glaucoma_features[n_preds//4:n_preds//2] + \
                                                        0.25 * glaucoma_features[n_preds//2:3*n_preds//4] + 0.25 * glaucoma_features[3*n_preds//4:]

                        glaucoma_features_probs.append(glaucoma_features)


    glaucoma_features_probs = np.concatenate(glaucoma_features_probs, axis=0)
    #print('after loop over tiff files', glaucoma_likelihood_probs, len(glaucoma_likelihood_probs))
    #print('after loop over tiff files', glaucoma_features_probs, glaucoma_features_probs.shape)

    # prepare to save
    glaucoma_likelihood_probs = [float(g) for g in glaucoma_likelihood_probs]
    glaucoma_likelihood_preds = [bool(g > 0.5) for g in glaucoma_likelihood_probs]
    glaucoma_features = []
    for j in range(len(glaucoma_likelihood_probs)):
        d = glaucoma_features_probs[j]
        glaucoma_features.append({
            "appearance neuroretinal rim superiorly": bool(d[0] > FEATURES_OPT_THRESH[0]),
            "appearance neuroretinal rim inferiorly": bool(d[1] > FEATURES_OPT_THRESH[1]),
            "retinal nerve fiber layer defect superiorly": bool(d[2] > FEATURES_OPT_THRESH[2]),
            "retinal nerve fiber layer defect inferiorly": bool(d[3] > FEATURES_OPT_THRESH[3]),
            "baring of the circumlinear vessel superiorly": bool(d[4] > FEATURES_OPT_THRESH[4]),
            "baring of the circumlinear vessel inferiorly": bool(d[5] > FEATURES_OPT_THRESH[5]),
            "nasalization of the vessel trunk": bool(d[6] > FEATURES_OPT_THRESH[6]),
            "disc hemorrhages": bool(d[7] > FEATURES_OPT_THRESH[7]),
            "laminar dots": bool(d[8] > FEATURES_OPT_THRESH[8]),
            "large cup": bool(d[9] > FEATURES_OPT_THRESH[9]),
        })
    #print('after loop over tiff files', len(glaucoma_features), glaucoma_features)

    with open(f"/output/multiple-referable-glaucoma-binary.json", "w") as f:
        f.write(json.dumps(glaucoma_likelihood_preds))

    with open(f"/output/multiple-referable-glaucoma-likelihoods.json", "w") as f:
        f.write(json.dumps(glaucoma_likelihood_probs))

    with open(f"/output/stacked-referable-glaucomatous-features.json", "w") as f:
        f.write(json.dumps(glaucoma_features))

    return 0


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
