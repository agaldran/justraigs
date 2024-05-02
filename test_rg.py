import sys, json, os, time, random
import os.path as osp
import numpy as np
import torch
from tqdm import trange
from utils.data_load import get_train_val_loaders, modify_loader
from utils.metric_factory import hamming_loss
from utils.model_factory_mh import get_model
from utils.loss_factory import get_loss
from utils.sam import SAM, disable_running_stats, enable_running_stats
def get_args_parser():
    import argparse

    def str2bool(v):
        # as seen here: https://stackoverflow.com/a/43357954/3208255
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', 'yes'):
            return True
        elif v.lower() in ('false', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('boolean value expected.')

    parser = argparse.ArgumentParser(description='Training for Biomedical Image Classification')
    parser.add_argument('--csv_path_tr', type=str, default='data/tr_mini_madder.csv', help='csv path training data')
    parser.add_argument('--model_name', type=str, default='resnet18', help='architecture')
    parser.add_argument('--n_heads', type=int, default=1, help='model heads')
    parser.add_argument('--batch_size', type=int, default=8, help=' batch size')
    parser.add_argument('--im_size', type=str, default='512/512', help='im size/spatial xy dimension')
    parser.add_argument('--save_path', type=str, default='delete', help='path to save model (defaults to date/time')
    parser.add_argument('--seed', type=int, default=None, help='fixes random seed (slower!)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers')
    args = parser.parse_args()

    return args


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

def build_preds(model, loader):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    y_prob, y_true = [], []
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for (i_batch, batch_data) in enumerate(loader):
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs_0 = model(inputs).sigmoid()
            outputs_1 = model(torch.flip(inputs, dims=(1,))).sigmoid()
            outputs_2 = model(torch.flip(inputs, dims=(2,))).sigmoid()
            outputs_3 = model(torch.flip(inputs, dims=(1,2))).sigmoid()
            outputs = torch.stack([outputs_0, outputs_1, outputs_2, outputs_3], dim=0).mean(dim=0)
            y_prob.append(outputs.cpu().numpy().tolist())
            y_true.append(labels.cpu().numpy().tolist())
            n_elems += 1
            t.update()
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()

    
    return y_true, y_prob

if __name__ == '__main__':

    args = get_args_parser()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # logging
    load_path = osp.join('experiments', args.save_path)

    # gather parser parameters
    model_name = args.model_name
    bs, csv_path_tr, nw = args.batch_size, args.csv_path_tr, args.num_workers

    im_size = args.im_size.split('/')
    im_size = tuple(map(int, im_size))

    print('* Instantiating a {} model'.format(model_name))
    n_classes = 1
    n_heads = args.n_heads
    model = get_model(args.model_name, n_classes=n_classes, n_heads=n_heads)
    model.load_state_dict(torch.load(osp.join(load_path, 'best_model.pth')))
    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs, nw))

    _, _, vl_loader = get_train_val_loaders(csv_path_tr, bs, im_size, soft_labels=False, multi_label=False, num_workers=nw)

    model = model.to(device)
    print('Total params: {0:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    print('* Starting to test\n', '-' * 10)
    start = time.time()
    with torch.no_grad():
        y_true, y_prob = build_preds(model, vl_loader)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Testing time: {:0>2}min {:05.2f}secs'.format(int(minutes), seconds))
    with (open(osp.join(load_path, 'log.txt'), 'a') as f):
        print('\nTesting time: {:0>2}min {:05.2f}secs'.format(int(minutes), seconds), file=f)
    preds_name = osp.join(load_path, 'preds.npy')
    labels_name = osp.join(load_path, 'labels.npy')

    np.save(preds_name, y_prob)
    np.save(labels_name, y_true)
    hamming = hamming_loss(y_true, y_prob)

