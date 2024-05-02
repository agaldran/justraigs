import sys, json, os, time, random
import os.path as osp
import numpy as np
import torch
from tqdm import trange
from utils.data_load import get_train_val_loaders, modify_loader
from utils.metric_factory import get_imb_metrics
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
    parser.add_argument('--csv_path_tr', type=str, default='data/tr_mini.csv', help='csv aith trainind data')
    parser.add_argument('--model_name', type=str, default='resnet18', help='architecture')
    parser.add_argument('--n_heads', type=int, default=1, help='model heads')
    parser.add_argument('--loss1', type=str, default='bce',   choices=('bce', 'f1'), help='1st loss')
    parser.add_argument('--loss2', type=str, default=None, choices=('bce', 'f1', None), help='2nd loss')
    parser.add_argument('--alpha1', type=float, default=1., help='multiplier in alpha1*loss1+alpha2*loss2')
    parser.add_argument('--alpha2', type=float, default=0., help='multiplier in alpha1*loss1+alpha2*loss2')
    parser.add_argument('--bce_ls', type=float, default=0., help='label smoothing for bce loss')
    parser.add_argument('--batch_size', type=int, default=8, help=' batch size')
    parser.add_argument('--sam', type=str2bool, nargs='?', const=True, default=False, help='use sam wrapping optimizer')
    parser.add_argument('--ovsmpl', type=str2bool, nargs='?', const=True, default=False, help='oversampling (sqrt)')
    parser.add_argument('--soft_labels', type=str2bool, nargs='?', const=True, default=False, help='soft labels - multi-rater')
    parser.add_argument('--im_size', type=str, default='512/512', help='im size/spatial xy dimension')
    parser.add_argument('--optimizer', type=str, default='nadam', choices=('sgd', 'adamw', 'nadam'), help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-4, help='max learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0, help='label smoothing')
    parser.add_argument('--n_epochs', type=int, default=15, help='training epochs')
    parser.add_argument('--vl_interval', type=int, default=3, help='how often we check performance and maybe save')
    parser.add_argument('--cyclical_lr', type=str2bool, nargs='?', const=True, default=True, help='re-start lr each vl_interval epochs')
    parser.add_argument('--metric', type=str, default='AP', help='which metric to use for monitoring progress (AUC)')
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def validate(model, loader, loss_fn=torch.nn.functional.binary_cross_entropy_with_logits):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    y_prob, y_true, losses = [], [], []
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for (i_batch, batch_data) in enumerate(loader):
            # if i_batch < 1264: continue
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.view_as(outputs))

            losses.append(loss.item())
            y_prob.extend(outputs.sigmoid().cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())

            n_elems += 1
            running_loss += loss
            run_loss = running_loss / n_elems
            t.set_postfix(LOSS='{:.2f}'.format(100 * run_loss))
            t.update()

    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()

    au_prc, ap, sens_at_95_spec = get_imb_metrics(y_true, y_prob)
    m = [100 * au_prc, 100 * ap, 100 * sens_at_95_spec, np.mean(np.array(losses))]
    return m

def disable_bn(model):
    for module in model.modules():
      if isinstance(module, torch.nn.BatchNorm3d):
        module.eval()
def enable_bn(model):
    model.train()

def train_one_epoch(model, loader, loss_fn, optimizer, scheduler):
    model.train()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for (i_batch, batch_data) in enumerate(loader):
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.float().to(device)
            # first forward-backward step
            enable_running_stats(model)
            logits = model(inputs).squeeze()
            loss = loss_fn(logits, labels)

            loss.backward()
            if isinstance(optimizer, SAM):
                optimizer.first_step(zero_grad=True)
                # second forward-backward step
                disable_running_stats(model)
                logits = model(inputs).squeeze()
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            lr=get_lr(optimizer)
            running_loss += loss.detach().item() * inputs.shape[0]
            n_elems += inputs.shape[0]  # total nr of items processed
            run_loss = running_loss / n_elems
            t.set_postfix(LOSS_lr='{:.4f}/{:.6f}'.format(run_loss, lr))
            t.update()

def set_tr_info(tr_info, epoch=0, ovft_metrics=None, vl_metrics=None, best_epoch=False):
    # I customize this for each project.
    # Here tr_info contains Dice Scores, AUCs, and loss values.
    # Also, and vl_metrics contain (in this order) dice, auc and loss
    if best_epoch:
        tr_info['best_tr_prauc'] = tr_info['tr_praucs'][-1]
        tr_info['best_vl_prauc'] = tr_info['vl_praucs'][-1]
        tr_info['best_tr_ap'] = tr_info['tr_aps'][-1]
        tr_info['best_vl_ap'] = tr_info['vl_aps'][-1]
        tr_info['best_tr_se@95sp'] = tr_info['tr_se@95sps'][-1]
        tr_info['best_vl_se@95sp'] = tr_info['vl_se@95sps'][-1]
        tr_info['best_tr_loss'] = tr_info['tr_losses'][-1]
        tr_info['best_vl_loss'] = tr_info['vl_losses'][-1]
        tr_info['best_epoch'] = epoch
    else:
        tr_info['tr_praucs'].append(ovft_metrics[0])
        tr_info['vl_praucs'].append(vl_metrics[0])
        tr_info['tr_aps'].append(ovft_metrics[1])
        tr_info['vl_aps'].append(vl_metrics[1])
        tr_info['tr_se@95sps'].append(ovft_metrics[2])
        tr_info['vl_se@95sps'].append(vl_metrics[2])
        tr_info['tr_losses'].append(ovft_metrics[-1])
        tr_info['vl_losses'].append(vl_metrics[-1])

    return tr_info

def init_tr_info():
    # I customize this function for each project.
    tr_info = dict()
    tr_info['tr_praucs'], tr_info['vl_praucs'] = [], []
    tr_info['tr_aps'], tr_info['vl_aps'] = [], []
    tr_info['tr_se@95sps'], tr_info['vl_se@95sps'] = [], []
    tr_info['tr_losses'], tr_info['vl_losses'] = [], []

    return tr_info

def get_eval_string(tr_info, epoch, finished=False, vl_interval=1):
    # I customize this function for each project.
    # Pretty prints first three values of train/val metrics to a string and returns it
    # Used also by the end of training (finished=True)
    ep_idx = len(tr_info['tr_losses'])-1
    if finished:
        ep_idx = epoch
        epoch = (epoch+1) * vl_interval - 1

    s = 'Ep. {}: Train||Val AP: {:5.2f}||{:5.2f} - PR-AUC: {:5.2f}||{:5.2f} - Sens@95Spec: {:5.2f}||{:5.2f} - Loss: {:.4f}||{:.4f}'.format(
        str(epoch+1).zfill(3), tr_info['tr_aps'][ep_idx], tr_info['vl_aps'][ep_idx],
                                     tr_info['tr_praucs'][ep_idx], tr_info['vl_praucs'][ep_idx],
                                     tr_info['tr_se@95sps'][ep_idx], tr_info['vl_se@95sps'][ep_idx],
                                     tr_info['tr_losses'][ep_idx], tr_info['vl_losses'][ep_idx])
    return s
def train_model(model, optimizer, loss_fn, tr_loader, ovft_loader, vl_loader, ovsmpl, scheduler, metric, n_epochs, vl_interval, save_path):
    best_metric, best_epoch = -1, 0
    tr_info = init_tr_info()

    for epoch in range(n_epochs):
        print('Epoch {:d}/{:d}'.format(epoch + 1, n_epochs))
        # train one epoch
        loader = modify_loader(tr_loader, oversampling=ovsmpl)
        train_one_epoch(model, loader, loss_fn, optimizer, scheduler)
        if (epoch + 1) % vl_interval == 0:
            with torch.no_grad():
                ovft_metrics = validate(model, ovft_loader)
                vl_metrics = validate(model, vl_loader)
            tr_info = set_tr_info(tr_info, epoch, ovft_metrics, vl_metrics)
            s = get_eval_string(tr_info, epoch)
            print(s)

            with open(osp.join(save_path, 'train_log.txt'), 'a') as f:
                print(s, file=f)
            # check if performance was better than anyone before and checkpoint if so
            if metric == 'AP': curr_metric = tr_info['vl_aps'][-1]
            elif metric =='PRAUC': curr_metric = tr_info['vl_praucs'][-1]
            elif metric == 'SE@95SP': curr_metric = tr_info['vl_se@95sps'][-1]
            else: sys.exit('bad metric')

            if curr_metric > best_metric:
                print('-------- Best {} attained. {:.2f} --> {:.2f} --------'.format(metric, best_metric, curr_metric))
                torch.save(model.state_dict(), osp.join(save_path, 'best_model.pth'))
                best_metric, best_epoch = curr_metric, epoch + 1
                tr_info = set_tr_info(tr_info, epoch+1, best_epoch=True)
            else:
                print('-------- Best {} so far {:.2f} at epoch {:d} --------'.format(metric, best_metric, best_epoch))
    torch.save(model.state_dict(), osp.join(save_path, 'last_model.pth'))
    del model, tr_loader, vl_loader
    # maybe this works also? tr_loader.dataset._fill_cache
    torch.cuda.empty_cache()
    return tr_info

if __name__ == '__main__':

    args = get_args_parser()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # logging
    save_path = osp.join('experiments', args.save_path)
    os.makedirs(save_path, exist_ok=True)
    config_file_path = osp.join(save_path, 'config.cfg')
    with open(config_file_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # gather parser parameters
    model_name = args.model_name
    optimizer_choice = args.optimizer
    lr, bs, ovsmpl, sl = args.lr, args.batch_size, args.ovsmpl, args.soft_labels
    n_epochs, vl_interval, metric = args.n_epochs, args.vl_interval, args.metric
    csv_path_tr, nw = args.csv_path_tr, args.num_workers

    im_size = args.im_size.split('/')
    im_size = tuple(map(int, im_size))

    print('* Instantiating a {} model'.format(model_name))
    n_classes = 1
    n_heads = args.n_heads
    model = get_model(args.model_name, n_classes=n_classes, n_heads=n_heads)

    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs, nw))

    tr_loader, ovft_loader, vl_loader = get_train_val_loaders(csv_path_tr, bs, im_size, soft_labels=sl, num_workers=nw)
    # x, y = next(iter(tr_loader))
    # print(x.shape)
    # print(model(x).shape)
    # sys.exit()

    model = model.to(device)
    print('Total params: {0:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if optimizer_choice == 'adam':
        if args.sam:
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'nadam':
        if args.sam:
            base_optimizer = torch.optim.NAdam
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr)
        else:
            optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'sgd':
        if args.sam:
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    else:
        sys.exit('please choose between sgd, adam or nadam optimizers')

    if args.cyclical_lr:
        if args.sam: scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=vl_interval*len(tr_loader), eta_min=0)
        else: scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=vl_interval*len(tr_loader), eta_min=0)
    else:
        if args.sam: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=n_epochs*len(tr_loader), eta_min=0)
        else: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs*len(tr_loader), eta_min=0)

    if n_heads>1:
        sys.exit('use n_heads=1 with this script')
    else:
        loss_fn = get_loss(args.loss1, args.loss2, args.alpha1, args.alpha2, args.bce_ls)
        print('* Instantiating loss function {:.2f}*{} + {:.2f}*{}'.format(args.alpha1, args.loss1, args.alpha2, args.loss2))

    print('* Starting to train\n', '-' * 10)
    start = time.time()
    tr_info = train_model(model, optimizer, loss_fn, tr_loader, ovft_loader, vl_loader, ovsmpl, scheduler, metric, n_epochs, vl_interval, save_path)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

    with (open(osp.join(save_path, 'log.txt'), 'a') as f):
        print('Best epoch = {}/{}: Tr/Vl AP={:.2f}/{:.2f} - Tr/Vl PR-AUC={:.2f}/{:.2f} - Tr/Vl Sens@95Spec={:.2f}/{:.2f} - Loss = {:.5f}/{:.5f}\n'.format(
      tr_info['best_epoch'], n_epochs, tr_info['best_tr_ap'], tr_info['best_vl_ap'], tr_info['best_tr_prauc'],
            tr_info['best_vl_prauc'], tr_info['best_tr_se@95sp'], tr_info['best_vl_se@95sp'],
            tr_info['best_tr_loss'], tr_info['best_vl_loss']), file=f)
        for j in range(n_epochs//vl_interval):
            s = get_eval_string(tr_info, epoch=j, finished=True, vl_interval=vl_interval)
            print(s, file=f)
        print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

    print('Done. Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))
