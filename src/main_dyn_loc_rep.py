import datetime
import math
import os
import torch
import torch.utils.data
import argparse
import losses
import time
import wandb
import json

from torch import nn
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae
from data import MREDataset
from util import get_transforms
from models.resnet3d import SupConResNet


def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--fold', type=int, help='fold (of 5 CV)', default=0)
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--save_freq', type=int, help='save frequency', default=1) # TODO: default 50
    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/data')
    parser.add_argument('--amp', type=arg2bool, help='use amp', default=False)
    parser.add_argument('--clip_grad', type=arg2bool, help='clip gradient to prevent nan', default=False)
    parser.add_argument('--wandb_name', type=str,
                        help='wandb project name', default='contrastive-brain-age-prediction')
    parser.add_argument('--save_model', type=arg2bool, help='save model', default=False)

    # Model
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')
    parser.add_argument('--pretrained', type=str, help='use pretrained model',
                        choices=['expw', 'not'], default='not')

    # Optimizer
    parser.add_argument('--epochs', type=int, help='number of epochs', default=300)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--lr_decay', type=str, help='type of decay',
                        choices=['cosine', 'step'], default='step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str,
                        help='steps of lr decay (list)', default="700,800,900")
    parser.add_argument('--lr_decay_step', type=int,
                        help='decay rate step (overwrites lr_decay_epochs', default=10)
    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--optimizer', type=str, help="optimizer (adam or sgd)",
                        choices=["adam", "sgd"], default="adam")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=5e-5)

    # Data
    parser.add_argument('--tf', type=str, help='data augmentation',
                        choices=['none', 'crop', 'cutout', 'all', 'noise'], default='none')
    parser.add_argument('--noise_std', type=float, help='std for noise augmentation', default=0.05)
    parser.add_argument('--modality', type=str, help='imaging modality',
                        choices=['T1', 'dr', 'stiffness'], default='stiffness')
    
    # Loss 
    parser.add_argument('--method', type=str, help='loss function',
                        choices=['supcon', 'yaware', 'threshold', 'expw'], default='supcon')
    parser.add_argument('--kernel', type=str, help='Kernel function (not for supcon)',
                        choices=['cauchy', 'gaussian', 'rbf'], default=None)
    parser.add_argument('--delta_reduction', type=str,
                        help='use mean or sum to reduce 3d delta mask (only for method=threshold)', default='sum')
    parser.add_argument('--temp', type=float, help='loss temperature', default=0.1)
    parser.add_argument('--alpha', type=float, help='infonce weight', default=1.)
    parser.add_argument('--sigma', type=float, help='gaussian-rbf kernel sigma / cauchy gamma', default=1)
    parser.add_argument('--n_views', type=int, help='num. of multiviews', default=2)

    # hyperparameter for new modifications
    parser.add_argument('--NN_nb_step_size', type=int, help='step size for NN_nb', default=0)
    parser.add_argument('--end_NN_nb', type=int, help='label type', default=4)
    parser.add_argument('--NN_nb_selection', type=str, help='selection method for NN_nb',
                        choices=['euclidean', 'similarity', 'manhattan', 'chebyshev', 'no'], default='similarity')

    opts = parser.parse_args()

    if opts.batch_size > 256:
        print("Forcing warm")
        opts.warm = True

    if opts.lr_decay_step is not None:
        opts.lr_decay_epochs = list(range(opts.lr_decay_step, opts.epochs, opts.lr_decay_step))
        print(f"Computed decay epochs based on step ({opts.lr_decay_step}):", opts.lr_decay_epochs)
    else:
        iterations = opts.lr_decay_epochs.split(',')
        opts.lr_decay_epochs = list([])
        for it in iterations:
            opts.lr_decay_epochs.append(int(it))

    if opts.warm:
        opts.warmup_from = 0.01
        opts.warm_epochs = 10
        if opts.lr_decay == 'cosine':
            eta_min = opts.lr * (opts.lr_decay_rate ** 3)
            opts.warmup_to = eta_min + (opts.lr - eta_min) * (
                    1 + math.cos(math.pi * opts.warm_epochs / opts.epochs)) / 2
        else:
            opts.milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
            opts.warmup_to = opts.lr

    if opts.method == 'supcon':
        print('method == supcon, binning age')
        opts.label = 'bin'
    else:
        print('method != supcon, using real age value')
        opts.label = 'cont'

    if opts.method == 'supcon' and opts.kernel is not None:
        print('Invalid kernel for supcon')
        exit(0)
    
    if opts.method != 'supcon' and opts.kernel is None:
        print('Kernel cannot be None for method != supcon')
        exit(1)
    
    if opts.model == 'densenet121':
        opts.n_views = 1

    if torch.cuda.is_available():
        print('on GPU')
        device = 'cuda'
        opts.device = device
        opts.location = 'cluster'
    else:
        print('not on GPU')
        device = 'cpu'
        opts.device = device
        opts.location = 'local'

    return opts


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_arguments_json():
    config_path = '/home/jnt27/contrastive-brain-age-MRE/src/exp/supcon_adam_kernel.json'
    config_dict = load_config(config_path)
    opts = Config(config_dict)

    if opts.batch_size > 256:
        print("Forcing warm")
        opts.warm = True

    if opts.lr_decay_step is not None:
        opts.lr_decay_epochs = list(range(opts.lr_decay_step, opts.epochs, opts.lr_decay_step))
        print(f"Computed decay epochs based on step ({opts.lr_decay_step}):", opts.lr_decay_epochs)
    else:
        iterations = opts.lr_decay_epochs.split(',')
        opts.lr_decay_epochs = list([])
        for it in iterations:
            opts.lr_decay_epochs.append(int(it))

    if opts.warm:
        opts.warmup_from = 0.01
        opts.warm_epochs = 10
        if opts.lr_decay == 'cosine':
            eta_min = opts.lr * (opts.lr_decay_rate ** 3)
            opts.warmup_to = eta_min + (opts.lr - eta_min) * (
                    1 + math.cos(math.pi * opts.warm_epochs / opts.epochs)) / 2
        else:
            opts.milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
            opts.warmup_to = opts.lr

    if opts.method == 'supcon':
        print('method == supcon, binning age')
        opts.label = 'bin'
    else:
        print('method != supcon, using real age value')
        opts.label = 'cont'

    if opts.method == 'supcon' and opts.kernel is not None:
        print('Invalid kernel for supcon')
        exit(0)

    if opts.method != 'supcon' and opts.kernel is None:
        print('Kernel cannot be None for method != supcon')
        exit(1)

    if opts.model == 'densenet121':
        opts.n_views = 1

    if torch.cuda.is_available():
        print('on GPU')
        device = 'cuda'
        opts.device = device
        opts.location = 'cluster'
    else:
        print('not on GPU')
        device = 'cpu'
        opts.device = device

    return opts


def load_data(opts):
    T_train, T_test = get_transforms(opts)
    T_train = NViewTransform(T_train, opts.n_views)

    train_dataset = MREDataset(opts.modality, transform=T_train, train=True, location=opts.location, fold=opts.fold)

    train_dataset.norm()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)

    train_dataset_score = MREDataset(opts.modality, transform=T_train, train=True, location=opts.location,
                                     fold=opts.fold)

    train_dataset_score.norm()

    train_loader_score = torch.utils.data.DataLoader(train_dataset_score, batch_size=opts.batch_size, shuffle=False)

    test_dataset = MREDataset(opts.modality, transform=T_test, train=False, location=opts.location, fold=opts.fold)

    test_dataset.norm()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)

    return train_loader, train_loader_score, test_loader


def load_model(opts):
    if 'resnet' in opts.model:
        model = SupConResNet(opts.model, feat_dim=128)
    else:
        raise ValueError("Unknown model", opts.model)

    if opts.pretrained == 'expw':
        print("Using pretrained expw model")
        save_pretrained_file = ('/rds/user/jnt27/hpc-work/contrastive-brain-age-MRE/output/brain-age-mri/pretrain/'
                                'openbhb_pretrain_models/resnet18_expw_adam_tfnone_lr0.0001_step_step10_rate0.9_'
                                'temp0.1_wd5e-05_bsz32_views2_kernel_rbf_sigma2.0_f1.0_trial0/ckpt_epoch_300.pth')
        if torch.cuda.is_available():
            state = torch.load(save_pretrained_file)
        else:
            state = torch.load(save_pretrained_file, map_location='cpu')

        model.load_state_dict(state['model'])
        del state

    if opts.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)

    model = model.to(opts.device)

    def gaussian_kernel(x):
        x = x - x.T
        return torch.exp(-(x**2) / (2*(opts.sigma**2))) / (math.sqrt(2*torch.pi)*opts.sigma)
    
    def rbf(x):
        x = x - x.T
        return torch.exp(-(x**2)/(2*(opts.sigma**2)))
    
    def cauchy(x):
        x = x - x.T
        return 1. / (opts.sigma*(x**2) + 1)

    kernels = {
        'none': None,
        'cauchy': cauchy,
        'gaussian': gaussian_kernel,
        'rbf': rbf
    }

    infonce = losses.DynLocRep_loss(method=opts.method, temperature=opts.temp, kernel=kernels[opts.kernel],
                                    delta_reduction=opts.delta_reduction, epochs=opts.epochs,
                                    NN_nb_step_size=opts.NN_nb_step_size, end_NN_nb=opts.end_NN_nb,
                                    NN_nb_selection=opts.NN_nb_selection)

    infonce = infonce.to(opts.device)

    return model, infonce


def load_optimizer(model, opts):
    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, 
                                    momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

    return optimizer


def train(train_loader, model, infonce, optimizer, opts, epoch):
    loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    scaler = torch.cuda.amp.GradScaler() if opts.amp else None
    model.train()

    t1 = time.time()
    for idx, (images, labels, (_)) in enumerate(train_loader):
        data_time.update(time.time() - t1)

        images = torch.cat(images, dim=0).to(opts.device)
        images = torch.unsqueeze(images, 1)
        bsz = labels.shape[0]

        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast(scaler is not None):
            projected = model(images)
            projected = torch.split(projected, [bsz]*opts.n_views, dim=0)
            projected = torch.cat([f.unsqueeze(1) for f in projected], dim=1)

            if opts.NN_nb_step_size > 0:
                running_loss = infonce(features=projected,
                                       labels=labels.to(opts.device),
                                       epoch=epoch)

            else:
                running_loss = infonce(features=projected,
                                       labels=labels.to(opts.device))
        
        optimizer.zero_grad()
        if scaler is None:
            running_loss.backward()
            if opts.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        else:
            scaler.scale(running_loss).backward()
            if opts.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
        
        loss.update(running_loss.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t")

    return loss.avg, batch_time.avg, data_time.avg


if __name__ == '__main__':
    opts = parse_arguments()
    
    set_seed(opts.trial)

    train_loader, train_loader_score, test_loader = load_data(opts)
    model, infonce = load_model(opts)
    optimizer = load_optimizer(model, opts)

    model_name = opts.model
    if opts.warm:
        model_name = f"{model_name}_warm"
    if opts.amp:
        model_name = f"{model_name}_amp"
    
    method_name = opts.method
    if opts.method == 'threshold':
        method_name = f"{method_name}_reduction_{opts.delta_reduction}"

    optimizer_name = opts.optimizer
    if opts.clip_grad:
        optimizer_name = f"{optimizer_name}_clipgrad"

    kernel_name = opts.kernel
    if opts.kernel == "gaussian" or opts.kernel == 'rbf':
        kernel_name = f"{kernel_name}_sigma{opts.sigma}"
    elif opts.kernel == 'cauchy':
        kernel_name = f"{kernel_name}_gamma{opts.sigma}"

    if opts.pretrained == 'expw':
        prefix = 'pretrained_expw'
    else:
        prefix = 'no_pretrained'

    run_name = (f"{prefix}_{opts.modality}_{method_name}_E{opts.epochs}_trial{opts.trial}_fold{opts.fold}_"
                f"NN_nb_selection_{opts.NN_nb_selection}_end_NN_nb_{opts.end_NN_nb}"
                f"_NN_nb_step_size_{opts.NN_nb_step_size}")

    save_dir = os.path.join(opts.save_dir, f"models", run_name)
    if opts.save_model:
        ensure_dir(save_dir)

    opts.model_class = model.__class__.__name__
    opts.criterion = infonce.__class__.__name__
    opts.optimizer_class = optimizer.__class__.__name__

    wandb.init(entity='jakobwandb', project=opts.wandb_name, config=opts, name=run_name,
               settings=wandb.Settings(code_dir="/src"), tags=['to test'])

    print('Config:', opts)
    print('Model:', model.__class__.__name__)
    print('Criterion:', infonce)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)

    if opts.amp:
        print("Using AMP")
    
    start_time = time.time()
    best_acc = 0.
    for epoch in range(1, opts.epochs + 1):
        adjust_learning_rate(opts, optimizer, epoch)

        t1 = time.time()
        loss_train, batch_time, data_time = train(train_loader, model, infonce, optimizer, opts, epoch)
        t2 = time.time()
        wandb.log({"train/loss": loss_train, "lr": optimizer.param_groups[0]['lr'], "BT": batch_time, "DT": data_time,
                   "epoch": epoch})

        print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} loss {loss_train:.4f}")
        mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        wandb.log({"train/mae": mae_train, "test/mae": mae_test, "epoch": epoch})

        if epoch % opts.save_freq == 0:
            if opts.save_model:
                print(f"SAVING MODEL... (mae_test: {mae_test:.2f})")
                save_file = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
                save_model(model, optimizer, opts, epoch, save_file)

            print("Age MAE:", mae_train, mae_test)
    
    mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
    wandb.log({"train/mae": mae_train, "test/mae": mae_test, "epoch": epoch})

    wandb.log({'mae_train': mae_train})
    wandb.log({'mae_test': mae_test})
