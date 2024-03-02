import datetime
import math
import os
import torch
import torch.utils.data
import argparse
import losses
import time
import wandb

from torch import nn
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae
from data import OpenBHB
from util import get_transforms
from models.resnet3d import SupConResNet


def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--save_freq', type=int, help='save frequency', default=50)
    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/data')
    parser.add_argument('--amp', type=arg2bool, help='use amp', default=False)
    parser.add_argument('--clip_grad', type=arg2bool, help='clip gradient to prevent nan', default=False)
    parser.add_argument('--wandb_name', type=str, help='wandb project name', default='contrastive-pretrain')

    # Model
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')

    # Optimizer
    parser.add_argument('--epochs', type=int, help='number of epochs', default=300)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step'], default='step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="700,800,900")
    parser.add_argument('--lr_decay_step', type=int, help='decay rate step (overwrites lr_decay_epochs', default=10)
    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--optimizer', type=str, help="optimizer (adam or sgd)", choices=["adam", "sgd"],
                        default="adam")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=5e-5)

    # Data
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'all'],
                        default='none')

    # Loss
    parser.add_argument('--method', type=str, help='loss function', choices=['supcon', 'yaware', 'threshold', 'expw'],
                        default='supcon')
    parser.add_argument('--kernel', type=str, help='Kernel function (not for supcon)',
                        choices=['cauchy', 'gaussian', 'rbf'], default=None)
    parser.add_argument('--delta_reduction', type=str,
                        help='use mean or sum to reduce 3d delta mask (only for method=threshold)', default='sum')
    parser.add_argument('--temp', type=float, help='loss temperature', default=0.1)
    parser.add_argument('--alpha', type=float, help='infonce weight', default=1.)
    parser.add_argument('--sigma', type=float, help='gaussian-rbf kernel sigma / cauchy gamma', default=1)
    parser.add_argument('--n_views', type=int, help='num. of multiviews', default=2)

    opts = parser.parse_args()

    #check if cuda is available
    if not torch.cuda.is_available():
        opts.device = 'cpu'
        print("CUDA not available, using CPU")
    else:
        opts.device = 'cuda'
        print("CUDA available, using GPU")


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

    return opts


def load_data(opts):
    T_train, T_test = get_transforms(opts)
    T_train = NViewTransform(T_train, opts.n_views)

    train_dataset = OpenBHB(opts.data_dir, train=True, transform=T_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    train_loader_score = torch.utils.data.DataLoader(
        OpenBHB(opts.data_dir, train=True, transform=T_train), batch_size=opts.batch_size, shuffle=True)

    test = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, transform=T_test),
                                                batch_size=opts.batch_size, shuffle=False)

    return train_loader, train_loader_score, test


def load_model(opts):
    if 'resnet' in opts.model:
        model = SupConResNet(opts.model, feat_dim=128)
    else:
        raise ValueError("Unknown model", opts.model)

    if opts.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    model = model.to(opts.device)

    def gaussian_kernel(x):
        x = x - x.T
        return torch.exp(-(x ** 2) / (2 * (opts.sigma ** 2))) / (math.sqrt(2 * torch.pi) * opts.sigma)

    def rbf(x):
        x = x - x.T
        return torch.exp(-(x ** 2) / (2 * (opts.sigma ** 2)))

    def cauchy(x):
        x = x - x.T
        return 1. / (opts.sigma * (x ** 2) + 1)

    kernels = {
        'none': None,
        'cauchy': cauchy,
        'gaussian': gaussian_kernel,
        'rbf': rbf
    }

    infonce = losses.KernelizedSupCon(method=opts.method, temperature=opts.temp, kernel=kernels[opts.kernel],
                                      delta_reduction=opts.delta_reduction)

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
    for idx, (images, labels, sites) in enumerate(train_loader):
        data_time.update(time.time() - t1)

        images = torch.cat(images, dim=0).to(opts.device)
        images = torch.unsqueeze(images, 1)
        bsz = labels.shape[0]

        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast(scaler is not None):
            projected = model(images)
            projected = torch.split(projected, [bsz] * opts.n_views, dim=0)
            projected = torch.cat([f.unsqueeze(1) for f in projected], dim=1)

            running_loss = infonce(projected, labels.to(opts.device))

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

    run_name = (f"_{model_name}_{method_name}_"
                f"{optimizer_name}_"
                f"tf{opts.tf}_"
                f"lr{opts.lr}_{opts.lr_decay}_step{opts.lr_decay_step}_rate{opts.lr_decay_rate}_"
                f"temp{opts.temp}_"
                f"wd{opts.weight_decay}_"
                f"bsz{opts.batch_size}_views{opts.n_views}_"
                f"kernel_{kernel_name}_"
                f"f{opts.alpha}_"
                f"trial{opts.trial}")
    save_dir = os.path.join(opts.save_dir, f"openbhb_pretrain_models", run_name)
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

        print(f"epoch {epoch}, total time {t2 - start_time:.2f}, epoch time {t2 - t1:.3f} loss {loss_train:.4f}")
        mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        wandb.log({"train/mae": mae_train, "test/mae": mae_test, "epoch": epoch})

        if epoch % opts.save_freq == 0:
            print("Age MAE:", mae_train, mae_test)

            save_file = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
            save_model(model, optimizer, opts, epoch, save_file)

    mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
    wandb.log({"train/mae": mae_train, "test/mae": mae_test, "epoch": epoch})

    wandb.log({'mae_train': mae_train})
    wandb.log({'mae_test': mae_test})