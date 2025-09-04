import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from copy import deepcopy

import moco.builder
import moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import sys
from pathlib import Path

from models.yolo import Model
from utils.torch_utils import select_device

from moco.utils import AverageMeter, ProgressMeter, adjust_learning_rate, accuracy, save_checkpoint

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch SID RAW Training")
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/datasets/steelpipe_moco/images", help="dataset path")
    parser.add_argument("--pair_list", type=str, default="train.txt", help="dataset path")
    parser.add_argument("--epochs", type=int, default=200, help="number of total epochs to run")
    parser.add_argument("--start-epoch", type=int, default=0, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", type=int, default=128, help="mini batch")
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="momentum of SGD solver")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay (default: 1e-4)",)
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", default=32, type=int, help="number of data loading workers (default: 32)")
    parser.add_argument("--save_path", type=str, default="/root/autodl-tmp", help="dataset path")
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    device = select_device(args.device)

    print(">>> Creating Model <<<")
    model = moco.builder.MoCo(Model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    print(">>> Data Loading <<<")
    # train_set = moco.loader.MoCoData(args.dataset_path, args.pair_list)
    #mocov2
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    train_dataset = datasets.ImageFolder(
        args.dataset_path, moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    print(">>> Training <<<")
    for epoch in range(args.start_epoch, args.epochs):
        step_one_train(train_loader, model, criterion, optimizer, epoch, args, device)
        if epoch>=180:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": model.float().state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=os.path.join(args.save_path, "checkpoint_{:04d}.pt".format(epoch))
            )


def step_one_train(train_loader, model, criterion, optimizer, epoch, args, device):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()

    end = time.time()

    for i, (images, _)in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_q = images[0].to(device)
        input_k = images[1].to(device)

        output, target = model(im_q=input_q, im_k=input_k)
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_q.size(0))
        top1.update(acc1[0], input_q.size(0))
        top5.update(acc5[0], input_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
if __name__ == "__main__":
    main()
