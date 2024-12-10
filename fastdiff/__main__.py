#
# 3rd party
import torch
from tqdm import tqdm

import torchvision
from torchvision import transforms

# builtin
import os
import shutil
import argparse

# local
import mlutils
import fastdiff

OUT_DIR = 'out/'
DATA_DIR = 'data/'

#======================================================================#
def main(dataset, device, args):
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    out_dir = os.path.join(OUT_DIR, args.case_dir)
    model_file = os.path.join(out_dir, 'model.pt')

    if LOCAL_RANK == 0:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    #=================#
    # MODEL
    #=================#

    ###
    # UNET (batch_size = 16 * 3)
    ###

    batch_size = 12
    if args.mode == 0:
        lr = 1e-3
        weight_decay = 0e-0
        model = fastdiff.UNet(32)
        nepochs = 100
    else:
        # lr = 1e-4
        # weight_decay = 1e-1
        # model = fastdiff.UNet(32)
        # nepochs = 100

        lr = 1e-4
        weight_decay = 1e-1
        model = fastdiff.UNet(32)
        nepochs = 100

    ###
    # DIT
    ###

    # lr = 1e-4
    # weight_decay = 1e-1 # 1e-1 - 1e-2
    # batch_size = 8
    # model = fastdiff.DiT_S_2(input_size=args.image_size, in_channels=3,)

    #=================#
    model = fastdiff.Diffusion(model, args.mode)
    #=================#

    #=================#
    # TRAIN
    #=================#

    callback = fastdiff.Callback(
        out_dir, args.image_size, args.save_every,
        dataset.root, args.noise_seed,
    )

    def callback_fn(trainer: mlutils.Trainer):
        if LOCAL_RANK == 0:
            callback(trainer)
        return

    if args.train:
        def batch_lossfun(batch, trainer: mlutils.Trainer):
            return trainer.model(batch)

        kw = dict(
            Opt='AdamW', lr=lr, nepochs=nepochs, weight_decay=weight_decay,
            _batch_size=batch_size, static_graph=True, drop_last=True,
            batch_lossfun=batch_lossfun, device=device, stats_every=-1,
        )

        trainer = mlutils.Trainer(model, dataset, **kw)
        trainer.add_callback('epoch_end', callback_fn)
        trainer.train()

    #=================#
    # final evaluation
    #=================#

    if LOCAL_RANK == 0:
        trainer = mlutils.Trainer(model, dataset, device=device)
        callback.load(trainer)
        callback.fid = True
        callback(trainer, final=True)

    return

#======================================================================#
if __name__ == "__main__":

    #===============#
    # Arguments
    #===============#

    mlutils.set_seed(123)
    parser = argparse.ArgumentParser(description = 'FastDiffusion')

    parser.add_argument('--dataset', default='AFHQ', help='dataset', type=str)
    parser.add_argument('--data_class', default='cat', help='data class', type=str)
    parser.add_argument('--image_size', default=64, help='dataset', type=int)
    parser.add_argument('--train', action="store_true", help='train or eval')
    parser.add_argument('--case_dir', default='test', help='case_dir', type=str)
    parser.add_argument('--mode', default=0, help='FM (0) or SM (1)', type=int)
    parser.add_argument('--save_every', default=10, help='epochs', type=int)
    parser.add_argument('--noise_seed', default='out/noise.pt', type=str)

    args = parser.parse_args()

    #===============#
    # DATASET
    #===============#

    if args.dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])

        root = os.path.join(DATA_DIR, args.dataset)
        dataset = torchvision.datasets.CelebA(
            root=root, split='train', download=True, transform=transform)

    elif args.dataset == 'AFHQ':
        root = os.path.join(DATA_DIR, 'AFHQ', 'train')
        dataset = fastdiff.AFHQDataset(root, args.image_size, args.data_class)

    else:
        assert False

    #===============#
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    if DISTRIBUTED:
        mlutils.dist_setup()
        device = LOCAL_RANK
    else:
        device = mlutils.select_device()

    #===============#
    main(dataset, device, args)
    #===============#

    if DISTRIBUTED:
        mlutils.dist_finalize()
    #===============#

    exit()
#======================================================================#
#
