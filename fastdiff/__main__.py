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
    # DIFFUSION TYPE
    #=================#

    if args.case_type == 0: # flow matching
        sample_fun = fastdiff.sample_flow_matching
        batch_lossfun = fastdiff.loss_flow_matching
    else: # shortcut
        sample_fun = fastdiff.sample_shortcut
        batch_lossfun = fastdiff.loss_shortcut

    #=================#
    # MODEL
    #=================#

    # model = fastdiff.unet()
    # model = fastdiff.DiT_S_2(input_size=args.image_size, in_channels=3,)
    model = fastdiff.DiT_B_2(input_size=args.image_size, in_channels=3,)

    #=================#
    # TRAIN
    #=================#

    visualizer = fastdiff.Visualizer(
        sample_fun, out_dir, args.image_size,
        args.save_every, dataset.root, fid=False
    )

    def callback(trainer: mlutils.Trainer):
        if LOCAL_RANK == 0:
            visualizer(trainer)
        return

    if args.train:
        kw = dict(
            Opt='Adam', lr=1e-4, nepochs=100, weight_decay=0e-5,
            _batch_size=16, _batch_size_=500,
            batch_lossfun=batch_lossfun,
            device=device, stats_every=args.save_every,
        )

        trainer = mlutils.Trainer(model, dataset, **kw)
        trainer.add_callback('epoch_end', callback)
        trainer.train()

        if LOCAL_RANK==0:
            torch.save(model.to("cpu").state_dict(), model_file)

    #=================#
    # final evaluation
    #=================#

    if LOCAL_RANK == 0:
        model.eval()
        model_state = torch.load(model_file, weights_only=True, map_location='cpu')
        model.load_state_dict(model_state)
        model.to(device)

        # trainer = mlutils.Trainer(model, dataset)
        # visualizer(trainer)

    return

#======================================================================#
if __name__ == "__main__":

    #===============#
    # Arguments
    #===============#

    mlutils.set_seed(123)
    parser = argparse.ArgumentParser(description = 'FastDiffusion')

    # dataset
    parser.add_argument('--dataset', default='AFHQ', help='dataset', type=str)
    parser.add_argument('--data_class', default='cat', help='data class', type=str)
    parser.add_argument('--image_size', default=32, help='dataset', type=int)

    # parser.add_argument('--fid_final', default=True, help='compute_fid', type=bool)
    parser.add_argument('--train', default=True, help='train or eval', type=bool)
    parser.add_argument('--case_dir', default='test', help='case_dir', type=str)
    parser.add_argument('--case_type', default=0, help='FM or SM', type=int)
    parser.add_argument('--save_every', default=5, help='epochs', type=int)

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

    pass
#======================================================================#
#
