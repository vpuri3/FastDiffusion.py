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

DATADIR = 'data/'

#======================================================================#
def main(dataset, device, out_dir, res_dir, image_size, train=True):
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    case_name = 'flow_matching'
    out_name = os.path.join(out_dir, case_name)
    res_name = os.path.join(res_dir, case_name)
    model_file  = out_name + ".pth"

    #=================#
    # DATA:
    #=================#
    dataset = dataset

    #=================#
    # MODEL
    #=================#
    # model = fastdiff.unet()
    model = fastdiff.DiT_S_2(input_size=image_size, in_channels=3,)

    #=================#
    # TRAIN
    #=================#
    if train:
        kw = dict(
            Opt='Adam', lr=1e-4, nepochs=100, weight_decay=0e-5,
            _batch_size=32, _batch_size_=500,
            batch_lossfun=fastdiff.loss_flow_matching,
            device=device, stats_every=5,
        )

        trainer = mlutils.Trainer(model, dataset, **kw)
        # trainer.add_callback('epoch_start', xxx)
        trainer.train()

        if LOCAL_RANK==0:
            torch.save(model.to("cpu").state_dict(), model_file)

    #=================#
    # ANALYSIS
    #=================#
    # if LOCAL_RANK == 0:
    #     model.eval()
    #     model_state = torch.load(model_file, weights_only=True, map_location='cpu')
    #     model.load_state_dict(model_state)
    #     model.to(device)
    #
    #     # model(x)

    return

#======================================================================#
if __name__ == "__main__":

    #===============#
    # Choose dataset
    #===============#
    mlutils.set_seed(123)
    parser = argparse.ArgumentParser(description = 'FastDiffusion')
    parser.add_argument('--dataset', default='AFHQ', help='dataset', type=str)
    parser.add_argument('--data_class', default='cat', help='data class', type=str)
    parser.add_argument('--image_size', default=32, help='dataset', type=int)
    args = parser.parse_args()

    # Transform
    if args.dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        root = os.path.join(DATADIR, args.dataset)
        dataset = torchvision.datasets.CelebA(
            root=root, split='train', download=True, transform=transform)

    elif args.dataset == 'AFHQ':
        root = os.path.join(DATADIR, 'AFHQ', 'train')
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
    out_dir = "./out/"
    res_dir = "./res/"

    if LOCAL_RANK == 0:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)

    #===============#
    main(dataset, device, out_dir, res_dir, args.image_size, train=True)

    #===============#
    if DISTRIBUTED:
        mlutils.dist_finalize()
    #===============#

    pass
#======================================================================#
#
