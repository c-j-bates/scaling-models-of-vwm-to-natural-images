from networks import VAEConv
import numpy as np
import os
from PIL import Image
from pathlib import Path
from pytorch_utils import VWMDataset, CustomPlaces365, make_webdataset
import re
import torch
import torch.nn as nn
import torchvision.models as tv_models
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# INPUT_SIZE = 64
CROP_SIZE = 256
VWM_DATASETS = ["colored_squares", "gabors"]

TORCH_DATASET_DIR = os.environ.get(
    "TORCH_DATASET_DIR",
    "/n/gershman_lab/lab/cjbates/pytorch_datasets/"
    # "/n/holyscratch01/gershman_lab/Lab/"
)
TORCH_DATASET_DIR = Path(TORCH_DATASET_DIR).joinpath("Places365")
CHECKPOINT_DIR = os.environ.get(
    "TORCH_CKPT_DIR",
    "/n/gershman_lab/lab/cjbates/pytorch_checkpoints/"
)
CHECKPOINT_DIR = Path(CHECKPOINT_DIR).joinpath(
    "psychophysical_scaling"
)

NUM_WORKERS = 2  # Number of workers for pytorch loader

MODEL_DICT = {}


def make_checkpoint_path(model, epoch, ckpt_dir):
    load_path = Path(ckpt_dir).joinpath(
        f"{model.name}_epoch_{epoch}.pth"
    )
    return load_path


def save_checkpoint(model, optimizer, epoch, ckpt_dir):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    save_path = make_checkpoint_path(model, epoch, ckpt_dir)
    print(f"Saving checkpoint (epoch {epoch})")
    print(f"Save path: {save_path}")
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        save_path
    )


def load_checkpoint(ckpt_dir, model, optimizer, epoch):
    load_path = make_checkpoint_path(model, epoch, ckpt_dir)
    print(f"Loading checkpoint (epoch {epoch})")
    print(f"Load path: {load_path}")
    checkpoint = torch.load(load_path, map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer_to(optimizer, device)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def make_dataloader(dataset_dir=None, batch_size=32, split="train-standard",
        download=False, attention="none", optimal_attention=False, delta=0,
        ss_min=1, ss_max=12, num_foci=1, small=False, ds_name="places365",
        pchange=1., ds_size=500000, input_size=256):

    if ds_name in VWM_DATASETS:
        dataset = VWMDataset(
            np.arange(max(ss_min, num_foci), ss_max + 1),
            ds_size,
            stim_type=ds_name,
            img_size=input_size,
            sqsize=4,
            eccentricity=0.3,
            spatial_jitter=4,
            return_probe=False,
            num_foci=num_foci,
            optimal_attention=optimal_attention,
        )
    elif ds_name == "places365":
        if split == "val":
            # Not worried about loading speed for validation set, so load
            # jpg files directly
            dataset = CustomPlaces365(
                dataset_dir,
                download=download,
                split=split,
                small=small,
                crop_size=CROP_SIZE,
                input_size=input_size,
                attention=attention,
                delta=delta,
                pchange=pchange,
            )
        elif split == "train-standard":
            if small:
                # Still need to create tar files for 256x256 images...
                raise NotImplementedError()
            else:
                # Use Webdataset package and tar files
                url = str(dataset_dir) + "/data_large_standard/tar/dataset_shard{0..180}.tar"
                loader = make_webdataset(
                    url,
                    attention,
                    CROP_SIZE,
                    input_size,
                    batch_size=batch_size,
                    delta=delta,
                    pchange=pchange,
                )
                # Use special webdataset loader for efficiency
                return loader
    else:
        raise NotImplementedError()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    return loader


class TrainloaderScheduled():
    def __init__(self, args):
        self.dataset_dir = Path(args.dataset_dir)
        self.download = args.download
        self.batch_schedule = args.batch_schedule
        if self.batch_schedule is not None:
            self.batch_schedule = eval(self.batch_schedule)
        self.default_batch_size = args.batch_size
        self.input_size = args.input_size
        self.split = args.split
        self.small = args.small
        self.dataset = args.dataset_vae
        self.delta = args.delta
        self.pchange = args.pchange
        self.attention = "attend" if args.attend else "none"
        self.num_foci = args.num_foci
        self.optimal_attention = args.optimal_attention
        self.ds_size = args.ds_size
        self.ss_min = args.ss_min
        self.ss_max = args.ss_max

    def __getitem__(self, epoch):
        """
        Create and return a pytorch dataloader that satisfies schedule for
        give epoch
        """
        if self.batch_schedule is None:
            batch_size = self.default_batch_size
        else:
            for i in range(len(self.batch_schedule)):
                if epoch >= self.batch_schedule[i][0]:
                    batch_size = self.batch_schedule[i][1]
        print(f"Setting batch size to {batch_size}")
        loader = make_dataloader(
            self.dataset_dir,
            ds_name=self.dataset,
            input_size=self.input_size,
            ds_size=self.ds_size,
            download=self.download,
            split=self.split,
            small=self.small,
            ss_min=self.ss_min,
            ss_max=self.ss_max,
            batch_size=batch_size,
            attention=self.attention,
            num_foci=self.num_foci,
            optimal_attention=self.optimal_attention,
            delta=self.delta,
            pchange=self.pchange,
        )
        loader.stim_type = self.dataset
        return loader


def make_and_load(args, basename="vae", opt_type="adam"):
    if args.pixel_only:
        loss_str = "pixel_mse"
    else:
        loss_str = f"layer_{args.layer}_wpx_{args.pixel_weight}"
    beta_str = f"beta_{args.beta}"
    dataset_str = args.dataset_vae
    if args.dataset_vae in VWM_DATASETS and args.optimal_attention:
        dataset_str += "_optatt"
    attention_str = f"_no_attend" if not args.attend else f"_foci_{args.num_foci}"
    name = f"{basename}{args.input_size}_{dataset_str}_{loss_str}_{beta_str}{attention_str}"
    print('Model name: ', name)
    model = VAEConv(input_size=args.input_size).to(device)
    start_epoch = args.start_epoch
    model.name = name
    if opt_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            momentum=0.9
        )
    elif opt_type == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            momentum=0.9
        )
    elif opt_type == "radam":
        optimizer = torch.optim.RAdam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError()
    if start_epoch >= 0:
        model, optimizer = load_checkpoint(
            args.ckpt_dir,
            model,
            optimizer,
            start_epoch,
            task=args.task,
            delta=args.delta,
        )
    return model, optimizer


def dnn_embed(x, layer_idx=9, model_name="vgg19"):
    if model_name not in MODEL_DICT.keys():
        net = getattr(tv_models, model_name)(pretrained=True).to(device)
        net = net.eval()
        MODEL_DICT[model_name] = net
    else:
        net = MODEL_DICT[model_name]
    for i in range(layer_idx + 1):
        x = net.features[i](x)
    return x


def downsample_att_map(att_map, size):
    """
    Assumes square image
    """
    size0 = att_map.shape[-1]
    factor = size0 / size
    bins = torch.tensor([factor * i for i in range(1, size + 1)]).to(device)
    att_map1 = []
    for i in range(len(att_map)):
        xy = torch.where(torch.squeeze(att_map[i]))
        m = torch.zeros((size, size)).to(device)
        num_focal_points = len(xy[0])
        for idx in range(num_focal_points):
            x0 = xy[0][idx]
            y0 = xy[1][idx]
            ii = torch.bucketize(x0, bins, right=True)
            jj = torch.bucketize(y0, bins, right=True)
            m[ii, jj] = 1.
        att_map1.append(m)
    att_map1 = torch.stack(att_map1).unsqueeze(1)
    return att_map1


def get_att_mask(att_map, scale=0.001):
    mask_batch = []
    for i in range(len(att_map)):
        xy = torch.where(torch.squeeze(att_map[i]))
        num_focal_points = len(xy[0])
        mask = torch.zeros_like(att_map[i])
        for idx in range(num_focal_points):
            x0 = xy[0][idx]
            y0 = xy[1][idx]
            xcoords = torch.arange(att_map[i].shape[1]).to(device) - x0
            ycoords = torch.arange(att_map[i].shape[1]).to(device) - y0
            xx, yy = torch.meshgrid(xcoords, ycoords, indexing="ij")
            radius = torch.sqrt(xx ** 2 + yy ** 2)
            mask += torch.exp(-radius ** 2 * scale).unsqueeze(0)
        mask = mask / mask.sum() * mask.shape.numel()
        # DEBUG #####
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # im = ax.imshow(mask[0], cmap="gray")
        # plt.colorbar(im)
        # plt.show()
        # ##########
        mask_batch.append(mask)
    mask_batch = torch.stack(mask_batch)
    return mask_batch


def recon_loss_func(x, pred, att_map, pixel_only=False, layer_idx=9, w_px=1.):
    if att_map is None or not att_map.sum():
        loss = nn.MSELoss(reduction="sum")(x, pred) * w_px
        if not pixel_only:
            x_perc = dnn_embed(x, layer_idx=layer_idx)
            pred_perc = dnn_embed(pred, layer_idx=layer_idx)
            loss += nn.MSELoss(reduction="sum")(x_perc, pred_perc)
    else:
        loss = nn.MSELoss(reduction="none")(x, pred) * w_px
        mask = get_att_mask(att_map, scale=0.02)
        loss = (loss * mask).sum()
        if not pixel_only:
            x_perc = dnn_embed(x, layer_idx=layer_idx)
            pred_perc = dnn_embed(pred, layer_idx=layer_idx)
            att_map_perc = downsample_att_map(att_map, x_perc.shape[-1])
            mask_perc = get_att_mask(att_map_perc, scale=0.1)
            loss1 = nn.MSELoss(reduction="none")(x_perc, pred_perc) * mask_perc
            loss += loss1.sum()
    loss = loss / len(x)
    return loss


def pick_report_location(item_coords, idxs, img_size):
    att_map = torch.zeros((len(item_coords), img_size, img_size)).float()
    for i, xys in enumerate(item_coords):
        ix, iy = xys[idxs[i]]
        att_map[i][ix.long(), iy.long()] = 1.
    att_map.unsqueeze(1)
    att_map = att_map.to(device)
    return att_map


def circular_loss_func(Y, pred):
    loss = nn.MSELoss(reduction="mean")(torch.cos(Y), pred[:, 0])
    loss += nn.MSELoss(reduction="mean")(torch.sin(Y), pred[:, 1])
    return loss


def train_mlp(
    model, optimizer, loader, start_epoch=-1, end_epoch=100
):
    model.to(device)
    model = model.train()

    for epoch in (pbar := tqdm(range(start_epoch + 1, end_epoch), dynamic_ncols=True)):
        running_loss = 0
        for batch, data in enumerate(loader):
            (X, y) = data
            X = X.to(device)
            y = y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = nn.MSELoss()(y, pred)

            running_loss += loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 200 == 0:
                pbar.set_description(f'Running loss: {running_loss / (batch + 1)}')
    return model


def train_vae_mlp(
    model,
    optimizer,
    trainloader,
    end_epoch=1,
    beta=1.,
    ckpt_dir=None,
    start_epoch=-1,
):
    """
    Difference with `train_vae_conv` is that there is no attention map input
    and reconstruction loss has no extra bells and whistles
    """
    model.to(device)
    model = model.train()

    for epoch in range(start_epoch + 1, end_epoch):
        running_loss = 0
        running_recon_loss = 0
        for batch, data in enumerate(trainloader):
            X = data[0]
            X = X.to(device)

            # Compute prediction error
            z_mu, z_samp, log_var, pred = model(X)
            recon_loss = nn.MSELoss(reduction="sum")(X, pred) / len(X)
            kl_loss = torch.mean(
                0.5 * torch.sum(
                    torch.exp(log_var) + z_mu ** 2 - 1. - log_var, 1
                )
            )
            loss = recon_loss + beta * kl_loss
            running_loss += loss
            running_recon_loss += recon_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"Epoch {epoch}: loss: {loss.item():>7f}, {recon_loss.item():>7f}")
            if batch % 1000 == 0:
                # save_checkpoint(model, optimizer, epoch, ckpt_dir)
                ave_loss = running_loss / (batch + 1)
                ave_recon_loss = running_recon_loss / (batch + 1)
                print(f"Epoch {epoch}: Running mean loss: {ave_loss:>7f}, {ave_recon_loss:>7f}")

        ave_loss = running_loss / (batch + 1)
        ave_recon_loss = running_recon_loss / (batch + 1)
        print(f"Mean loss for epoch: {ave_loss:>7f}, {ave_recon_loss:>7f}")
        # save_checkpoint(model, optimizer, epoch, ckpt_dir)
    return model, (ave_loss, ave_recon_loss)


def train_vae_conv(
    model,
    optimizer,
    trainloader,
    end_epoch=1,
    beta=1.,
    ckpt_dir=None,
    start_epoch=-1,
    w_px=1,
    layer_idx=9,
    pixel_only=True,
    patience=10
):

    model.to(device)
    model = model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience
    )

    for epoch in range(start_epoch + 1, end_epoch):
        running_loss = 0
        running_recon_loss = 0
        for batch, data in enumerate(trainloader[epoch]):
            X, att_map = data[:2]
            X = X.to(device)
            att_map = att_map.to(device)

            # Compute prediction error
            z_mu, z_samp, log_var, pred = model(X, att_map)
            recon_loss = recon_loss_func(
                X,
                pred,
                att_map,
                pixel_only=pixel_only,
                layer_idx=layer_idx,
                w_px=w_px,
            )
            kl_loss = torch.mean(
                0.5 * torch.sum(
                    torch.exp(log_var) + z_mu ** 2 - 1. - log_var, 1
                )
            )
            loss = recon_loss + beta * kl_loss
            running_loss += loss
            running_recon_loss += recon_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"loss: {loss.item():>7f}, {recon_loss.item():>7f}")
            if batch % 1000 == 0:
                save_checkpoint(model, optimizer, epoch, ckpt_dir)
                ave_loss = running_loss / (batch + 1)
                ave_recon_loss = running_recon_loss / (batch + 1)
                print(f"Running mean loss: {ave_loss:>7f}, {ave_recon_loss:>7f}")

        ave_loss = running_loss / (batch + 1)
        ave_recon_loss = running_recon_loss / (batch + 1)
        print(f"Mean loss for epoch: {ave_loss:>7f}, {ave_recon_loss:>7f}")
        save_checkpoint(model, optimizer, epoch, ckpt_dir)
        scheduler.step(ave_loss)
    return model


def make_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Train beta-VAE"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=TORCH_DATASET_DIR,
        help="Location of pytorch downloaded datasets."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=CHECKPOINT_DIR,
        help="Directory for saving and loading checkpoints."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train-standard",
        help="Which dataset split to use."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.,
        help="Value of beta in KL loss."
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Width in pixels of input images."
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=-1,
        help="Specify epoch for resuming VAE training"
    )
    parser.add_argument(
        "--end-epoch",
        type=int,
        default=500,
        help="Specify epoch for ending VAE training"
    )
    parser.add_argument(
        "--start-epoch-decoder",
        type=int,
        default=-1,
        help="Specify epoch for resuming decoder training"
    )
    parser.add_argument(
        "--end-epoch-decoder",
        type=int,
        default=500,
        help="Specify epoch for ending decoder training"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=36,
        help="Index of layer to use from pretrained classifier in perceptual "
        "embedding. (Max pool layers occur at the multiples of 9, up to 36.)"
    )
    parser.add_argument(
        "--attend",
        action="store_true",
        help="Implement attentional loss, where attention is implemented as a"
        " higher penalty on particular areas of the image."
    )
    parser.add_argument(
        "--pixel-only",
        action="store_true",
        help="Only use pixel error in loss function."
    )
    parser.add_argument(
        "--pixel-weight",
        type=float,
        default=1.,
        help="Weight by which to multiply pixel MSE loss term."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.,
        help="Weight decay in optimizer (L2 regularization)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size."
    )
    parser.add_argument(
        "--batch-schedule",
        type=str,
        default=None,
        help="Batch size schedule (e.g. increase batch size over time). Input"
        " should be a list of two-tuples, specifying (batch-size, epoch) "
        "pairs. Ex.: [(0, 32), (10, 64), (20, 128)]. When value is None "
        "(default), this argument is ignored and superceded by --batch-size."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download pytorch dataset. (Use if it hasn't been downloaded"
        " before.)"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Train on small version of Places365."
    )
    parser.add_argument(
        "--dataset-vae",
        type=str,
        default="places365",
        help="Which dataset to train VAE on."
    )
    parser.add_argument(
        "--ds-size",
        type=int,
        default=500000,
        help="Number of samples that count as an epoch in VWM datasets."
    )
    parser.add_argument(
        "--ss-min",
        type=int,
        default=1,
        help="Minimum set-size to sample in VWM datasets."
    )
    parser.add_argument(
        "--ss-max",
        type=int,
        default=8,
        help="Maximum set-size to sample in VWM datasets."
    )
    # DEPRECATED
    parser.add_argument(
        "--optimal-attention",
        action="store_true",
        help="If training on VWM stimuli, enforce optimal, multi-focal "
        "attentional allocation."
    )
    # DEPRECATED
    parser.add_argument(
        "--num-foci",
        type=int,
        default=0,
        help="Number of spotlights to divide attention over."
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    trainloader = TrainloaderScheduled(args)
    model, optimizer = make_and_load(args)
    train_vae_conv(
        model,
        optimizer,
        trainloader,
        end_epoch=args.end_epoch,
        beta=args.beta,
        ckpt_dir=args.ckpt_dir,
        start_epoch=args.start_epoch,
        layer_idx=args.layer,
        pixel_only=args.pixel_only,
        w_px=args.pixel_weight,
    )
