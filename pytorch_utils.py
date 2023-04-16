import numpy as np
from PIL import Image
from stimuli import make_colors_ring, make_gabors_ring
import torch
import torchvision
from torchvision import transforms
from typing import Any, Callable, List, Optional, Tuple
import webdataset as wds


def sample_angles(setsize, spacing="random", amin=0, amax=2 * np.pi):
    if spacing == "uniform":
        x = np.linspace(amin, amax, setsize + 1)[:-1]
        x += np.random.uniform(amin, amax)
    elif spacing == "random":
        x = np.random.uniform(amin, amax, size=setsize)
    else:
        raise NotImplementedError()
    return x


def attention_map(input_size, num_foci=0):
    """
    Choose attentional focal point in image and create map indicating its
    location. Randomly sample focal point(s) according to PMF that drops off
    linearly from center.
    """
    dist_from_center = np.abs(np.arange(input_size) - input_size // 2)
    z = np.zeros((input_size, input_size))
    if num_foci > 0:
        probs = -(dist_from_center - input_size / 2) / (input_size / 2)
        probs = probs / probs.sum()
        x0, y0 = np.random.choice(input_size, p=probs, size=2, replace=True)
        z[x0, y0] = 255
    return z.astype("uint8")


def attention_map_vwm_optimal(input_size, sqsize, xs, ys):
    """
    Place an attentional focal point on each item in the VWM display.
    """
    z = np.zeros((input_size, input_size))
    for x, y in zip(xs, ys):
        xi = x + int(round(sqsize / 2))  # Get center of square, not corner
        yi = y + int(round(sqsize / 2))
        z[xi, yi] = 255
    return z.astype("uint8")


class VWMDataset(torch.utils.data.Dataset):
    def __init__(self, setsize, N, img_size=64, angle_jitter=2 * np.pi,
            spatial_jitter=4, sqsize=2, eccentricity=0.2, blur=False,
            return_probe=True, delta=None, optimal_attention=False,
            num_foci=0, stim_type="colored_squares"):

        if not hasattr(setsize, "__len__"):
            self.setsize = [setsize]
        else:
            self.setsize = setsize
        if max(self.setsize) > 12:
            raise Exception()
        self.stim_type = stim_type
        self.img_size = img_size
        self.angle_jitter = angle_jitter
        self.spatial_jitter = spatial_jitter
        self.N = N
        self.sqsize = sqsize
        self.eccentricity = eccentricity
        self.blur = blur
        self.return_probe = return_probe
        self.num_foci = num_foci
        self.optimal_attention = optimal_attention
        self.delta = np.pi if delta is None else delta
        
    def __len__(self):
        return self.N

    def make_stimulus(self, spatial_jitter, angle_jitter, x=None, setsize=1):
        if self.stim_type == "colored_squares":
            if x is None:
                x = sample_angles(setsize)
            img, xjit, yjit, angle_jit, xs, ys = make_colors_ring(
                x,
                size=self.img_size,
                spatial_jitter=((0, 0), (self.spatial_jitter, self.spatial_jitter)),
                angle_jitter=angle_jitter,
                sqsize=self.sqsize,
                eccentricity=self.eccentricity,
                blur=self.blur,
            )
        elif self.stim_type == "gabors":
            if x is None:
                x = sample_angles(setsize, amin=-np.pi / 2, amax=np.pi / 2)
            img, xjit, yjit, angle_jit, xs, ys = make_gabors_ring(
                x,
                size=self.img_size,
                spatial_jitter=spatial_jitter,
                angle_jitter=angle_jitter,
                eccentricity=self.eccentricity,
            )
        else:
            raise NotImplementedError()
        return img, xjit, yjit, angle_jit, xs, ys, x

    def __getitem__(self, idx):
        # Randomly sample from possible set-sizes
        setsize = np.random.choice(self.setsize)
        img, xjit, yjit, angle_jit, xs, ys, x = self.make_stimulus(
            ((0, 0), (self.spatial_jitter, self.spatial_jitter)),
            (0, self.angle_jitter),
            setsize=setsize,
        )
        if self.optimal_attention:
            idx = np.random.choice(len(xs), size=self.num_foci, replace=False)
            att_map = transforms.ToTensor()(attention_map_vwm_optimal(
                    self.img_size, self.sqsize, xs[idx], ys[idx]
            ))
        else:
            att_map = transforms.ToTensor()(
                attention_map(self.img_size, num_foci=self.num_foci)
            )
        img = transforms.ToTensor()(img)
        coords = np.zeros((12, 2))  # Hardcoding 12 as maximum possible set-size
        coords[:len(xs)] = np.vstack([xs, ys]).T
        coords = torch.tensor(coords).float()
        item_vals = np.zeros(12)  # Hardcoding 12 as maximum possible set-size
        item_vals[:len(x)] = x
        item_vals = torch.tensor(item_vals).float()

        if self.return_probe:
            x1 = x.copy()
            i = np.random.choice(len(x))
            x1[i] += self.delta * np.random.choice([1, -1])
            probe_img = self.make_stimulus(
                ((xjit, yjit), (0, 0)),
                (angle_jit, 0),
                x=x1
            )[0]
            probe_img = transforms.ToTensor()(probe_img)
            return img, att_map, item_vals, coords, setsize, probe_img
        else:
            return img, att_map, item_vals, coords, setsize


def get_aperture_pair(img, crop_size, delta, input_size, pchange=1.):
    h, w = np.asarray(img.shape[1:])
    top = np.random.choice(h - crop_size)
    left = np.random.choice(w - crop_size)
    img1 = transforms.functional.crop(
        img, top, left, crop_size, crop_size
    )
    # With probability pchange, sample a probe that is shifted by delta.
    # Otherwise, just copy original aperture (i.e. a 'same' trial)
    delta_ = np.random.choice([0, delta], p=(1 - pchange, pchange))
    # Sample along ring of radius=delta until a valid location is drawn
    while True:
        theta = np.random.uniform(0, 2 * np.pi)
        x = left + int(np.round(np.cos(theta) * delta_))
        y = top + int(np.round(np.sin(theta) * delta_))
        if (
            x >= 0 and
            y >= 0 and
            x + crop_size < w and
            y + crop_size < h
        ):
            break
    img2 = transforms.functional.crop(img, y, x, crop_size, crop_size)
    coords = (top, left, y, x)
    img1 = transforms.Resize(input_size)(img1)
    img2 = transforms.Resize(input_size)(img2)
    return img1, img2, coords


def apply_natural_img_tranformations(image, attention, crop_size, input_size):
    if crop_size < 1:
        trans = transforms.Compose([transforms.ToTensor()])
        img = trans(image)
        return img
    else:
        trans = transforms.Compose([
            transforms.RandomCrop(crop_size, pad_if_needed=True),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        img = trans(image)
    return img


def make_attention_map(attention, input_size):
    if attention == "none":
        att_map = transforms.ToTensor()(
            np.zeros((input_size, input_size)).astype("uint8")
        )
    elif attention == "attend":
        att_map = transforms.ToTensor()(attention_map(input_size))
    else:
        raise NotImplementedError()
    return att_map


class CustomPlaces365(torchvision.datasets.Places365):
    def __init__(
            self,
            root: str,
            split: str = "train-standard",
            small: bool = False,
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = torchvision.datasets.folder.default_loader,
            attention: str = "none",
            crop_size: int = 64,
            input_size: int = 64,
            buffer_size: int = 100000000,
            delta: bool = 0,
            pchange: float = 1.,
        ) -> None:
            super().__init__(root)

            self.split = self._verify_split(split)
            self.small = small
            self.loader = loader

            self.classes, self.class_to_idx = self.load_categories(download)
            self.imgs, self.targets = self.load_file_list(download)

            self.crop_size = crop_size
            self.input_size = input_size
            self.attention = attention
            self.delta = delta
            self.pchange = pchange

            self.buffer = {}
            self.buffer_size = buffer_size

            if download:
                self.download_images()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # NOTE TO SELF: Was trying to speed up data loading by caching to RAM
        # but this didn't work very well and I switched to 'webdataset' package
        if index in self.buffer.keys():
            image = self.buffer[index]
        else:
            file, target = self.imgs[index]
            image = self.loader(file)
            if len(self.buffer) < self.buffer_size:
                self.buffer[index] = image

        if self.delta > 0:
            # Make change-detection stimulus pair with aperture design, where
            # the distance (in pixels) between target and probe is given by
            # delta.
            image = apply_natural_img_tranformations(
                image, self.attention, -1, None
            )
            img1, img2, coords = get_aperture_pair(
                image,
                self.crop_size,
                self.delta,
                self.input_size,
                pchange=self.pchange,
            )
            att_map = make_attention_map(self.attention, self.input_size)
            x = (img1, img2, att_map, att_map, coords)
        else:
            image = apply_natural_img_tranformations(
                image, self.attention, self.crop_size, self.input_size
            )
            att_map = make_attention_map(self.attention, self.input_size)
            x = (image, att_map)
        return x


class NaturalImgTrans():
    def __init__(self, attention, crop_size, input_size, delta=0, pchange=1.):
        self.attention = attention
        self.crop_size = crop_size
        self.input_size = input_size
        self.delta = delta
        self.pchange = pchange

    def __call__(self, x):
        if self.delta > 0:
            # Make change-detection stimulus pair with aperture design, where
            # the distance (in pixels) between target and probe is given by
            # delta.
            image = apply_natural_img_tranformations(
                x["jpg"], self.attention, -1, None
            )
            img1, img2, coords = get_aperture_pair(
                image,
                self.crop_size,
                self.delta,
                self.input_size,
                pchange=self.pchange,
            )
            att_map = make_attention_map(self.attention, self.input_size)
            x['jpg'] = img1
            x['jpg2'] = img2
            x['att_map'] = att_map
            x['att_map2'] = att_map
            x['change'] = coords[:2] == coords[2:]
            x['coords'] = coords
        else:
            image = apply_natural_img_tranformations(
                x["jpg"], self.attention, self.crop_size, self.input_size
            )
            att_map = make_attention_map(self.attention, self.input_size)
            x["jpg"] = image
            x["att_map"] = att_map

        # OLD CODE THAT DIDN'T HANDLE CHANGE-DETECTION TASK
        # img, att_map = apply_natural_img_tranformations(
        #     x["jpg"], self.attention, self.crop_size, self.input_size
        # )
        # x["jpg"] = img
        # x["att_map"] = att_map
        return x


def make_webdataset(
    url, attention, crop_size, input_size, batch_size=32, delta=0, pchange=1.
    ):
    trans_obj = NaturalImgTrans(
        attention, crop_size, input_size, delta=delta, pchange=pchange
    )
    dataset = wds.WebDataset(url).shuffle(1000).decode("pilrgb")
    if delta > 0:
        # Change-detection task
        dataset = dataset.map(trans_obj).to_tuple(
            "jpg", "jpg2", "att_map", "att_map2", "change", "coords"
        )
    else:
        dataset = dataset.map(trans_obj).to_tuple("jpg", "att_map")
    dataset = dataset.batched(batch_size)
    loader = wds.WebLoader(dataset, num_workers=4, batch_size=None)
    loader.dataset = dataset
    return loader

