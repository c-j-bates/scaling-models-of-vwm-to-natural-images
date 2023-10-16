from argparse import Namespace
from itertools import product
import json
# import matplotlib.pyplot as plt
import multiprocessing
from networks import VAEMLP
import numpy as np
from omnidata.torch.modules.midas.dpt_depth import DPTDepthModel
import os
from p_tqdm import p_map
import pandas as pd
from pathlib import Path
import PIL
from PIL import Image
import platform
import psutil
from scipy.stats import spearmanr
from scipy.spatial import distance
from sklearn.covariance import GraphicalLassoCV
from sklearn.decomposition import PCA, IncrementalPCA, NMF
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from stimuli import (
    make_colors_ring, make_gabors_ring, rgb_from_angle, make_colored_lines_ring
)
import sys
from torch.cuda.amp import autocast
import torch
import torchvision
import torchvision.models as tv_models
from torchvision import transforms
from train import (
    train_vae_mlp,
    make_and_load,
    TORCH_DATASET_DIR,
    CHECKPOINT_DIR,
)
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available() and platform.system() == 'Linux':
    torch.set_num_threads(multiprocessing.cpu_count())
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
scene_wheel_imgs = {}
wheels_pth = Path('scene_wheels_mack_lab_osf').joinpath(
    'scene_wheel_images', 'sceneWheel_images_webp'
)
spth = os.environ.get('DATA_STORAGE')
if spth is not None:
    DATA_STORAGE = Path(spth).joinpath('psychophysical_scaling')
else:
    DATA_STORAGE = Path('./')
print(f'Data storage path: {DATA_STORAGE}')
MODEL_OUT_PATH = DATA_STORAGE.joinpath('data_tcc')
HUGGING_CACHE = os.environ.get('HUGGINGFACE_CACHE')
CLIP_CACHE = os.environ.get('CLIP_CACHE')
MODEL_DICT = {}
TCC_SAMPLE = {}


def cossim(x, y):
    """
    x is vector
    y is vector or batched vector (matrix)
    """
    d = np.matmul(x, y.T)
    d = d / np.linalg.norm(x, axis=-1)
    d = d / np.linalg.norm(y, axis=-1)
    return d


def cossim_torch(x, y, dev=None):
    """
    Matmul works differently in torch than numpy. It infers presence of a batch
    dimension. Here, y is assumed to be possibly batched, but x should not be.
    """
    if dev is None:
        dev = device
    else:
        dev = 'cpu'
    x = torch.tensor(x).to(dev)
    y = torch.tensor(y).to(dev)
    d = torch.matmul(y, x)
    d = d / torch.linalg.norm(x, axis=-1)
    d = d / torch.linalg.norm(y, axis=-1)
    return d


def standardize_angles(x, deg=False):
    """
    Standardize angles by putting them in range (-180, 180] deg
    """
    if deg:
        val = 180
    else:
        val = np.pi
    x = (x % (val * 2) + val * 2) % (val * 2)
    if not hasattr(x, '__len__'):
        if x > val:
            x -= 2 * val
    else:
        x = np.array(x)
        x[x > val] = x[x > val] - 2 * val
    return x


def load_wheel_imgs(size=224):
    radius_vals = [2, 4, 8, 16, 32]
    wheels = {}
    for i in range(5):
        pth = wheels_pth.joinpath(f"Wheel0{i + 1}")
        wheels[i + 1] = {}
        for r in radius_vals:
            pth1 = pth.joinpath(f'wheel0{i + 1}_r{str(r).zfill(2)}')
            wheels[i + 1][r] = [
                transforms.Resize(size)(
                    transforms.ToTensor()(
                        Image.open(pth1.joinpath(f'{str(j).zfill(6)}.webp'))
                    ),
                )
                for j in range(360)
            ]
    return wheels


def kl_divergence(mu_p, mu_q, log_var_p, log_var_q):
    """
    Multivariate gaussian KL divergence (diagonal covariance)
    """
    mu_p = torch.tensor(mu_p).to(device)
    mu_q = torch.tensor(mu_q).to(device)
    log_var_p = torch.tensor(log_var_p).to(device)
    log_var_q = torch.tensor(log_var_q).to(device)
    x = 0.5 * torch.sum(
        torch.exp(log_var_p - log_var_q) - 1. +
        (mu_p - mu_q) ** 2 / torch.exp(log_var_q) + log_var_q - log_var_p
    )
    return x


def count_model_params(name):
    print(name)
    if name.startswith('clip'):
        model = load_clip_model(name.split('_')[-1])[0]
    elif name.startswith('convnext'):
        model = load_convnext_model(name)[0]
    elif name.startswith('harmonized'):
        model = load_harmonization_model(name)
    elif name in ['vgg19', 'resnet50']:
        model = getattr(tv_models, name)()
    else:
        raise NotImplementedError()
    if 'harmonized' in name:
        # Keras
        total_params = model.count_params()
    else:
        # PyTorch
        total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    return total_params


def get_torchvision_features(x, layer_idx=9, arch="vgg19"):
    def _vgg19(x):
        with torch.no_grad():
            for i in range(layer_idx + 1):
                x = net.features[i](x)
        return x.float()

    def _resnet50(x):
        """
        Note: Since this is a small enough model, I'm opting to just run the
        entire network, rather than mess with the forward function to do a
        partial forward pass correctly. I get hooked activations from whichever
        layer is specified.
        """
        def get_activation(name):
            def hook(model, input, output):
                if type(output) is tuple:
                    output = output[0]
                hooked_activations[name] = output.detach()
            return hook

        modules = list(net._modules.values())
        layers_per_module = [
            1 if not hasattr(m, '__len__') else len(m) for m in modules
        ]
        num_layers = sum(layers_per_module)
        cumsum = np.cumsum(layers_per_module)
        cumsum_padded = np.append(0, cumsum)
        mod_idx = np.digitize(layer_idx, bins=cumsum)
        hooked_activations = {}  # Repository for hooked activations
        if not hasattr(modules[mod_idx], '__len__'):
            handle = modules[mod_idx].register_forward_hook(get_activation('act'))
        else:
            i = layer_idx - cumsum_padded[mod_idx]
            handle = modules[mod_idx][i].conv3.register_forward_hook(get_activation('act'))
        with torch.no_grad():
            net(x)
            x = hooked_activations['act'].detach().cpu()
            handle.remove()
        return x

    if arch not in MODEL_DICT.keys():
        net = getattr(tv_models, arch)(weights='DEFAULT').to(device)
        net = net.eval()
        MODEL_DICT[arch] = net
    else:
        net = MODEL_DICT[arch]
    x = transforms.Resize((224, 224))(x)
    x = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(x)
    if arch == 'vgg19':
        x = _vgg19(x)
    elif arch == 'resnet50':
        x = _resnet50(x)
    else:
        raise NotImplementedError()
    return x


def load_harmonization_model(arch):
    if arch == 'harmonized_RN50':
        from harmonization.models import load_ResNet50
        net = load_ResNet50()
    elif arch == 'harmonized_ViT_B16':
        from harmonization.models import load_ViT_B16
        net = load_ViT_B16()
    else:
        raise NotImplementedError()
    return net


def get_harmonization_features(x, layer_idx=9, arch="harmonized_RN50"):
    """
    From: https://serre-lab.github.io/Harmonization/models/
    """
    from keras import Model

    def _resnet50(x):
        layers = [
            l.name for i, l in enumerate(net.layers)
            # First two layers are input and padding, so skip and define 0 to
            # start at 3rd layer
            if l.name.startswith('add') or 1 < i < 6 or 'global_average_pooling2d' in l.__str__()
        ]
        x = Model(
            inputs=net.input,
            outputs=net.get_layer(layers[layer_idx]).output
        )(x).numpy()
        return x

    def _vit(x):
        raise NotImplementedError()
        # layers = [
        #     l.name for l in net.layers
        #     if l.name.startswith('activation') or l.name == 'avg_pool'
        # ]
        # x = Model(
        #     inputs=net.input,
        #     outputs=net.get_layer(layers[layer_idx]).output
        # )(x).numpy()
        return x

    x = transforms.Resize((224, 224))(x)
    x = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(x)
    x = np.moveaxis(x.detach().cpu().numpy(), 1, -1)  # Keras accepts numpy inputs
    if arch not in MODEL_DICT.keys():
        net = load_harmonization_model(arch)
        MODEL_DICT[arch] = net
    else:
        net = MODEL_DICT[arch]
    if arch == 'harmonized_RN50':
        x = _resnet50(x)
    elif arch == 'harmonized_ViT_B16':
        x = _vit(x)
    else:
        raise NotImplementedError()
    return x


def load_places_cnn(arch='resnet50'):
    """
    Load pytorch models according to: https://github.com/CSAILVision/places365
    """
    # Models use pytorch's model zoo, but with custom weights downloaded from
    # CSAIL repo
    model_file = f'{arch}_places365.pth.tar'
    if arch not in MODEL_DICT.keys():
        net = getattr(tv_models, arch)(num_classes=365).to(device)
        checkpoint = torch.load(
            f'places_cnn_models/{model_file}',
            map_location=lambda storage,
            loc: storage
        )
        # Load pretrained weights
        state_dict = {
            str.replace(k,'module.',''): v
            for k,v in checkpoint['state_dict'].items()
        }
        net.load_state_dict(state_dict)
        net = net.eval()
        MODEL_DICT[arch] = net
    else:
        net = MODEL_DICT[arch]

    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return net, trans


def get_places_cnn_features(X, arch='resnet18', layer_idx=1):
    """
    Extract features from pretrained places-CNN networks
    (https://github.com/CSAILVision/places365)

    Networks were trained to classify scenes and are based on various resnet
    architectures.
    """
    def _resnet18():
        # There are three modules prior to the ones that we care about:
        # conv1, bn1, and relu1. These are followed by torch.Sequential
        # modules with naming convention 'layer{i}'. For simplicity, we
        # just consider the outputs of each of the 'layer{i}' modules.
        y = []
        with torch.no_grad():
            for x in loader:
                x = trans(x[0])
                for i in range(layer_idx + 4):
                    x = modules[i](x)
                y.append(x.detach())
        return y

    def _resnet50():
        # There are four modules prior to the ones that we care about:
        # conv1, bn1, relu, and maxpool. These are followed by torch.Sequential
        # modules with naming convention 'layer{i}'. For simplicity, we
        # just consider the outputs of each of the 'layer{i}' modules.
        y = []
        with torch.no_grad():
            for x in loader:
                x = trans(x[0])
                for i in range(layer_idx + 5):
                    x = modules[i](x)
                y.append(x.detach())
        return y

    model_name = 'places_cnn_' + arch
    if model_name not in MODEL_DICT.keys():
        net, trans = load_places_cnn(arch)
        net = net.eval()
        MODEL_DICT[model_name] = (net, trans)
    else:
        net, trans = MODEL_DICT[model_name]

    ds = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(ds, batch_size=32)
    modules = [mod for mod in net._modules.values()]
    if arch == 'resnet18':
        y = _resnet18()
    elif arch == 'resnet50':
        y = _resnet50()
    else:
        raise NotImplementedError()
    y = torch.stack(y).float()
    return y


def load_omnidata_model(task):
    if task == 'depth':
        image_size = 384
        pretrained_weights_path = Path('omnidata').joinpath(
            'torch',
            'pretrained_models',
            'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
        )
        model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(image_size),
            # transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)])
    return model, trans_totensor


def get_omni_features(X, layer_idx=-1, task="depth"):
    """
    Extract features from pretrained omnidata networks (see here:
    https://docs.omnidata.vision/pretrained.html#Download-pretrained-models)

    Task is either 'depth' (depth estimation) or 'normal' (surface normals
    estimation). Both consist of a Vision Transformer backbone (
    https://openaccess.thecvf.com/content/ICCV2021/papers/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.pdf)

    The models are subdivided into 'pretrained' and 'scratch' modules, where
    the former seems to correspond to the ViT (plus some fine tuning?) and
    scratch is task-specific. We take features from the scratch module.
    """
    model_name = 'omnidata_' + task
    if model_name not in MODEL_DICT.keys():
        net, trans = load_omnidata_model(task)
        MODEL_DICT[model_name] = (net.eval(), trans)
    else:
        net, trans = MODEL_DICT[model_name]

    def get_activation(name):
        def hook(model, input, output):
            hooked_activations[name] = output.detach()
        return hook

    hooked_activations = {}  # Repository for collected activations
    handle = net.scratch.output_conv[layer_idx].register_forward_hook(
        get_activation(f"layer_{layer_idx}")
    )
    ds = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    y = []
    with torch.no_grad():
        for x in loader:
            net(trans(x[0]))
            y.append(hooked_activations[f"layer_{layer_idx}"])
        y = torch.stack(y).float()
    handle.remove()
    return y


def load_clip_model(arch='RN50', finetuned=False):
    """
    Vision model from CLIP
    https://arxiv.org/pdf/2103.00020.pdf
    https://github.com/openai/CLIP

    Available architectures:
    'RN50',
    'RN101',
    'RN50x4',
    'RN50x16',
    'RN50x64',
    'ViT-B/32',
    'ViT-B/16',
    'ViT-L/14',
    'ViT-L/14@336px'
    """
    import clip
    if arch == 'ViT-B32':
        arch = 'ViT-B/32'
    elif arch == 'ViT-B16':
        arch = 'ViT-B/16'
    elif arch == 'ViT-L14':
        arch = 'ViT-L/14'
    elif arch == 'ViT-L14@336px':
        arch = 'ViT-L/14@336px'    
    model, preprocess = clip.load(arch, download_root=CLIP_CACHE)
    if finetuned:
        # Overwrite with VWM fine-tuned checkpoint
        pth = Path(f'{CHECKPOINT_DIR}').joinpath(
            f"clip_{arch.replace('/', '')}_finetune.pth"
        )
        checkpoint = torch.load(
            pth,
            map_location=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded clip from {pth}')
    return model, preprocess


@autocast()
def get_clip_features(X, layer_idx=14, arch='RN50', finetuned=False):

    def _resnet():
        def get_activation(name):
            def hook(model, input, output):
                if type(output) is tuple:
                    output = output[0]
                hooked_activations[name] = output.detach()
            return hook

        y = []
        modules = list(net._modules.values())
        layers_per_module = [
            1 if not hasattr(m, '__len__') else len(m) for m in modules
        ]
        num_layers = sum(layers_per_module)
        cumsum = np.cumsum(layers_per_module)
        cumsum_padded = np.append(0, cumsum)
        mod_idx = np.digitize(layer_idx, bins=cumsum)
        # with autocast():
        with torch.no_grad():
            for x in loader:
                x = torch.stack([
                    preprocess(transforms.ToPILImage()(x_[0])) for x_ in x
                ]).to(device)
                # Forward up to penultimate module
                for i in range(mod_idx):
                    x = modules[i](x)
                # Forward layer-by-layer within module that contains layer given by
                # layer_idx
                if not hasattr(modules[mod_idx], '__len__'):
                    x = modules[mod_idx](x)
                else:
                    hooked_activations = {}  # Repository for hooked activations
                    handles = [
                        # Take conv3 layer in block, rather than ReLU output
                        mod.conv3.register_forward_hook(get_activation(i))
                        for i, mod in enumerate(modules[mod_idx])
                    ]
                    for i in range(layer_idx - cumsum_padded[mod_idx] + 1):
                        x = modules[mod_idx][i](x)
                    x = hooked_activations[i]  # Take hooked values instead of output
                    [h.remove() for h in handles]
                y.append(x.detach().cpu())
        return y

    def partial_forward(self, x: torch.Tensor, idx: int):
        """
        Copied and modified from:
        https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L223
        """
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],
                1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device),
                x
            ],
            dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Was this:
        # x = self.transformer(x)

        # Instead this:
        def get_activation(name):
            def hook(model, input, output):
                if type(output) is tuple:
                    output = output[0]
                hooked_activations[name] = output.detach()
            return hook

        hooked_activations = {}  # Repository for hooked activations

        modules = [mod for mod in self.transformer._modules.values()][0]
        handles = []
        for i, mod in enumerate(modules):
            # Get output of multi-head attention before further processing
            h0 = mod.attn.register_forward_hook(get_activation(f'attn_{i}'))
            # Get output of residual attention block
            h1 = mod.ln_2.register_forward_hook(get_activation(f'ln2_{i}'))
            handles.append(h0)
            handles.append(h1)
        count = 0
        for i in range(len(modules)):
            x = modules[i](x)
            count += 2
            if count >= idx + 1:
                break
        if count == idx + 1:
            # Even layers
            x = hooked_activations[f'ln2_{i}']
        else:
            # Odd layers
            x = hooked_activations[f'attn_{i}']
        # Old way (doesn't take sub-layers from attention blocks)
        # for i in range(idx + 1):
        #     x = modules[i](x)
        [h.remove() for h in handles]
        return x

    def _vit():
        y = []
        with autocast():
            with torch.no_grad():
                for x in loader:
                    x1 = preprocess(
                        transforms.ToPILImage()(x[0][0])
                    ).unsqueeze(0).to(device)
                    x1 = net.partial_forward(x1, layer_idx)
                    y.append(x1.detach().cpu())
        return y

    model_name = 'clip_' + arch
    if model_name not in MODEL_DICT.keys():
        net, preprocess = load_clip_model(arch, finetuned=finetuned)
        net = net.visual
        MODEL_DICT[model_name] = (net.eval(), preprocess)
    else:
        net, preprocess = MODEL_DICT[model_name]
    ds = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)  # Creating slowdowns?
    if arch.startswith('RN'):
        y = _resnet()
    elif arch.startswith('ViT'):
        # Hack to modify their forward class method to only compute graph up to
        # the layer whose activations we're taking
        net.partial_forward = partial_forward.__get__(net)
        y = _vit()
    else:
        raise NotImplementedError()
    y = torch.stack(y).float()  # Remove half precision
    return y


def get_vit_features(X, layer_idx=12, arch=''):

    def _vit():
        with torch.no_grad():
            for x in loader:
                outputs = net(**feature_extractor(
                    transforms.ToPILImage()(x[0][0]), return_tensors='pt'
                ).to(device))
                y.append(outputs.hidden_states[layer_idx])
        return y

    from transformers import ViTFeatureExtractor, ViTModel
    from datasets import load_dataset

    model_name = arch
    if model_name not in MODEL_DICT.keys():
        wt_pths = {
            'ViT-B16': 'google/vit-base-patch16-224-in21k',
            'ViT-B32': 'google/vit-base-patch32-224-in21k',
        }
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            wt_pths[arch], cache_dir=HUGGING_CACHE
        )
        net = ViTModel.from_pretrained(
            wt_pths[arch], cache_dir=HUGGING_CACHE, output_hidden_states=True
        )
        MODEL_DICT[model_name] = (net, feature_extractor)
    else:
        net, feature_extractor = MODEL_DICT[model_name]
    ds = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    y = []
    y = _vit()
    y = torch.stack(y).float()  # Remove half-precision
    return y


def load_convnext_model(arch):
    from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification

    wt_pths = {
        'convnext_base': 'facebook/convnext-base-224-22k',
        'convnext_large': 'facebook/convnext-large-224-22k',
        'convnext_base_1k': 'facebook/convnext-base-224',
        'convnext_large_1k': 'facebook/convnext-large-224',
    }
    feature_extractor = ConvNextFeatureExtractor.from_pretrained(
        wt_pths[arch], cache_dir=HUGGING_CACHE
    )
    net = ConvNextForImageClassification.from_pretrained(
        wt_pths[arch], cache_dir=HUGGING_CACHE, output_hidden_states=True
    )
    return net, feature_extractor


def get_convnext_features(X, layer_idx=12, arch=''):
    def _extract():
        with torch.no_grad():
            for x in loader:
                outputs = net(**feature_extractor(
                    transforms.ToPILImage()(x[0][0]), return_tensors='pt'
                ).to(device))
                y.append(hooked_activations[layer_idx])
        return y

    def _idx2hook():
        """
        Register hook corresponding to layer number
        """
        # Layers per 'stage' (both base and large model): [3, 3, 27, 3]
        handles = []
        idx = 0
        for stage in net.convnext.encoder.stages:
            for layer in stage.layers:
                if layer_idx == idx:
                    h = layer.register_forward_hook(
                        get_activation(layer_idx)
                    )
                    handles.append(h)
                idx += 1
        return handles

    from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
    # from datasets import load_dataset

    def get_activation(name):
        def hook(model, input, output):
            hooked_activations[name] = output.detach()
        return hook

    hooked_activations = {}  # Repository for collected activations

    model_name = arch
    if model_name not in MODEL_DICT.keys():
        net, feature_extractor = load_convnext_model(arch)
        MODEL_DICT[model_name] = (net, feature_extractor)
    else:
        net, feature_extractor = MODEL_DICT[model_name]

    handles = _idx2hook()  # Set up activation hook
    ds = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    y = []
    y = _extract()
    y = torch.stack(y).float()  # Remove half-precision
    [h.remove() for h in handles]
    return y


def sample_TCC(sims, dprime, N=100, noise_type='gaussian'):
    samples = np.array([sims] * N) * dprime
    if noise_type == 'gumbel':
        noise = np.random.gumbel(size=samples.shape)
    elif noise_type == 'gaussian':
        noise = np.random.randn(*samples.shape)
    samples += noise
    idxs = np.argmax(samples, axis=1)
    return idxs


def get_flat_no_nans(x):
    x = x.reshape(-1)
    return x[np.logical_not(np.isnan(x))]


def pixel_embed(X):
    Z = torch.stack(X)
    Z = Z.detach().cpu().numpy()
    Z = Z.reshape(len(Z), -1)
    return Z


def rgb_embed(X):
    Z = torch.stack(X).mean(axis=(-2, -1))  # Collapse down to 3 dims (R, G, B)
    Z = Z.detach().cpu().numpy()
    Z = Z.reshape(len(Z), -1)
    return Z


def torchvision_embed(X, layer_idx, arch, flatten=True):
    Z = torch.stack(X).float().to(device)
    Z = get_torchvision_features(Z, arch=arch, layer_idx=layer_idx)
    Z = Z.detach().cpu().numpy()
    if flatten:
        Z = Z.reshape(len(Z), -1)
    return Z


def places_cnn_embed(X, layer_idx, arch):
    Z = torch.stack(X).float().to(device)
    Z = get_places_cnn_features(Z, arch=arch, layer_idx=layer_idx)
    Z = Z.detach().cpu().numpy()
    Z = Z.reshape(len(Z), -1)
    return Z


def clip_embed(X, layer_idx, arch, finetuned=False):
    Z = torch.stack(X).float().to(device)
    Z = get_clip_features(
        Z, arch=arch, layer_idx=layer_idx, finetuned=finetuned
    )
    Z = Z.detach().cpu().numpy()
    Z = Z.reshape(len(Z), -1)
    return Z


def vit_embed(X, layer_idx, arch):
    Z = torch.stack(X).float().to(device)
    Z = get_vit_features(Z, arch=arch, layer_idx=layer_idx)
    Z = Z.detach().cpu().numpy()
    Z = Z.reshape(len(Z), -1)
    return Z


def convnext_embed(X, layer_idx, arch):
    Z = torch.stack(X).float().to(device)
    Z = get_convnext_features(Z, arch=arch, layer_idx=layer_idx)
    Z = Z.detach().cpu().numpy()
    Z = Z.reshape(len(Z), -1)
    return Z


def harmonized_embed(X, layer_idx, arch):
    Z = torch.stack(X).float().to(device)
    Z = get_harmonization_features(Z, arch=arch, layer_idx=layer_idx)
    Z = Z.reshape(len(Z), -1)
    return Z


def vae_embed(X, model, as_np=True):
    att_map = transforms.ToTensor()(
        np.zeros((model.input_size, model.input_size)).astype("uint8")
    )
    att_maps = torch.stack([att_map] * len(X))
    imgs = torch.stack(X)
    imgs = transforms.Resize((model.input_size, model.input_size))(imgs)
    with torch.no_grad():
        mu = model(imgs.to(device), att_maps.to(device))[0]
    # mu, _, log_var, _ = model(imgs.to(device), att_maps.to(device))
    if as_np:
        mu = mu.detach().cpu().numpy()
        # log_var = log_var.detach().cpu().numpy()
    # return mu, log_var
    # For simplicity and correspondence with other model classes, just take
    # mu as a deterministic feature vector and ignore the probability
    # distribution aspect of the VAE.
    return mu


def omni_embed(X, layer_idx, task='depth'):
    Z = torch.stack(X).float().to(device)
    Z = get_omni_features(Z, task='depth', layer_idx=layer_idx)
    Z = Z.detach().cpu().numpy()
    Z = Z.reshape(len(Z), -1)
    return Z


def load_or_train_attention_vae(
    X, ds_key, model_name='', beta=1., lr=0.00002, epochs=10000
):
    ckpt_pth = Path(
        f'attention_vae_ckpts/{model_name}_beta{beta}_{hash(ds_key)}.ckpt'
    )
    ckpt_pth.parent.mkdir(parents=True, exist_ok=True)
    model = VAEMLP(input_size=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(torch.tensor(X))
    if ckpt_pth.exists():
        # Load trained model
        checkpoint = torch.load(ckpt_pth)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # Train from scratch
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
        model, losses = train_vae_mlp(
            model, optimizer, loader, beta=beta, end_epoch=epochs
        )
        # Use hash of unique identifier of attention dataset
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_mean_loss': losses[0],
                'epoch_mean_recon_loss': losses[1],
            },
            ckpt_pth
        )
    return model


def load_bae():
    """
    NOTE: There is an implied mapping from 'TargetColor' to the CIELAB angle.
    Specifically, there are 180 colors, spaced every two degrees but starting
    at 1 (not zero). So, theta=0 is color 1, theta=2 is color 2, etc.
    'Rotation' in is the color space, such that

        TargetAngle = (TargetColor - Rotation + 1) * 2

    """
    pth = Path('bae').joinpath('ColorMemoryData_JEPG.csv')
    df = pd.read_csv(pth)
    df = df.rename(columns={n: n.strip() for n in df.columns})
    df['target'] = (df.TargetColor - 1) * 2 % 360
    df = df.rename(columns={'num_objects': 'K'})
    df['error'] = standardize_angles(df.ClickAngle - df.TargetAngle, deg=True)
    df['response'] = np.round(df.target + df.error) % 360
    # Dataset is only set-size 1, so ok to ignore other columns when setting
    # hues and probe location
    df['hues'] = [(x,) for x in df.target.values]
    df['probe_loc'] = np.zeros_like(df.target.values)
    df = df.astype(
        {key: 'int32' for key in df.columns if key not in ['ClickAngle', 'hues']}
    )
    return df


def load_panichello():
    from pymatreader import read_mat
    pth = Path('panichello').joinpath('Panichello19_exp1_human.mat')
    data = read_mat(pth)['data']
    # data is dict with keys: ['target', 'response', 'err', 'RI', 'K']
    # The value stored with each key is a list with one array per subject
    # (N=90). Each array has length number of trials (N=200).
    # Compile all data into a single data frame with a new column for subject
    # ID.
    num_subj = len(data['target'])
    dfs = []
    for i in range(num_subj):
        dfs.append(pd.DataFrame({key: val[i] for key, val in data.items()}))
        dfs[-1]['subj_id'] = [i] * len(dfs[-1])
    dfs = pd.concat(dfs)
    dfs = dfs.rename(columns={'err': 'error'})
    dfs.response = np.round(dfs.response.values * 180 / np.pi) % 360
    dfs.target = np.round(dfs.target.values * 180 / np.pi) % 360
    dfs.error = standardize_angles(dfs.error.values * 180 / np.pi, deg=True)
    dfs = dfs.astype({'response': 'int32'})
    return dfs


def load_taylor_bays(exp='bays2014'):
    if exp == 'taylor_bays2018':
        # Downloaded from: https://osf.io/2ghms/
        # Load data from 'congruent' condition of experiment. (Incongruent
        # condition drew stimuli from a non-uniform distribution to test for
        # adaptation effects.)
        # NOTE: Angles divided by factor of 2 since the original data remapped
        # from [0, 180) to [0, 360)
        pth = Path('taylor_bays').joinpath('Exp_1B_congruent_data.csv')
        df = pd.read_csv(pth)
        df.response = np.round(df.response.values * 180 / np.pi) % 360 / 2
        df.target = np.round(df.target.values * 180 / np.pi) % 360 / 2
        df['error'] = standardize_angles(df.error.values * 180 / np.pi, deg=True) / 2
        df = df.astype({
            key: 'int32' for key in df.columns if key not in ['error', 'nontargets']
        })
    elif exp.startswith('bays2014'):
        # Downloaded from: https://osf.io/s7dhn/files/osfstorage
        # NOTE: Angles divided by factor of 2 since the original data remapped
        # from [0, 180) to [0, 360)
        pth = Path('taylor_bays').joinpath('Exp1_bays2014.mat')
        # Each element of 'nontargets' column needs to be converted to
        # tuple for pandas dataframe
        from pymatreader import read_mat
        x = read_mat(pth)
        x['nontargets'] = np.round(((x['nontargets'] * 180 / np.pi) % 360) / 2)
        x['nontargets'] = [
            tuple([int(xij) for xij in xi if not np.isnan(xij)])
            for xi in x['nontargets']
        ]
        x['target'] = np.round(((x['target'] * 180 / np.pi) % 360) / 2) % 180
        x['error'] = standardize_angles(
            x['error'] * 180 / np.pi, deg=True
        ) / 2
        x['response'] = np.round(((x['response'] * 180 / np.pi) % 360) / 2)
        df = pd.DataFrame({
            # Exclude header info, etc.
            key: val for key, val in x.items() if not key.startswith('_')
        })
        df = df.astype({
            key: 'int32'
            for key in df.columns if key not in ['error', 'nontargets']
        })
        df['K'] = df.n_items.values
    else:
        raise NotImplementedError()
    return df


def load_brady_alvarez():
    from pymatreader import read_mat
    pth = Path('brady_alvarez').joinpath('dataTestAllDegreesOrd.mat')
    x = read_mat(pth)
    x['correctAnswer'] = standardize_angles(
        np.round(x['correctAnswer']), deg=True
    )
    # Data is packed into arrays of shape (participants, stimuli, locations) =
    # (300, 48, 3). Turn this into dataframe format with one row per trial.
    shp = x['subjectReport'].shape
    df = []
    idx = 0
    for i, j, k in product(*[range(s) for s in shp]):
        if not np.isnan(x['subjectReport'][i, j, k]):
            df.append(
                pd.DataFrame({
                    'subj_id': [i],
                    'stimulus_id': [j],
                    # Shift target values into range [0, 360) so that output of
                    # sample_TCC(.), which is indexes, can be subtracted from
                    # target value in stimulus key to get error. Otherwise,
                    # would need to convert from indexes to stimulus values.
                    # (See code elsewhere)
                    'target1': [(x['correctAnswer'][i, j, 0] + 360) % 360],
                    'target2': [(x['correctAnswer'][i, j, 1] + 360) % 360],
                    'target3': [(x['correctAnswer'][i, j, 2] + 360) % 360],
                    'probe_loc': [k],
                    'response': [(x['subjectReport'][i, j, k] + 360) % 360],
                    'error': [-x['subjectError'][i, j, k]],  # Sign convention for errors was flipped in dataset!
                    'K': [3],  # Setsize (for consistency with other datasets)
                }, index=[idx])
            )
            idx += 1
    df = pd.concat(df)
    df = df.astype({
        key: 'int32' for key in df.columns if key not in ['error']
    })
    return df


class TCCModel():
    """
    Base class for TCC models
    """
    def __init__(
            self,
            model_classes=None,
            imgsize_vae=256,
            imgsize_dcnn=224,
            imgsize_omni=384,
            N=4000,
            eps=1 / 4000,
            guess_rate=0,
            noise_type='gaussian',
            dp_min=0.5,
            dp_max=20,
            ngrid=48,
            dprimes=None,
            dataset='scene_wheels',
            dprime_data_pth=None,
            load_pth_samples=None,
            inherit_from=None,
            attention_method=None,
            pca_explained_var=0.98,
            nmf_comp=10,
            att_vae_beta=1.,
            max_mem=64,
            # vgg_layers=None,
            # rn50_layers=None,
            model_layers='{}',
            use_window=True,
            num_cpus=None,
            just_in_time_loading=False,
        ):
        """
        """
        # Maximum memory to allocate for caching model embeddings.
        self.max_mem = max_mem

        # When using cluster resources, sometimes I want to compute all model
        # similarities before proceeding with other computations, since I may
        # request different kinds of nodes/resource levels for different
        # components of the pipeline. In other cases, I would like to not
        # require this preloading step, and instead load similarities for
        # individual models as needed (just in time).
        self.just_in_time_loading = just_in_time_loading

        # CPU count for p_map
        if num_cpus is None:
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = num_cpus

        # Fitting parameters
        self.guess_rate = guess_rate
        self.noise_type = noise_type
        self.dp_min = dp_min
        self.dp_max = dp_max
        self.ngrid = ngrid
        self.N = N
        self.eps = eps
        # Params for 'attention' mechanism (using dimensionality reduction
        # on set of model embeddings with dataset being set of response options)
        self.attention_method = attention_method
        self.pca_explained_var = pca_explained_var
        self.nmf_comp = nmf_comp
        self.att_vae_beta = att_vae_beta
        self.att_models = {}  # Some sort of memory leak?
        self.att_dataset_max_size = 2 ** 10
        if self.attention_method == 'pca':
            att_str = f'_pca{self.pca_explained_var}'
        elif self.attention_method == 'nmf':
            att_str = f'_nmf{self.nmf_comp}'
        elif self.attention_method == 'vae':
            att_str = f'_vae{self.att_vae_beta}'
        else:
            att_str = ''

        # Data collection
        self.dataset = dataset
        if dprime_data_pth is None:
            dprime_data_pth = f'{MODEL_OUT_PATH}/dprime_fits/{dataset}'
        if load_pth_samples is None:
            load_pth_samples = f'{MODEL_OUT_PATH}/samples/resps_{dataset}_inherit_{inherit_from}'
        self.load_pth_samples = load_pth_samples
        self.dprime_data_pth = dprime_data_pth
        self.dprime_dict = {}
        self.logliks_dict = {}
        self.sims = {}
        self.rngs = {}
        # To save computation, cache already computed model embeddings
        self.embeddings_cache = {}
        # In order to make predictions across datasets using TCC model, we
        # scale similarity values according to the dataset that was used to fit
        # d'. (Note that we could opt not to rescale at all, but rescaling is
        # convenient when models vary widely in their ranges of similarity
        # values.)
        self.inherit_from = inherit_from  # Name of stimulus set to inherit from

        # Build dataframe for model metadata

        # VAE with 256x256 input
        vae_betas = [0.01]
        vae_losses = [None] * 3  # None for px-mse, int for dnn-embed layer
        vae_ckpts = [12, 12, 12]
        vae_specs = list(zip(vae_betas, vae_losses, vae_ckpts))

        model_layers = eval(model_layers)  # Possibly overwrite layer selection for subset of models
        layers_dict = {
            'vgg19': model_layers.get('vgg19') or list(range(5, 37)),  # Trained IN-1k
            'resnet50': model_layers.get('resnet50') or list(range(0, 21)),  # Trained IN-1k
            'places_resnet50': model_layers.get('places_resnet50') or [0, 1, 2, 3, 4],
            'clip_ViT-B16': model_layers.get('clip_ViT-B16') or list(range(24)),
            'finetuned_clip_ViT-B16': model_layers.get('finetuned_clip_ViT-B16') or list(range(24)),
            'clip_ViT-B32': model_layers.get('clip_ViT-B32') or list(range(24)),
            'clip_RN50': model_layers.get('clip_RN50') or list(range(0, 27)),
            'finetuned_clip_RN50': model_layers.get('finetuned_clip_RN50') or list(range(27)),
            'clip_RN50x4': model_layers.get('clip_RN50x4') or list(range(13, 37)),
            'clip_RN50x16': model_layers.get('clip_RN50x16') or list(range(25, 51)),
            'clip_RN50x64': model_layers.get('clip_RN50x64') or list(range(60, 75)),
            'clip_RN101': model_layers.get('clip_RN101') or list(range(17, 44)),
            'ViT-B16': model_layers.get('ViT-B16') or [10, 11, 12],  # Input size 224, trained IN-22k
            'ViT-B32': model_layers.get('ViT-B32') or [10, 11, 12],  # Input size 224, trained IN-22k
            'convnext_base': model_layers.get('convnext_base') or list(range(10, 36)),  # IN-22k
            'convnext_base_1k': model_layers.get('convnext_base_1k') or list(range(10, 36)),
            'convnext_large': model_layers.get('convnext_large') or list(range(10, 36)),  # IN-22k
            'convnext_large_1k': model_layers.get('convnext_large_1k') or list(range(10, 36)),
            'harmonized_RN50': model_layers.get('harmonized_RN50') or list(range(0, 21)),
            'harmonized_ViT_B16': model_layers.get('harmonized_RN50') or list(range(20, 50)),  # TODO
            'omnidata_depth': model_layers.get('omnidata_depth') or [0],  # [0, 5]
        }
        imgsize_dict = {
            'vgg19': imgsize_dcnn,
            'resnet50': imgsize_dcnn,
            'places_resnet50': imgsize_dcnn,
            'clip_ViT-B16': imgsize_dcnn,
            'finetuned_clip_ViT-B16': imgsize_dcnn,
            'clip_ViT-B32': imgsize_dcnn,
            'clip_RN50': imgsize_dcnn,
            'finetuned_clip_RN50': imgsize_dcnn,
            'clip_RN50x4': imgsize_dcnn,
            'clip_RN50x16': imgsize_dcnn,
            'clip_RN50x64': imgsize_dcnn,
            'clip_RN101': imgsize_dcnn,
            'ViT-B16': imgsize_dcnn,
            'ViT-B32': imgsize_dcnn,
            'convnext_base': imgsize_dcnn,
            'convnext_base_1k': imgsize_dcnn,
            'convnext_large': imgsize_dcnn,
            'convnext_large_1k': imgsize_dcnn,
            'harmonized_RN50': imgsize_dcnn,
            'harmonized_ViT_B16': imgsize_dcnn,
            'omnidata_depth': imgsize_omni,
        }
        models = pd.DataFrame({
            'name': [],
            'path': [],
            'model_class': [],
            'specs': [],
            'image_size': [],
        })
        mclasses_ = ['rgb'] + ['pixels'] + ['vae']  * len(vae_betas)
        for name, idxs in layers_dict.items():
            mclasses_ += [name] * len(idxs)
        models['model_class'] = mclasses_

        # Model: RGB (color channel means)
        models.loc[models.model_class == 'rgb', 'name'] = 'rgb' + att_str
        models.loc[models.model_class == 'rgb', 'image_size'] = imgsize_dcnn

        # Model: Pixels
        models.loc[models.model_class == 'pixels', 'name'] = 'pixels' + att_str
        models.loc[models.model_class == 'pixels', 'image_size'] = imgsize_dcnn

        # Model: beta-VAE
        models.loc[models.model_class == 'vae', 'name'] = [
            f'vae_beta{beta}{f"_layer{x}" if x is not None else ""}{att_str}'
            for beta, x in zip(vae_betas, vae_losses)
        ]
        models.loc[models.model_class == 'vae', 'image_size'] = imgsize_vae
        models.loc[models.model_class == 'vae', 'specs'] = [
            str(x) for x in vae_specs
        ]

        for name in layers_dict.keys():
            models.loc[models.model_class == name, 'name'] = [
                f'{name}_l{layer}{att_str}' for layer in layers_dict[name]
            ]
            models.loc[models.model_class == name, 'image_size'] = imgsize_dict[name]
            models.loc[models.model_class == name, 'specs'] = layers_dict[name]

        # File path names
        models.path = [
            Path(f'{MODEL_OUT_PATH}/similarity_data/{dataset}/sims_{n}.npy')
            for n in models.name.values
        ]

        models = models.astype({'image_size': 'int32'})
        self.models = models

        # Input image sizes for different model classes
        self.imgsize_dcnn = imgsize_dcnn
        self.imgsize_vae = imgsize_vae
        self.imgsize_omni = imgsize_omni

        # Filter for specified subset of models
        if model_classes is None:
            self.idxs = list(range(len(models)))
        else:
            self.idxs = [
                i for i in range(len(models))
                if models.model_class.values[i] in model_classes
            ]

        if dprimes is None:
            dprimes = np.linspace(self.dp_min, self.dp_max, self.ngrid)
        elif not hasattr(dprimes, '__len__'):
            dprimes = [dprimes]
        self.dprimes = {}
        for mclass in models.model_class:
            if type(dprimes) is dict:
                self.dprimes[mclass] = dprimes.get(
                    mclass, np.linspace(self.dp_min, self.dp_max, self.ngrid)
                )
            else:
                self.dprimes[mclass] = dprimes

    def _embed(
        self, model_name, target_id, choice_ids, embed_func, img_size, *args, **kwargs,
    ):
        """
        target_id and choice_ids (interable) are unique ids, which we use to
        look up already-computed model embeddings, or compute them for the
        first time if they do not exist
        """
        if not hasattr(self, 'embeddings_cache'):
            self.embeddings_cache = {}
        # NOTE: NO LONGER EXPERIMENTING WITH ATTENTION STUFF
        # If using attention, get dimensionality reduction transform first, to
        # transform the embeddings
        att_model = None
        if self.attention_method:
            Y = []
            att_dataset_id = tuple(self._attention_dataset_iter(target_id))
            if att_dataset_id in self.att_models:
                att_model, scaler = self.att_models[att_dataset_id]
            else:
                for sid in att_dataset_id:
                    if sid in self.embeddings_cache.keys():
                        Y.append(self.embeddings_cache[sid])
                    else:
                        img = self._image_from_stimulus_id(sid, img_size)
                        Y.append(embed_func([img], *args, **kwargs))
                        # Only add to cache if we haven't run out of room
                        nbytes = psutil.Process(os.getpid()).memory_info().vms
                        if nbytes < self.max_mem * 1e9:
                            self.embeddings_cache[sid] = Y[-1]
                att_model, scaler = self._get_attention_transform(
                    np.vstack(Y), ds_key=att_dataset_id, model_name=model_name
                )
                self.att_models[att_dataset_id] = (att_model, scaler)
            del Y  # Could be quite big
        Z = []
        for sid in [target_id] + choice_ids:
            if sid in self.embeddings_cache.keys():
                Z.append(self.embeddings_cache[sid])
            else:
                img = self._image_from_stimulus_id(sid, img_size)
                Z.append(embed_func([img], *args, **kwargs))
                # Only add to cache if we haven't run out of room
                nbytes = psutil.Process(os.getpid()).memory_info().vms
                if nbytes < self.max_mem * 1e9:
                    self.embeddings_cache[sid] = Z[-1]
        Z = np.vstack(Z)
        if self.attention_method in ['pca', 'nmf']:
            if scaler:
                Z = att_model.transform(scaler.transform(Z))
            else:
                Z = att_model.transform(Z)
        elif self.attention_method == 'vae':
            Z = att_model(torch.tensor(Z).to(device))[0].detach().cpu().numpy()
        return Z

    # MEMORY-HEAVY IMPLEMENTATION
    def _get_model_similarities(self, idx):
        print("Collecting model similarities")

        img_size = int(self.models.image_size.values[idx])
        mname = self.models.name.values[idx]

        # Load VAE if necessary
        mclass = self.models.model_class.values[idx]
        if mclass == 'vae':
            beta, layer_idx, epoch = eval(self.models.specs.values[idx])
            if layer_idx is None:
                pixel_only = True
            else:
                pixel_only = False
            vae = make_and_load(make_vae_args(
                beta=beta,
                start_epoch=epoch,
                pixel_only=pixel_only,
                layer=layer_idx
            ))[0].eval()

        # Collect similarities
        sims_dict = {}
        for key, target_id, choice_ids in self._stimulus_id_iter(sims_dict):
            if mclass == 'vae':
                Z = self._embed(
                    mname, target_id, choice_ids, vae_embed, img_size, vae
                )
            elif mclass in ['vgg19', 'resnet50']:
                Z = self._embed(
                    mname,
                    target_id,
                    choice_ids,
                    torchvision_embed,
                    img_size,
                    *(self.models.specs.values[idx], mclass),
                )
            elif 'places' in mclass:  # Same size for all Places-CNN nets
                Z = self._embed(
                    mname,
                    target_id,
                    choice_ids,
                    places_cnn_embed,
                    img_size,
                    *(self.models.specs.values[idx], mclass.split('_')[-1])
                )
            elif mclass == 'omnidata_depth':
                Z = self._embed(
                    mname,
                    target_id,
                    choice_ids,
                    omni_embed,
                    img_size,
                    self.models.specs.values[idx]
                )
            elif mclass.startswith('clip'):
                Z = self._embed(
                    mname,
                    target_id,
                    choice_ids,
                    clip_embed,
                    img_size,
                    *(self.models.specs.values[idx], mclass.split('_')[-1]),
                )
            elif mclass.startswith('finetuned_clip'):
                Z = self._embed(
                    mname,
                    target_id,
                    choice_ids,
                    clip_embed,
                    img_size,
                    *(self.models.specs.values[idx], mclass.split('_')[-1]),
                    finetuned=True,
                )
            elif mclass.startswith('ViT'):
                Z = self._embed(
                    mname,
                    target_id,
                    choice_ids,
                    vit_embed,
                    img_size,
                    *(self.models.specs.values[idx], mclass),
                )
            elif mclass.startswith('convnext'):
                Z = self._embed(
                    mname,
                    target_id,
                    choice_ids,
                    convnext_embed,
                    img_size,
                    *(self.models.specs.values[idx], mclass),
                )
            elif mclass.startswith('harmonized'):
                Z = self._embed(
                    mname,
                    target_id,
                    choice_ids,
                    harmonized_embed,
                    img_size,
                    *(self.models.specs.values[idx], mclass),
                )
            elif mclass == 'pixels':
                Z = self._embed(
                    mname, target_id, choice_ids, pixel_embed, img_size
                )
            elif mclass == 'rgb':
                Z = self._embed(
                    mname, target_id, choice_ids, rgb_embed, img_size
                )
            else:
                raise NotImplementedError()
            sims = cossim_torch(Z[0], Z[1:]).detach().cpu()
            sims_dict[key] = np.array(sims)
        del self.embeddings_cache  # Free up space
        return sims_dict

    def _scale_shift_similarities(self, sims_dict, name):
        if name in self.rngs.keys():
            smin, smax = self.rngs[name]
        else:
            # Get range for this similarity data
            smin = np.inf
            smax = -np.inf
            for key, val in sims_dict.items():
                vmin = val.min()
                vmax = val.max()
                if vmin < smin:
                    smin = vmin
                if vmax > smax:
                    smax = vmax
            self.rngs[name] = (smin, smax)
        sims_dict1 = {}
        for key, val in sims_dict.items():
            sims_dict1[key] = (val - smin) / (smax - smin)
        return sims_dict1

    def _load_or_get_sims(self, idxs=None):
        if idxs is None:
            idxs = self.idxs
        for i in idxs:
            mname = self.models.name.values[i]
            mclass = self.models.model_class.values[i]
            pth = self.models.path.values[i]
            # Load or compute similarities
            if '+' in mname:
                # Average over window of consecutive layers
                idx0 = self.models.specs.values[i]
                models_win = self.models[
                    (self.models.model_class == mclass) &
                    (self.models.specs.isin(self.models.specs.values[i]))
                ]
                sims = []
                for j in range(len(models_win)):
                    pth1 = models_win.path.values[j]
                    if pth1.exists():
                        sims_i = np.load(pth1, allow_pickle=True).item()
                    else:
                        sims_i = self._get_model_similarities(
                            int(models_win.index.values[j])
                        )
                    sims.append(sims_i)
                self.sims[mname] = {
                    key: np.mean([x[key] for x in sims], axis=0)
                    for key in sims[0].keys()
                }
            elif pth.exists():
                self.sims[mname] = np.load(
                    pth, allow_pickle=True
                ).item()
                print(f"Loaded similarities from {pth}")
            else:
                print(f'Path {pth} does not exist. Computing similarities.')
                self.sims[mname] = self._get_model_similarities(i)
                pth.parent.mkdir(parents=True, exist_ok=True)
                np.save(pth, self.sims[mname])
            # Rescale similarities to fall within a range
            if self.inherit_from:
                rng_pth = Path(
                    f'{MODEL_OUT_PATH}/similarity_data/{self.inherit_from}/rng_{mname}.npy'
                )
                # Handle two cases:
                # 1. Range data has already been computed, so just load
                # 2. Range data has not yet been computed, but similarities
                #     have. (Compute range on the fly from similarities.)
                if not rng_pth.exists():
                    print(
                        f'Could not find range data for {self.inherit_from}. '
                        'Loading sims and computing...'
                    )
                    sim_inherit_pth = Path(
                        f'{MODEL_OUT_PATH}/similarity_data/{self.inherit_from}/sims_{mname}.npy'
                    )
                    sims_inherit = np.load(
                        sim_inherit_pth, allow_pickle=True
                    ).item()
                    # Range gets added to self.rngs inside this function
                    self._scale_shift_similarities(sims_inherit, mname)
                    np.save(rng_pth, self.rngs[mname])
                else:
                    print(
                        f'Loading similarity rescaling parameters from {rng_pth}'
                    )
                    self.rngs[mname] = np.load(rng_pth)
            else:
                self.sims[mname] = self._scale_shift_similarities(
                    self.sims[mname], mname
                )
                rng_pth = Path(
                    f'{MODEL_OUT_PATH}/similarity_data/{self.dataset}/rng_{mname}.npy'
                )
                rng_pth.parent.mkdir(parents=True, exist_ok=True)
                np.save(rng_pth, self.rngs[mname])
        return

    def _load_or_fit_dprimes(self, idxs=None):
        if idxs is None:
            idxs = self.idxs
        root = Path(self.dprime_data_pth)
        root.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {'model_name': [], 'dprime': [], 'loglik': []}
        )
        pths = [
            root.joinpath(f'{self.models.name.values[idx]}.csv')
            for idx in idxs
        ]
        if not self.just_in_time_loading and not all([pth.exists() for pth in pths]):
            # If any path is missing, load all similarities. Doing it this way
            # rather than one by one because when using the cluster, I like to
            # request different resources for d-prime calculations vs
            # similarities. Thus, I want to keep them in separate blocks. At
            # the same time, if I've already done all the d-prime calculations,
            # I don't want to bother loading similarities.
            self._load_or_get_sims()
        for idx, pth in zip(idxs, pths):
            mclass = self.models.model_class.values[idx]
            mname = self.models.name.values[idx]
            if not pth.exists():
                self._fit(idxs=[idx])
                df1 = pd.DataFrame({
                    'model_name': [mname] * len(self.dprimes[mclass]),
                    'dprime': self.dprimes[mclass],
                    'loglik': self.logliks_dict[mname]
                })
                df1.to_csv(pth)
            else:
                df1 = pd.read_csv(pth, index_col=0)
                if not np.allclose(df1.dprime.values, np.array(self.dprimes[mclass])):
                    raise Exception(
                        f'd-prime values loaded from {pth} do not match.'
                    )
            df = pd.concat([df, df1], ignore_index=True)
        df.to_csv(root.joinpath('summary.csv'))
        self.dprime_data = df
        return

    def _stimulus_id_iter(self):
        """
        Overwrite this method for specific dataset when subclassing
        """
        raise NotImplementedError()

    def _get_key_from_idx(self, idx):
        """
        Overwrite this method for specific dataset when subclassing
        """
        raise NotImplementedError()

    def _attention_dataset_iter(self, stim_id):
        """
        Overwrite this method for specific dataset when subclassing
        """
        raise NotImplementedError()

    def _get_attention_transform(self, X, ds_key=None, model_name=''):
        """
        Apply 'attention' to TCC model, which consists of dimensionality
        reduction over a larger set of stimuli. The idea is that people try to
        allocate their limited attention within a stimulus to features that
        will be diagnostic given the set of responses. Here, we choose a subset
        of stimuli from the experimental distribution to represent the set of
        alternatives that people have in mind. (Note that it does not strictly
        have to be the case that the relevant subset lies wholly within the
        experimental distribution, so the code could be modified to incorporate
        that possibility.)
        """
        scaler = None
        if self.attention_method == 'pca':
            scaler = StandardScaler()
            model = PCA(n_components=self.pca_explained_var)
            # X1 = scaler.fit_transform(X)
            # model.fit(X1)
            model.fit(X)
        elif self.attention_method == 'nmf':
            model = NMF(n_components=self.nmf_comp)
            model.fit(X)
        elif self.attention_method == 'vae':
            model = load_or_train_attention_vae(
                X, ds_key, beta=self.att_vae_beta, model_name=model_name,
            )
        return model, scaler

    def _likelihood_TCC_gauss(self, idx, dprime):
        """
        Get set of Monte Carlo PMFs for likelihood of empirical data assuming
        Gaussian noise is added to similarity
        """
        sims_dict = self.sims[self.models.name.values[idx]]
        resps_dict = self.get_or_load_TCC_samples(idx, dprime)[0]
        probs = {}
        for key in sims_dict.keys():
            if key not in probs.keys():
                # Sample responses to get histogram
                resps = resps_dict[key]
                hist = np.histogram(resps, bins=range(361), density=True)[0]
                hist[hist == 0] = self.eps
                probs[key] = hist.copy()
        return probs

    def _likelihood_TCC_gumbel(self, idx, dprime):
        """
        Assume Gumbel error, which means likelihood is just a softmax
        """
        sims_dict = self.sims[self.models.name.values[idx]]
        probs = {}
        for key in sims_dict.keys():
            if key not in probs.keys():
                sims = sims_dict[key]
                # Softmax with temperature=dprime
                probs[key] = np.exp(dprime * sims) / np.exp(
                    dprime * sims).sum()
        return probs

    def _fit_TCC(self, idx):
        """
        Fit TCC model to response data.

        Similarities should be computed on the stimulus set used to collect
        responses.
        """
        if not hasattr(self, 'human_data'):
            raise Exception('Response data not found for fitting model.')

        def parfunc(dprime):
            probs = likelihood(idx, dprime)
            loglik = 0
            sims_dict = self.sims[self.models.name.values[idx]]
            for i in range(len(self.human_data)):
                key = self._get_key_from_idx(i)
                # loglik += np.log(probs[key][self.human_data.response.values[i]])
                p = probs[key][self.human_data.response.values[i]]
                loglik += np.log(
                    (1 - self.guess_rate) * p + self.guess_rate / 360
                )
            return loglik

        if self.noise_type == 'gumbel':
            # Gumbel noise --> softmax likelihood
            likelihood = self._likelihood_TCC_gumbel
        else:
            # Gaussian noise (histogram approximation for likelihood)
            likelihood = self._likelihood_TCC_gauss
        mclass = self.models.model_class.values[idx]
        logliks = p_map(
            parfunc,
            self.dprimes[mclass],
            desc=f'Fitting d-prime ({self.models.name.values[idx]})',
            num_cpus=self.num_cpus,
        )
        dp_best = self.dprimes[mclass][logliks.index(max(logliks))]
        self.dprime_dict[self.models.name.values[idx]] = dp_best
        self.logliks_dict[self.models.name.values[idx]] = logliks
        return dp_best, logliks

    def _fit(self, idxs=None):
        if idxs is None:
            idxs = self.idxs
        [self._fit_TCC(i) for i in idxs]

    def get_or_load_TCC_samples(self, idx, dprime, dilation_factor=1):
        name = self.models.name.values[idx]
        load_pth = Path(self.load_pth_samples).joinpath(
            f'{name}_dprime_{dprime}.npy'
        )
        if str(load_pth) in TCC_SAMPLE.keys():
            resps = TCC_SAMPLE[str(load_pth)]
            print(f'Loaded TCC samples for {name} from RAM')
        elif load_pth.exists():
            resps = np.load(load_pth, allow_pickle=True).item()
            print(f'Loaded TCC samples for {name} from {load_pth}')
            # TCC_SAMPLE[str(load_pth)] = resps
        else:
            if name not in self.sims.keys():
                self._load_or_get_sims([idx])
            print(f'Sampling TCC responses for {name}, d-prime={dprime}')
            resps = {}
            for key, sims in self.sims[name].items():
                # Sample responses to get histogram of errors
                resps[key] = sample_TCC(
                    sims, dprime, N=self.N, noise_type=self.noise_type
                )
            Path(load_pth).parent.mkdir(parents=True, exist_ok=True)
            np.save(load_pth, resps)
            # TCC_SAMPLE[str(load_pth)] = resps
        errs = {}
        for key, r in resps.items():
            errs[key] = standardize_angles(
                # Use dilation factor e.g. to handle angular stimuli that
                # are modulo 180 deg, rather than modulo 360 deg, such as
                # oriented gabor stimuli.
                (r - key[-1]) * dilation_factor, deg=True
            ) / dilation_factor
        return resps, errs

    def get_human_errors_by_key(self, idxs=None):
        # Optionally pass enumerable of indexes to dataset rows. This allows,
        # e.g. to bootstrap sample the data
        if idxs is None:
            idxs = np.arange(len(self.human_data))
        errs = {}
        for i in idxs:
            key = self._get_key_from_idx(i)
            err = self.human_data.error.values[i]
            if key not in errs.keys():
                errs[key] = [err]
            else:
                errs[key].append(err)
        return errs


class TCCSceneWheel(TCCModel):
    def __init__(self, *args, **kwargs):
        super(TCCSceneWheel, self).__init__(*args, **kwargs)

        # Load human data for scene wheels exp.
        self.human_data = pd.read_csv(
            "scene_wheels_mack_lab_osf/Data/sceneWheel_main_data_n20.csv"
        )
        self.human_data['error'] = standardize_angles(
            self.human_data.resp_error, deg=True
        )
        self.human_data.loc[self.human_data.response == 360, "response"] = 0
        self.wheel_ids = self.human_data.wheel_num.values
        self.radii = self.human_data.radius.values
        self.targets = self.human_data.answer.values
        self.radius_vals = np.unique(self.radii).tolist()
        self.wheel_id_vals = np.unique(self.wheel_ids).tolist()

    def _get_key_from_idx(self, idx):
        """
        Get stimulus look-up key (used in various dicts) from trial index
        corresponding to response data.
        """
        key = (self.wheel_ids[idx], self.radii[idx], self.targets[idx])
        return key

    def _stimulus_id_iter(self, sims_dict={}):
        """
        Identify each stimulus image with a unique id tag and iterate through
        them. This scheme allows for caching precomputed values, as we can
        keep a dictionary of model embeddings, whose keys are the unique
        stimulus ids.
        """
        for i in tqdm(range(len(self.human_data)), dynamic_ncols=True):
            key = self._get_key_from_idx(i)
            # For efficiency, only generate stimulus if we haven't already
            if key not in sims_dict.keys():
                Y_ids = [(self.wheel_ids[i], self.radii[i], j) for j in range(360)]
                x_id = (self.wheel_ids[i], self.radii[i], self.targets[i])
                yield key, x_id, Y_ids

    def _attention_dataset_iter(self, stim_id):
        """
        Generate dataset to do dimensionality reduction over for this stimulus.
        Here, we assume this consists of the whole stimulus set (i.e. 25 wheels
        x 360 stimuli per wheel).
        """
        for x in product(self.wheel_id_vals, self.radius_vals, range(360)):
            if x[-1] % 8 == 0:
                yield x

    def _image_from_stimulus_id(self, sid, size):
        if size not in scene_wheel_imgs:
            scene_wheel_imgs[size] = load_wheel_imgs(size)
        wheels = scene_wheel_imgs[size]
        img = wheels[sid[0]][sid[1]][sid[2]]
        return img

    def _get_values_list_flat(
        self, stimulus_keys, values_by_key, radii, bins, use_abs=True,
        transform=lambda x: x,
    ):
        """
        Handles bin-averaging for Scene Wheels dataset specifically, flattening
        results into a list. Nested iteration is over wheel ID, radius, and
        target value.
        """
        values_bin_ave_flat = []
        for wid in self.wheel_id_vals:
            for rad in radii:
                keys_set = [
                    key for key in stimulus_keys
                    if key[0] == wid and key[1] == rad
                ]
                values_dict = {
                    key: values_by_key[key]
                    for key in keys_set if key in values_by_key
                }
                values_binned = _get_binned_averages(
                    values_dict, bins, use_abs=use_abs, transform=transform
                )
                values_bin_ave_flat.extend([x for x in values_binned.values()])
        return values_bin_ave_flat

    def summary(self, num_boot=1000, binsize=30):
        """
        Get statistical summary of model performances against human data.
        """

        def _get_error_dataframe(idx):
            name = self.models.name.values[idx]
            mclass = self.models.model_class.values[idx]
            df_pth = Path(
                str(out_pth.with_suffix('')) + f'_{name}_rad_{rad}.csv'
            )
            if df_pth.exists():
                df_idx = pd.read_csv(df_pth, index_col=0)
            else:
                self._load_or_get_sims([idx])
                self._load_or_fit_dprimes([idx])
                df_ = self.dprime_data[self.dprime_data.model_name == name]
                dp_best = df_[df_.loglik == df_.loglik.max()].dprime.values[0]
                model_errs_by_stim = self.get_or_load_TCC_samples(
                    idx, dp_best
                )[1]
                merrs_bin_ave_flat = self._get_values_list_flat(
                    stimulus_keys, model_errs_by_stim, rad_set, bins
                )
                rho, pval = spearmanr(merrs_bin_ave_flat, herrs_bin_ave_flat)
                df_idx = pd.DataFrame({
                    'model_class': [mclass],
                    'model_name': [name],
                    'dprime': [dp_best],
                    'loglik': [df_.loglik.max()],  # LL for for all trials, therefore duplicated across radii
                    'radius': [rad],
                    'mean_abs_err': [np.mean(merrs_bin_ave_flat)],
                    'spearman_r': [rho],
                    'spearman_pval': [pval],
                    'conf_int95_lower': [np.nan],
                    'conf_int95_upper': [np.nan],
                })
                df_idx.to_csv(df_pth)
            return df_idx

        out_pth = Path(
            f'{MODEL_OUT_PATH}/scene_wheels_analysis/summary.csv'
        )
        out_pth.parent.mkdir(parents=True, exist_ok=True)

        data = pd.DataFrame({
            'model_class': [],
            'model_name': [],
            'dprime': [],
            'radius': [],
            'mean_abs_err': [],
            'spearman_r': [],
            'spearman_pval': [],
            'conf_int95_lower': [],
            'conf_int95_upper': [],
        })

        bins = np.arange(0, 360 + 1, binsize)
        human_errs_by_stim = self.get_human_errors_by_key()
        stimulus_keys = set([
            self._get_key_from_idx(i) for i in range(len(self.human_data))
        ])
        stimulus_keys = list(stimulus_keys)
        for j, rad in enumerate(['all'] + self.radius_vals):
            if rad == 'all':
                rad_set = self.radius_vals
            else:
                rad_set = [rad]
            herrs_bin_ave_flat = self._get_values_list_flat(
                stimulus_keys, human_errs_by_stim, rad_set, bins
            )
            # THIS ISN'T WORKING. COMPUTED VALUES ARE GETTING OVERWRITTEN
            # FOR SOME REASON. MAYBE A NAMESPACE ISSUE.
            # dfs_idx = p_map(
            #     _get_error_dataframe,
            #     self.idxs,
            #     desc=f'Sampling responses from best-fit TCC models',
            #     num_cpus=self.num_cpus,
            # )
            # data = pd.concat([data] + dfs_idx)
            for idx in self.idxs:
                df_idx = _get_error_dataframe(idx)
                data = pd.concat([data, df_idx])

            # Bootstrap analysis to get estimate of ceiling on rank correlations.
            # For each boostrap resample, compute rank correlation on abs error
            # with original data. Then get confidence interval on this correlation
            # coefficient. Do this for current radius, marginalizing over wheel ids.
            conf_int_pth = Path(
                f'{MODEL_OUT_PATH}/confidence_intervals/scene_wheels/radius_{rad}.npy'
            )
            if conf_int_pth.exists():
                conf_lower, conf_upper = np.load(conf_int_pth)
            else:
                spearman_boot = []
                for k in tqdm(range(num_boot), desc='Bootstrap analysis', dynamic_ncols=True):
                    # Bootstrap resample within radius
                    boot_idxs = self.human_data.groupby([
                        'wheel_num', 'radius', pd.cut(self.human_data.answer, np.arange(-1, 360, 30))
                    ]).sample(frac=1, replace=True).index
                    herrs_by_stim_k = self.get_human_errors_by_key(boot_idxs)
                    herrs_bin_ave_flat_k = self._get_values_list_flat(
                        stimulus_keys, herrs_by_stim_k, rad_set, bins
                    )
                    spearman_boot.append(
                        spearmanr(herrs_bin_ave_flat, herrs_bin_ave_flat_k)
                    )
                rhos = np.sort([sp[0] for sp in spearman_boot])
                conf_lower = rhos[int(round(num_boot * 0.05))]
                conf_upper = rhos[int(round(num_boot * 0.95))]
                conf_int_pth.parent.mkdir(parents=True, exist_ok=True)
                np.save(conf_int_pth, np.array([conf_lower, conf_upper]))
            human_df = pd.DataFrame({
                'model_class': ['human'],
                'model_name': ['human'],
                'dprime': [np.nan],
                'radius': [rad],
                'mean_abs_err': [np.mean(herrs_bin_ave_flat)],
                'spearman_r': [np.nan],
                'spearman_pval': [np.nan],
                'conf_int95_lower': [conf_lower],
                'conf_int95_upper': [conf_upper],
            })
            human_df_pth = Path(
                str(out_pth.with_suffix('')) + f'_human_rad_{rad}.csv'
            )
            human_df.to_csv(human_df_pth)
            data = pd.concat([data, human_df])
        data.to_csv(out_pth)
        return data


class TCCSetsize(TCCModel):
    """
    Test image-computable models on set-size displays by turning a continuous
    report task into an AFC with many choices (e.g. 360) and applying the TCC
    model. We can set d' to its best-fit value from natural image tasks to ask
    how well a model generalizes across the domains.
    """
    def __init__(
        self,
        setsize,
        *args,
        item_type='colors',
        num_incr=None,
        lab_center=[60., 22., 14.],
        lab_radius=52.,
        sqsize=12,
        dataset=None,
        **kwargs
    ):
        if item_type == 'colors':
            item_type += '_c' + '_'.join([str(x) for x in lab_center])
            item_type += f'_r{lab_radius}'
        if dataset is None:
            # Dataset name can be specified e.g. when subclassing, but if not,
            # the name defaults to this
            dataset = f'setsize{setsize}_{item_type}'
        super(TCCSetsize, self).__init__(
            *args,
            dataset=dataset,
            **kwargs
        )
        self.sqsize = sqsize
        self.setsize = setsize
        self.num_incr = num_incr
        self.item_type = item_type
        self.lab_center = [float(x) for x in lab_center]
        self.lab_radius = float(lab_radius)

    def _colors_ring_generator(self, hue_set):
        hues_gen = product(hue_set, repeat=self.setsize)
        for hues in hues_gen:
            for probe_loc in range(self.setsize):
                probes = []
                for j in range(360):
                    h = np.array(hues)
                    h[probe_loc] = j
                    probes.append(make_colors_ring(
                        h * np.pi / 180,
                        size=self.imgsize_dcnn,
                        sqsize=self.sqsize,
                        lab_center=self.lab_center,
                        radius=self.lab_radius,
                    )[0])
                target = make_colors_ring(
                    np.array(hues) * np.pi / 180,
                    size=self.imgsize_dcnn,
                    sqsize=self.sqsize,
                    lab_center=self.lab_center,
                    radius=self.lab_radius,
                )[0]
                yield (hues, probe_loc, target, probes)

    def _gabors_ring_generator(self, angle_set):
        angles_gen = product(angle_set, repeat=self.setsize)
        for angles in angles_gen:
            for probe_loc in range(self.setsize):
                probes = []
                for j in range(180):
                    h = np.array(angles)
                    h[probe_loc] = j
                    probes.append(make_gabors_ring(
                        h * np.pi / 180,
                        size=self.imgsize_dcnn,
                        sigx=4.,
                        sigy=6.,
                    )[0])
                target = make_gabors_ring(
                    np.array(angles) * np.pi / 180,
                    size=self.imgsize_dcnn,
                    sigx=4.,
                    sigy=6.,
                )[0]
                yield (angles, probe_loc, target, probes)

    def _attention_dataset_iter(self, stim_id):
        """
        Generate dataset to do dimensionality reduction over for this stimulus.
        Here, we assume people apply spatial attention, which implies that the
        dataset consists of all stimuli for which there are items at the exact
        same locations.
        """
        num_incr = int(np.floor(
            self.att_dataset_max_size ** (1 / self.setsize)
        ))
        num_incr = min(num_incr, 360)  # No higher than one sample per degree
        angles_gen = product(
            np.linspace(0, 360, num_incr + 1)[:-1], repeat=self.setsize
        )
        for angles in angles_gen:
            yield (angles, stim_id[1])

    def _stimulus_id_iter(self, sims_dict={}):
        """
        Identify each stimulus image with a unique id tag and iterate through
        them. This scheme allows for caching precomputed values, as we can
        keep a dictionary of model embeddings, whose keys are the unique
        stimulus ids.
        """
        # TODO: Redo so that stim ids match format of Bays/Panichello etc.
        # Specifically, need to choose item locations and add to id
        raise NotImplementedError()
        if 'gabors' in self.item_type:
            max_deg = 180
        else:
            max_deg = 360
        if self.num_incr is None:
            self.num_incr = max_deg
        angle_set = np.linspace(0, max_deg, self.num_incr + 1)[:-1]
        N = len(angle_set) ** self.setsize * self.setsize
        angles_gen = product(angle_set, repeat=self.setsize)
        for angles in tqdm(angles_gen, total=N, dynamic_ncols=True):
            for probe_loc in range(self.setsize):
                Y_ids = []
                for j in range(max_deg):
                    h = np.array(angles)
                    h[probe_loc] = j
                    Y_ids.append(tuple(h))
                key = (angles, probe_loc, angles[probe_loc])
                x_id = angles
                yield key, x_id, Y_ids

    def _image_from_stimulus_id(self, sid, size):
        values, locs = sid  # Item values and respective locations on ring
        if 'gabors' in self.item_type:
            img = make_gabors_ring(
                np.array(values) * np.pi / 180,
                locs=locs,
                size=self.imgsize_dcnn,
                sigx=4.,
                sigy=6.,
            )[0]
        elif 'colors' in self.item_type:
            img = make_colors_ring(
                np.array(values) * np.pi / 180,
                locs=locs,
                size=self.imgsize_dcnn,
                sqsize=self.sqsize,
                lab_center=self.lab_center,
                radius=self.lab_radius,
            )[0]
        elif 'lines' in self.item_type:
            img = make_colored_lines_ring(
                np.array(values) * np.pi / 180,
                locs=locs,
                size=self.imgsize_dcnn,
                line_length=self.line_length,
                line_thickness=self.line_thickness,
                eccentricity=self.item_eccentricity,
            )[0]
        else:
            raise NotImplementedError()
        return transforms.ToTensor()(img)

    def sample_model_errors(self, dprime, idxs=None):
        """
        Sample TCC model responses to get errors for given dprime value
        """
        if  'colors' in self.item_type:
            dilation_factor = 1
        elif self.item_type == 'gabors':
            dilation_factor = 2
        elif self.item_type == 'lines':
            dilation_factor = 2
        else:
            raise NotImplementedError()
        if idxs is None:
            idxs = self.idxs
        self.errs = {}
        for i in idxs:
            name = self.models.name.values[i]
            self.errs[name] = self.get_or_load_TCC_samples(
                i, dprime, dilation_factor=dilation_factor
            )[1]
        return self.errs


class TCCBays(TCCSetsize):
    """
    Analyze data from:
    'Efficient coding in visual working memory accounts for stimulus-specific
    variations in recall'
    http://www.bayslab.org/pdf/TayBay18.pdf

    That paper reanalyzes data from previous studies but also conducts new
    experiments. You can specify which dataset to load via the 'dataset'
    argument.
    """
    def __init__(self, setsize, *args, dataset='bays2014', reps=1, **kwargs):
        self.human_data = self._load_augmented_data(dataset, reps, setsize)
        dataset = f'{dataset}_setsize{setsize}'

        # Choose relative item sizes and spacings to exactly match those in
        # paper
        self.item_eccentricity = 0.25
        annulus_rad = 224 * self.item_eccentricity  # in pixels (assume image size 224)
        R = annulus_rad / np.sin(3 * np.pi / 180)
        self.line_length = int(np.round(R * np.sin(1 * np.pi / 180) * 2))
        self.line_thickness = int(np.round(R * np.sin(0.15 * np.pi / 180) * 2))

        super(TCCBays, self).__init__(
            setsize,
            *args,
            dataset=dataset,
            # I compared results using the actual stimuli as described in the
            # paper (colored lines) as well as monochrome gabors (sometimes
            # used in other papers)
            item_type='gabors' if 'gabors' in dataset else 'lines',
            **kwargs
        )

    def _load_augmented_data(self, dataset, reps, setsize):
        pth = Path(
            f'taylor_bays/{dataset}_setsize{setsize}_augmented_{reps}_samples.pkl'
        )
        if not pth.exists():
            human_data = load_taylor_bays(exp=dataset)
            human_data = human_data[human_data.K == setsize]
            # Because original dataset does not include info about item locations,
            # repeat trials and resample locations for each repitition
            human_data = pd.concat([human_data for i in range(reps)])
            item_locs = [
                tuple(
                    # 8 is maximum number of locations
                    np.random.choice(8, size=setsize, replace=False)
                )
                for i in range(len(human_data))
            ]
            human_data['item_locs'] = item_locs
            human_data['target_idx'] = [0] * len(human_data)
            # Save to file for loading later
            human_data.to_pickle(pth)
        else:
            # Use pickle load/save because we have tuple values in dataframe
            human_data = pd.read_pickle(pth)
        return human_data

    def _get_key_from_idx(self, idx):
        """
        Get stimulus look-up key (used in various dicts) from trial index
        corresponding to response data.
        """
        target = self.human_data.target.values[idx]
        nontargets = self.human_data.nontargets.values[idx]
        item_locs = self.human_data.item_locs.values[idx]
        target_idx = self.human_data.target_idx.values[idx]
        angles = list(nontargets)
        angles.insert(target_idx, target)
        angles = tuple(angles)
        target_loc = item_locs[target_idx]
        key = (angles, item_locs, target_loc, angles[target_idx])
        return key

    def _stimulus_id_iter(self, sims_dict={}):
        """
        Identify each stimulus image with a unique id tag and iterate through
        them. This scheme allows for caching precomputed values, as we can
        keep a dictionary of model embeddings, whose keys are the unique
        stimulus ids.
        """
        max_deg = 180
        if self.num_incr is None:
            self.num_incr = max_deg
        # Dataset does not include original positions of items, so marginalize
        # by sampling locations. Dataset can be augmented with additional
        # samples by concatenating copies of the dataset (see __init__).
        # Note that we assume 8 discrete locations
        for i in tqdm(range(len(self.human_data)), dynamic_ncols=True):
            target = self.human_data.target.values[i]
            nontargets = self.human_data.nontargets.values[i]
            item_locs = self.human_data.item_locs.values[i]
            target_idx = self.human_data.target_idx.values[i]
            angles = list(nontargets)
            angles.insert(target_idx, target)
            angles = tuple(angles)
            target_loc = item_locs[target_idx]
            Y_ids = []
            for j in range(max_deg):
                a = np.array(angles)
                a[0] = j
                Y_ids.append((tuple(a), item_locs))
            key = (angles, item_locs, target_loc, angles[target_idx])
            if key not in sims_dict.keys():
                x_id = (angles, item_locs)
                yield key, x_id, Y_ids

    def analyze(self, idxs=None, dprimes=None):
        if dprimes is None:
            mclass = self.models.model_class.values[idx]
            dprimes = self.dprimes[mclass]
        self._load_or_get_sims(idxs=idxs)
        self._load_or_fit_dprimes(idxs=idxs)
        errs = [self.sample_model_errors(dp, idxs=idxs) for dp in dprimes]
        return self.dprime_data, errs


class TCCBrady(TCCSetsize):
    """
    Analyze data from:
    Brady, T. F., & Alvarez, G. A. (2015). Contextual effects in visual working
    memory reveal hierarchically structured memory representations. Journal of
    vision, 15(15), 6-6.
    https://jov.arvojournals.org/article.aspx?articleid=2471226#109163949
    """
    def __init__(self, setsize, *args, **kwargs):
        dataset = 'brady_alvarez'
        reps = 1
        item_diam_to_annulus_rad_ratio = 60 / 110  # From Brady study
        annulus_rad = 224 * 0.25  # Mine (pixels)
        super(TCCBrady, self).__init__(
            setsize,
            *args,
            dataset=dataset,
            item_type='colors',
            lab_center=[70., 20., 38.],
            lab_radius=60.,
            sqsize=int(np.round(item_diam_to_annulus_rad_ratio * annulus_rad)),
            **kwargs
        )
        self.human_data = load_brady_alvarez()

    def _get_key_from_idx(self, idx):
        """
        Get stimulus look-up key (used in various dicts) from trial index
        corresponding to response data.
        """
        angles = (
            # Putting all three keys into one call to .loc coerces data type to
            # float! But doing separately keeps their int type.
            self.human_data.loc[idx, 'target1'],
            self.human_data.loc[idx, 'target2'],
            self.human_data.loc[idx, 'target3'],
        )
        # Items always at same locations. Here, indexes (0, 1, 2) correspond to
        # angles (0, 120, 240) deg along invisible anulus
        item_locs = (0. + 90., 120. + 90., 240. + 90.)  # Add 90 deg shift for correct visualization
        probe_loc = self.human_data.probe_loc.values[idx]
        key = (angles, item_locs, probe_loc, angles[probe_loc])
        return key

    def _stimulus_id_iter(self, sims_dict={}):
        """
        Identify each stimulus image with a unique id tag and iterate through
        them. This scheme allows for caching precomputed values, as we can
        keep a dictionary of model embeddings, whose keys are the unique
        stimulus ids.
        """
        max_deg = 360
        if self.num_incr is None:
            self.num_incr = max_deg
        for i in tqdm(range(len(self.human_data)), dynamic_ncols=True):
            key = self._get_key_from_idx(i)
            (angles, item_locs, probe_loc) = key[:3]
            Y_ids = []
            for j in range(max_deg):
                a = np.array(angles)
                a[probe_loc] = j  # Keep in range [0, 360)
                Y_ids.append((tuple(a), item_locs))
            if key not in sims_dict.keys():
                x_id = (angles, item_locs)
                yield key, x_id, Y_ids

    def analyze(self, idxs=None, dprimes=None):
        if dprimes is None:
            mclass = self.models.model_class.values[idx]
            dprimes = self.dprimes[mclass]
        self._load_or_get_sims(idxs=idxs)
        self._load_or_fit_dprimes(idxs=idxs)
        errs = [self.sample_model_errors(dp, idxs=idxs) for dp in dprimes]
        return self.dprime_data, errs


def _get_binned_averages(values_dict, bins, use_abs=True, transform=lambda x: x):
    """
    Bin by target value and average.
    'values_dict' is a dictionary whose keys are stimulus ids and values are
    the set of values to be binned and averaged (e.g. response errors).

    Note that we only bin according to the last value in the stimulus id tuple,
    which is the target angle on the response wheel.
    Thus, values_dict should be pre-filtered for the relevant subset of
    stimuli. E.g., for scene wheels stimulus set, one should pass in a subset
    corresponding to a single scene wheel.
    """
    values_binned = {bin_idx: None for bin_idx in range(1, len(bins))}
    for key, val in values_dict.items():
        bin_idx = np.digitize(key[-1], bins)  # Target value is key[-1]
        if bin_idx == len(bins):
            # Catch case where target value is equal to max_deg in case this
            # appeared by mistake in stimulus keys (due to bug in experiment)
            bin_idx = 1
        if values_binned[bin_idx] is not None:
            values_binned[bin_idx].extend(list(val))
        else:
            values_binned[bin_idx] = list(val)
    # Average within each bin
    values_binned1 = {
        key: transform(np.abs(list(val))).mean()
        if use_abs else transform(np.mean(list(val)))
        for key, val in values_binned.items()
    }
    return values_binned1


def _get_binned_averages_brady(errs_by_stim, bins):
    """
    Brady dataset needs to be handled differently than other VWM datasets,
    since we have complete data for all the unprobed items
    """
    errs = []
    # Every location is probed in each stimulus. Treat each response as
    # separate data item, i.e. three data points for each unique stimulus
    # display (48 * 3 = 144).
    for key, val in errs_by_stim.items():
        errs.append(np.abs(val).mean())
    errs_binned = {i: e for i, e in enumerate(errs)}
    return errs_binned


def human_setsize_analysis(
    cmd_args,
    dprimes,
    TCCClass,
    setsizes,
    *args,
    use_abs=True,
    num_boot=1000,
    **kwargs,
):
    """
    For each set-size, d', and model, collect data about model likelihood,
    error, and spearman rank correlations between model and human when binning
    by the target item's value.

    TODO: also include rank correlations when binning by stimulus id (i.e.
    do not marginalize over unprobed items, in order to get difficulty as
    function of the whole display).
    """

    err_str = '' if use_abs else '_signed_error'

    if TCCClass is TCCBays:
        # Symmetric, oriented objects repeat every 180 deg
        max_deg = 180
    else:
        max_deg = 360
    if TCCClass is TCCBrady:
        # For this dataset, we have response data for all items in each
        # display, so we have option to not marginalize over these values
        # (Note: If I return to this, it needs to be fixed so the ordering is
        # always the same for the 144 unique stimulus keys.)
        # bin_ave_func = _get_binned_averages_brady
        bin_ave_func = _get_binned_averages
    else:
        # For other datasets, we do not have complete data, so we should
        # marginalized over unprobed items
        bin_ave_func = _get_binned_averages
    # Bin target values and average for higher SNR
    binsize = 12
    if max_deg % binsize:
        raise Exception(
            'binsize should divide evenly into max_deg for even bin sizes.'
        )
    bins = np.arange(0, max_deg + 1, binsize)

    data = pd.DataFrame({
        'model_class': [],
        'model_name': [],
        'setsize': [],
        'dprime': [],
        'loglik': [],
        'mean_abs_err': [],
        f'spearman_r_binsize{binsize}': [],
        f'spearman_pval_binsize{binsize}': [],
        'conf_int95_lower': [],
        'conf_int95_upper': [],
    })

    for K in setsizes:
        tcc = TCCClass(
            K,
            *args,
            ngrid=cmd_args.num_grid,
            N=cmd_args.num_hist_samples,
            noise_type=cmd_args.noise_type,
            model_classes=cmd_args.model_classes,
            model_layers=cmd_args.model_layers,
            dprimes=dprimes,
            **kwargs
        )
        dataset = tcc.dataset
        df = tcc.human_data
        human_mean_abs_error = np.abs(df[df.K == K].error.values).mean()
        human_errs_by_stim = tcc.get_human_errors_by_key()
        human_errs_binned = bin_ave_func(
            human_errs_by_stim, bins, use_abs=use_abs
        )
        for mclass in cmd_args.model_classes:
            data_m = pd.DataFrame({
                'model_class': [],
                'model_name': [],
                'setsize': [],
                'dprime': [],
                'loglik': [],
                'mean_abs_err': [],
                f'spearman_r_binsize{binsize}': [],
                f'spearman_pval_binsize{binsize}': [],
                'conf_int95_lower': [],
                'conf_int95_upper': [],
            })
            for dprime in tcc.dprimes[mclass]:
                idxs = tcc.models[tcc.models.model_class == mclass].index
                for idx in idxs:
                    mname = tcc.models.name.values[idx]
                    load_pth = Path(
                        f'{MODEL_OUT_PATH}/setsize_analysis/{dataset}/summary_'
                        f'{mclass}_ss{K}_dp{dprime}_{mname}{err_str}.csv'
                    )
                    if load_pth.exists():
                        # Load summary from file to save time
                        row_data = pd.read_csv(load_pth, index_col=0)
                        print(f'Loaded summary data from {load_pth}')
                    else:
                        dprime_data, model_errs = tcc.analyze(
                            dprimes=[dprime], idxs=[idx]
                        )
                        df_sub = dprime_data[dprime_data.dprime == dprime]
                        likelihoods = {
                            df_sub.model_name.values[i]: df_sub.loglik.values[i]
                            for i in range(len(df_sub))
                        }
                        model_errs_binned = bin_ave_func(
                            model_errs[0][mname], bins, use_abs=use_abs
                        )
                        # Binned rank correlations (binning lowers noise by averaging)
                        rho, pval = spearmanr(
                            list(human_errs_binned.values()),
                            list(model_errs_binned.values())
                        )
                        row_data = {
                            'model_class': [mclass],
                            'model_name': [mname],
                            'setsize': [K],
                            'dprime': [dprime],
                            'loglik': [likelihoods[mname]],
                            'mean_abs_err': [
                                np.abs(np.concatenate(
                                    [e for e in model_errs[0][mname].values()]
                                )).mean()
                            ],
                            f'spearman_r_binsize{binsize}': [rho],
                            f'spearman_pval_binsize{binsize}': [pval],
                            'conf_int95_lower': [np.nan],
                            'conf_int95_upper': [np.nan],
                        }
                        row_data = pd.DataFrame(row_data)
                        load_pth.parent.mkdir(parents=True, exist_ok=True)
                        row_data.to_csv(load_pth)
                    data_m = pd.concat([data_m, row_data])
                TCC_SAMPLE = {}  # Clear to save memory
            data = pd.concat([data, data_m])

        conf_int_pth = Path(
            f'{MODEL_OUT_PATH}/confidence_intervals/{dataset}/setsize_{K}{err_str}.npy'
        )
        if conf_int_pth.exists():
            conf_lower, conf_upper = np.load(conf_int_pth)
        else:
            # Bootstrap analysis to get estimate of ceiling on rank correlations.
            # For each boostrap resample, compute rank correlation on abs error
            # with original data. Then get confidence interval on this correlation
            # coefficient.
            spearman_boot = []
            for i in tqdm(range(num_boot), desc='Bootstrap analysis', dynamic_ncols=True):
                boot_idxs = np.random.choice(
                    range(len(tcc.human_data)),
                    size=len(tcc.human_data),
                    replace=True
                )
                herrs_by_stim_i = tcc.get_human_errors_by_key(boot_idxs)
                herrs_binned_i = bin_ave_func(
                    herrs_by_stim_i, bins, use_abs=use_abs
                )
                spearman_boot.append(
                    spearmanr(
                        list(human_errs_binned.values()),
                        list(herrs_binned_i.values())
                    )
                )
            rhos = np.sort([sp[0] for sp in spearman_boot])
            conf_lower = rhos[int(round(num_boot * 0.05))]
            conf_upper = rhos[int(round(num_boot * 0.95))]
            conf_int_pth.parent.mkdir(parents=True, exist_ok=True)
            np.save(conf_int_pth, np.array([conf_lower, conf_upper]))

        # Append human data
        human_df = pd.DataFrame({
            'model_class': ['human'],
            'model_name': ['human'],
            'setsize': [K],
            'dprime': [np.nan],
            'loglik': [np.nan],
            'mean_abs_err': [human_mean_abs_error],
            f'spearman_r_binsize{binsize}': [np.nan],
            f'spearman_pval_binsize{binsize}': [np.nan],
            'conf_int95_lower': [conf_lower],
            'conf_int95_upper': [conf_upper],
        })
        human_df_pth = Path(
            f'{MODEL_OUT_PATH}/setsize_analysis/{dataset}/summary_human'
            f'_ss{K}{err_str}.csv'
        )
        human_df.to_csv(human_df_pth)
        data = pd.concat([data, human_df], ignore_index=True)
    return data


def brady_analysis(
    args,
    dprimes,
    use_abs=True,
    max_mem=64,
    num_cpus=None,
):
    """
    Tim Brady study:
    https://jov.arvojournals.org/article.aspx?articleid=2471226#109163949
    """
    model_data = human_setsize_analysis(
        args,
        dprimes,
        TCCBrady,
        [3],
        use_abs=use_abs,
        max_mem=max_mem,
        num_cpus=num_cpus,
    )


def taylor_bays_analysis(
    args,
    dprimes,
    dataset='bays2014',
    setsizes=[1, 2, 4, 8],
    reps=1,
    max_mem=64,
    use_abs=True,
    num_cpus=None,
):
    model_data = human_setsize_analysis(
        args,
        dprimes,
        TCCBays,
        setsizes,
        dataset=dataset,
        # To explain setsize effects, keep similarity scaling fixed (here we
        # just choose set-size 1, but could be any set-size)
        inherit_from=f'{dataset}_setsize1',
        reps=reps,
        max_mem=max_mem,
        use_abs=use_abs,
        num_cpus=num_cpus,
    )


def make_vae_args(beta=0.1, start_epoch=9, layer=None, pixel_only=True):

    args = Namespace(
        dataset_dir=TORCH_DATASET_DIR,
        ckpt_dir=CHECKPOINT_DIR,
        split='train-standard',
        input_size=256,
        beta=beta,
        start_epoch=start_epoch,
        end_epoch=0,
        start_epoch_decoder=-1,
        end_epoch_decoder=0,
        layer=layer,
        attend=False,
        pixel_only=pixel_only,
        pixel_weight=1.,
        lr=0.001,
        weight_decay=0.,
        batch_size=32,
        batch_schedule=None,
        download=False,
        small=False,
        dataset_vae="places365",
        dataset_decoder=None,
        ds_size=500000,
        ss_min=1,
        ss_max=1,
        optimal_attention=False,
        num_foci=0,
        decoder=False,
        task=None,
        delta=0,
        pchange=0.5,
    )
    return args


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--noise-type',
        type=str,
        default='gaussian',
        help='Kind of noise to apply in TCC model (gaussian or gumbel)'
    )
    parser.add_argument(
        '--model-classes',
        nargs='*',
        default=None,
        help='Which model classes to include in analysis.'
    )
    parser.add_argument(
        '--num-grid',
        type=int,
        default=48,
        help='Number of evenly spaced valued to test when fitting d-prime'
    )
    parser.add_argument(
        '--dprime-range',
        type=float,
        nargs=2,
        default=[0.5, 20],
        help='Min and max d-prime for grid search.'
    )
    parser.add_argument(
        '--num-hist-samples',
        type=int,
        default=8000,
        help='Number of samples to take when estimating histogram for '
        'likelihood.'
    )
    parser.add_argument(
        '--inherit-from',
        type=str,
        default=None,
        help='Which stimulus set to import d-prime fit data from.'
    )
    parser.add_argument(
        '--scene-wheels-summary',
        action='store_true',
        help='Do scene-wheel model comparison and get summary.'
    )
    parser.add_argument(
        '--taylor-bays',
        action='store_true',
        help='Do analysis with Taylor & Bays human data.'
    )
    parser.add_argument(
        '--brady-alvarez',
        action='store_true',
        help='Do analysis with Taylor & Bays human data.'
    )
    parser.add_argument(
        '--setsizes',
        nargs='*',
        type=int,
        default=[1, 2, 4, 8],
        help='Which set-sizes to run for setsize-effects analysis.'
    )
    parser.add_argument(
        '--lab-center',
        nargs=3,
        type=float,
        default=[60., 22., 14.],
        help='Center coordinate for color wheel in CIELAB space.'
    )
    parser.add_argument(
        '--lab-radius',
        type=float,
        default=52.,
        help='Radius of color wheel in CIELAB space.'
    )
    parser.add_argument(
        '--item-type',
        default='colors',
        help='Type of stimulus item to use in set-size displays.'
    )
    parser.add_argument(
        '--attention-method',
        default=None,
        help='What kind of attention model to apply (None for no attention.)'
    )
    parser.add_argument(
        '--pca-explained-var',
        type=float,
        default=1,
        help='This parameter is used in the attentional mechanism, which can be'
        ' added to models in order to "attend" to parts of the stimulus that '
        'are most useful for the task (e.g. when attending to one item in a '
        'multi-item array.'
    )
    parser.add_argument(
        '--nmf-comp',
        type=int,
        default=10,
        help='Number of components to use in non-negative matrix '
        'factorization model of attention.'
    )
    parser.add_argument(
        '--att-vae-beta',
        type=float,
        default=1.,
        help='Beta value in beta-VAE model of attention.'
    )
    parser.add_argument(
        '--num-setsize-increments',
        nargs='*',
        default=None,
        help='For each set-size examined (as specified using --setsize), '
        'specify number of evenly-spaced increments along the stimulus wheel'
        'to use when generating set-size stimuli (e.g. arrays of colored '
        'squares, where color values are specified by hue angle).'
    )
    parser.add_argument(
        '--model-layers',
        type=str,
        default='{}',
        help='Specify layers to include in analysis for any number of models.'
        ' Expects a string, which will be evaluated as a python expression in'
        ' dictionary form, where keys are model names and values are lists of '
        'layer indexes. (Overwrites defaults provided in definition of TCCModel.)'
    )
    parser.add_argument(
        '--max-mem',
        type=float,
        default=183,
        help='Maximum memory to use when caching model embeddings. When this '
        'amount is exceeded, new embeddings are no longer cached and are '
        'instead recomputed as needed. (In GB.)'
    )
    parser.add_argument(
        '--num-cpus',
        type=int,
        default=None,
        help='Number of CPUs to use in parallelized p_map function.'
    )
    args = parser.parse_args()

    if args.scene_wheels_summary:
        dprimes = {
            # Need different grid range for color-channels baseline model
            'rgb': np.linspace(300, 600, 4),
        }
        tcc = TCCSceneWheel(
            dprimes=dprimes,
            ngrid=args.num_grid,
            dp_min=args.dprime_range[0],
            dp_max=args.dprime_range[1],
            N=args.num_hist_samples,
            noise_type=args.noise_type,
            model_classes=args.model_classes,
            inherit_from=args.inherit_from,
            max_mem=args.max_mem,
            model_layers=args.model_layers,
            num_cpus=args.num_cpus,
        )
        df = tcc.summary()
        # WASN'T USEFUL
        # for rad in [2, 4, 8, 16, 32]:
        #     print(f'RADIUS={rad}')
        #     tcc.perceptual_similarity_analysis(
        #         df, args.model_classes, radius=rad
        #     )
    if args.brady_alvarez:
        dprimes_ss = [
            1., 1.5, 2., 2.5, 3., 3.5, 4.5, 5., 6., 7., 8., 9., 10., 12.5, 15., 17.5, 20.
        ]
        brady_analysis(args, dprimes_ss, num_cpus=args.num_cpus)
    if args.taylor_bays:
        dprimes_ss = {
            'vgg19': [
                1., 1.5, 2., 2.5, 3., 3.5, 4.5, 5., 6., 7., 8., 9., 10.,
                12.5, 15., 17.5, 20., 22.5, 25, 27.5, 30, 35, 40, 45, 50,
                60, 70, 80, 90, 100,
            ],
            'clip_RN50': [
                1., 1.5, 2., 2.5, 3., 3.5, 4.5, 5., 6., 7., 8., 9.,
                10., 12.5, 15., 17.5, 20., 22.5, 25, 27.5, 30, 35, 40, 45, 50,
                60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 400, 500,
                600, 700, 800, 900, 1000
            ],
            'clip_ViT-B16': [
                1., 1.5, 2., 2.5, 3., 3.5, 4.5, 5., 6., 7., 8., 9.,
                10., 12.5, 15., 17.5, 20., 22.5, 25, 27.5, 30, 35, 40, 45, 50,
                60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 400, 500,
                600, 700, 800, 900, 1000
            ]
        }
        taylor_bays_analysis(
            args,
            dprimes_ss,
            dataset='bays2014',  # Colored lines
            # dataset='bays2014_gabors',  # Test generality of results using gabors instead of lines
            setsizes=args.setsizes,
            max_mem=args.max_mem,
            num_cpus=args.num_cpus,
        )
