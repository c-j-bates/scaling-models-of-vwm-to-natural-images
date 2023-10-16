from itertools import product
import json
from mahotas.features import surf
import matplotlib.pyplot as plt
from networks import MLP
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.stats import spearmanr, circvar
import sys
from TCC_modeling import (
    standardize_angles,
    TCCSceneWheel,
    TCCBrady,
    TCCBays,
    make_and_load,
    make_vae_args,
    load_wheel_imgs,
    torchvision_embed,
    clip_embed,
    cossim,
    cossim_torch,
    sample_TCC,
    _get_binned_averages,
    rgb_from_angle,
    count_model_params,
)
import torch
from torchvision import transforms
from train import train_mlp
from tqdm import tqdm
import visualpriors

# Add CounTR local repository to path
sys.path.append(os.environ['COUNTR_PATH'])
# Pytorch device (global var)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DICT = {}
spth = os.environ.get('DATA_STORAGE')
if spth is not None:
    DATA_STORAGE = Path(spth).joinpath('psychophysical_scaling')
else:
    DATA_STORAGE = './'
# For analysis examining trial difficulty after controlling for response
# options (Note: was calling this "load analysis" but later switched to
# "difficulty".)
TRIAL_DIFFICULTY_PATH = Path(DATA_STORAGE).joinpath('load_analysis')

TCC_PATH = Path(DATA_STORAGE).joinpath('data_tcc')


def make_scene_wheel_df(wheels_pth, ext='webp'):
    radius_vals = [2, 4, 8, 16, 32]
    webp_imgs = {}
    size_uncomp = {}  # Uncompressed size
    size_on_disk = {}  # Compressed size
    df = pd.DataFrame({
        'wheel_id': [],
        'radius': [],
        'value': [],
        'disk_size': [],
        'uncomp_size': [],
    })
    for i in range(1, 6):
        pth = wheels_pth.joinpath(f"Wheel0{i}")
        webp_imgs[i + 1] = {}
        size_uncomp[i + 1] = {}
        size_on_disk[i + 1] = {}
        for r in radius_vals:
            pth1 = pth.joinpath(f'wheel0{i}_r{str(r).zfill(2)}')
            for j in range(360):
                pth2 = pth1.joinpath(f'{str(j).zfill(6)}.webp')
                img = Image.open(pth2)
                if ext == 'webp':
                    disk_size = os.stat(pth2).st_size
                else:
                    # img.convert(ext).save(f'_tmp.{ext}')  # Doesn't work due to bug...
                    # Hack it like this:
                    Image.fromarray(np.array(img)).save(f'_tmp.{ext}')
                    disk_size = os.stat(f'_tmp.{ext}').st_size
                uncomp_size = np.array(img).nbytes
                df = pd.concat([
                    df,
                    pd.DataFrame({
                        'file_path': [pth2],
                        'wheel_id': [i],
                        'radius': [r],
                        'value': [j],
                        'disk_size': [disk_size],
                        'uncomp_size': [uncomp_size],
                    })
                ])
    return df


def load_countr():
    from CounTR import models_mae_cross
    # Prepare model
    model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
    model.to(device)
    model_without_ddp = model
    weights_pth = './CounTR/weights/FSC147.pth'
    checkpoint = torch.load(weights_pth, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print("Resume checkpoint %s" % weights_pth)
    model.eval()
    return model


def estimate_setsize_scene_wheels(model, pth):
    """
    Use pre-trained object counting network to get estimated number of objects
    (essentially, set-size) for a scene wheel stimulus (loaded from `pth`).

    Using zero-shot inference mode of CounTR:
    https://github.com/Verg-Avesta/CounTR
    """

    def load_image(pth):
        image = Image.open(pth)
        image.load() 
        W, H = image.size
        # Resize the image size so that the height is 384
        new_H = 384
        new_W = 16*int((W/H*384)/16)
        scale_factor_H = float(new_H)/ H
        scale_factor_W = float(new_W)/ W
        image = transforms.Resize((new_H, new_W))(image)
        image = transforms.ToTensor()(image)
        return image

    def run_one_image(samples, model, boxes=[], pos=[]):
        _,_,h,w = samples.shape
        if boxes == []:
            boxes = torch.Tensor(boxes).unsqueeze(0).to(device, non_blocking=True)
        
        s_cnt = 0
        for rect in pos:
            if rect[2]-rect[0]<10 and rect[3] - rect[1]<10:
                s_cnt +=1

        density_map = torch.zeros([h,w])
        density_map = density_map.to(device, non_blocking=True)
        start = 0
        prev = -1
        with torch.no_grad():
            while start + 383 < w:
                # output, = model(samples[:,:,:,start:start+384], boxes, 3)
                output, = model(samples[:,:,:,start:start+384], boxes, 0)  # Zero-shot
                output=output.squeeze(0)
                b1 = torch.nn.ZeroPad2d(padding=(start, w-prev-1, 0, 0))
                d1 = b1(output[:,0:prev-start+1])
                b2 = torch.nn.ZeroPad2d(padding=(prev+1, w-start-384, 0, 0))
                d2 = b2(output[:,prev-start+1:384])            
                b3 = torch.nn.ZeroPad2d(padding=(0, w-start, 0, 0))
                density_map_l = b3(density_map[:,0:start])
                density_map_m = b1(density_map[:,start:prev+1])
                b4 = torch.nn.ZeroPad2d(padding=(prev+1, 0, 0, 0))
                density_map_r = b4(density_map[:,prev+1:w])
                density_map = density_map_l + density_map_r + density_map_m/2 + d1/2 +d2
                prev = start + 383
                start = start + 128
                if start+383 >= w:
                    if start == w - 384 + 128: break
                    else: start = w - 384
            pred_cnt = torch.sum(density_map/60).item()
            e_cnt = 0
            for rect in pos:
                e_cnt += torch.sum(density_map[rect[0]:rect[2]+1,rect[1]:rect[3]+1]/60).item()
            e_cnt = e_cnt / 3
            if e_cnt > 1.8:
                pred_cnt /= e_cnt
            return pred_cnt

    # Test on the new image
    samples = load_image(pth)
    samples = samples.unsqueeze(0).to(device, non_blocking=True)
    count = run_one_image(samples, model)
    return count


def segmentation_pca(preds):
    """
    For converting from Taskonomy's 2D and 2.5D segmentation to a 3-channel
    image.

    Taken from:
    https://github.com/StanfordVL/taskonomy/blob/master/taskbank/tools/task_viz.py

    See also:
    https://github.com/StanfordVL/taskonomy/blob/d486b5ecb7718531669a35d4fe3022a19c2bb377/code/tools/run_img_task.py
    """
    from sklearn.decomposition import PCA

    preds = np.squeeze(preds)
    preds_flat = preds.reshape((64, -1)).T
    pca = PCA(n_components=3)
    preds1 = pca.fit_transform(preds_flat).reshape((256, 256, -1))
    preds1 = (preds1 - preds1.min()) / (preds1.max() - preds1.min())
    preds1 = np.moveaxis(preds1, -1, 0)
    preds1 = torch.Tensor(preds1)
    # transforms.ToPILImage()(preds1).show()
    return preds1


def get_midvision_features(feature_type, pth, save=True):
    """
    Mid-level features come from:
    https://github.com/alexsax/midlevel-reps
    """
    feat_pth = Path(
        # f'midlevel_vision_features/{feature_type}/{pth}.npy'
        f'{TRIAL_DIFFICULTY_PATH}/midlevel_vision_features/{feature_type}/{pth}.npy'
    )
    if feat_pth.exists():
        Z = torch.Tensor(np.load(feat_pth))
    else:
        img = Image.open(pth)
        img = transforms.Compose([
            transforms.ToTensor(), transforms.Resize((256, 256))
        ])(img).unsqueeze(0) * 2 - 1
        Z = visualpriors.feature_readout(
            img.to(device), feature_type, device=device
        ).squeeze(0)
        feat_pth.parent.mkdir(parents=True, exist_ok=True)
        if save:
            np.save(feat_pth, Z.detach().cpu().numpy())
    return Z


def save_midvision_to_image(Z, out_pth, feature_type):
    if feature_type in ['segment_unsup2d', 'segment_unsup25d']:
        # Do PCA to get down to 3 channels
        Z = segmentation_pca(Z.detach().cpu().numpy())
    else:
        Z = (Z + 1.) / 2.
    Path(out_pth).parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(Z).save(out_pth)


def get_midvision_complexity(feature_type, pth, ext='jpg'):
    """
    Get 'complexity' of predictions from mid-level vision features by saving
    outputs as PIL image and reading out file size.

    """
    Z = get_midvision_features(feature_type, pth)
    if 'keypoints' in feature_type:
        # Summed activation as complexity
        complexity = float(Z.abs().sum().detach().cpu().numpy())
    else:
        out_pth = f'_tmp.{ext}'
        save_midvision_to_image(Z, out_pth, feature_type)
        complexity = os.stat(out_pth).st_size  # File size as complexity
    return complexity


def get_surf_complexity(pth):
    img = np.array(Image.open(pth).convert('L'))
    spoints = surf.surf(img)

    # Complexity as simple count of keypoints returned by SURF
    # complexity = len(spoints)

    # Complexity as sum of Hessian scores returned by SURF for each key point
    complexity = spoints[:, 3].sum()

    # Complexity as sum of scale sizes for each keypoint. Idea is to weight
    # larger scales more than smaller scales, if attention 'cares' more about
    # macroscopic objects
    # complexity = 1 / spoints[:, 2].sum()

    # Complexity as entropy of spatial distribution of keypoints, discretized
    # to grid
    # yi = spoints[:, 0].astype(int)
    # xi = spoints[:, 1].astype(int)
    # bins = np.arange(0, img.shape[0], 8)
    # xid = np.digitize(xi, bins)
    # yid = np.digitize(yi, bins)
    # ids = [str(x) for x in zip(xid, yid)]
    # counts = {(i, j): 1 for i, j in product(range(1, len(bins) + 1), repeat=2)}
    # for i, j in zip(xid, yid):
    #     counts[(i, j)] += 1
    # total = len(spoints)
    # probs = np.array(list(counts.values())) / total
    # entropy = (-np.log(probs) * probs).sum()
    # complexity = entropy

    return complexity


def get_torchvision_complexity(pth, model='vgg19', layer=33, ctype='mean_abs'):
    feat_pth = Path(
        f'{TRIAL_DIFFICULTY_PATH}/torchvision_features/{model}_{layer}/{pth}.npy'
        # f'data_tcc/load/torchvision_features/{model}_{layer}/{pth}.npy'
    )
    if feat_pth.exists():
        Z = np.load(feat_pth)
    else:
        img = Image.open(pth)
        X = [transforms.ToTensor()(img)]
        Z = torchvision_embed(X, layer, model, flatten=False)
        feat_pth.parent.mkdir(parents=True, exist_ok=True)
        np.save(feat_pth, np.array(Z))
    if ctype == 'mean_abs':
        complexity = np.abs(Z).mean()
    elif ctype == 'spatial_entropy':
        # Look at spatial distribution of activations. E.g., if high
        # activations are more evenly distributed across image, this might
        # mean greater attentional load in viewers.
        total = np.abs(Z).sum()
        n, m = Z.shape[-2:]
        probs = np.abs(Z).sum(axis=-3) / total
        probs[probs == 0] = 1e-8
        entropy = (-np.log(probs) * probs).sum()
        complexity = entropy
    else:
        raise NotImplementedError()
    # if layer == 36:
    #     print(f'Prop. units = 0: {(Z == 0).sum() / Z.size}')
    #     fig, ax = plt.subplots()
    #     ax.hist(Z.reshape(-1), bins=50)
    #     plt.show()
    #     from ipdb import set_trace; set_trace()
    return complexity


def load_vae(tcc, beta):
    idx_vae = tcc.models[tcc.models.name == f'vae_beta{beta}'].index.values[0]
    epoch = eval(tcc.models.specs.values[idx_vae])[-1]
    vae_model = make_and_load(make_vae_args(
        beta=beta,
        start_epoch=epoch,
        pixel_only=True,
    ))[0].eval()
    return vae_model


def get_vae_complexity(pth, tcc, beta=0.01):
    feat_pth = Path(f'{TRIAL_DIFFICULTY_PATH}/vae_features/beta{beta}/{pth}.npy')
    if feat_pth.exists():
        Z = torch.Tensor(np.load(feat_pth))
    else:
        name = f'vae_beta{beta}'
        if name in MODELS_DICT.keys():
            vae_model = MODELS_DICT[name]
        else:
            vae_model = load_vae(tcc, beta)
            MODELS_DICT[name] = vae_model

        # Get features
        img = Image.open(pth)
        preproc = transforms.Compose([
            transforms.Resize((tcc.imgsize_vae, tcc.imgsize_vae)),
            transforms.ToTensor(),
        ])
        X = preproc(img).unsqueeze(0)
        att_map = torch.zeros(
            (len(X), tcc.imgsize_vae, tcc.imgsize_vae)
        ).float()
        with torch.no_grad():
            Z, _, log_var, _ = vae_model(X.to(device), att_map.to(device))
        Z = Z.detach().cpu().numpy()
        feat_pth.parent.mkdir(parents=True, exist_ok=True)
        np.save(feat_pth, Z)
    complexity = np.abs(Z).mean()
    return complexity


def get_visual_complexity_data(bin_size, ext='jpg'):
    df_pth = Path(TRIAL_DIFFICULTY_PATH).joinpath('load_measures.csv')
    if df_pth.exists():
        df_binned = pd.read_csv(df_pth)
        print(f'Loaded dataframe from {df_pth}')
    else:
        wheels_pth = Path('scene_wheels_mack_lab_osf').joinpath(
            'scene_wheel_images', 'sceneWheel_images_webp'
        )
        df = make_scene_wheel_df(wheels_pth, ext=ext)
        tcc = TCCSceneWheel()

        # Load CounTR (for estimating object counts in natural images)
        countr_model = load_countr()

        dfh = tcc.human_data
        df_binned = pd.DataFrame({
            'wheel_id': [],
            'radius': [],
            'bin_left': [],
            'mean_error': [],
            'mean_disk_size': [],
        })
        for name, group in tqdm(df.groupby(['wheel_id', 'radius'])):
            for bin_ in np.arange(0, 360, bin_size):
                rng = range(bin_, bin_ + bin_size)
                mean_disk_size = group[group.value.isin(rng)].disk_size.mean()
                mean_uncomp_size = group[group.value.isin(rng)].uncomp_size.mean()
                dfh_g = dfh[(dfh.wheel_num == name[0]) & (dfh.radius == name[1])]
                mean_error = dfh_g[dfh_g.answer.isin(rng)].error.abs().mean()

                # VAE
                vae = [
                    get_vae_complexity(pth, tcc, beta=0.01)
                    for pth in group[group.value.isin(rng)].file_path.values
                ]

                # VGG19
                vgg19_layers = [4, 8, 17, 26, 33, 35]
                vgg19_mean_abs = {
                    layer: [
                        get_torchvision_complexity(
                            pth, model='vgg19', layer=layer, ctype='mean_abs'
                        )
                        for pth in group[group.value.isin(rng)].file_path.values
                    ]
                    for layer in vgg19_layers
                }
                vgg19_spat_ent = {
                    layer: [
                        get_torchvision_complexity(
                            pth, model='vgg19', layer=layer, ctype='spatial_entropy'
                        )
                        for pth in group[group.value.isin(rng)].file_path.values
                    ]
                    for layer in vgg19_layers
                }

                # CounTR
                count_estimates = [
                    estimate_setsize_scene_wheels(countr_model, pth)
                    for pth in group[group.value.isin(rng)].file_path.values
                ]

                # Taskonomy
                kp2d = [
                    get_midvision_complexity('keypoints2d', pth, ext=ext)
                    for pth in group[group.value.isin(rng)].file_path.values
                ]
                kp3d = [
                    get_midvision_complexity('keypoints3d', pth, ext=ext)
                    for pth in group[group.value.isin(rng)].file_path.values
                ]
                seg2d = [
                    get_midvision_complexity('segment_unsup2d', pth, ext=ext)
                    for pth in group[group.value.isin(rng)].file_path.values
                ]
                seg25d = [
                    get_midvision_complexity('segment_unsup25d', pth, ext=ext)
                    for pth in group[group.value.isin(rng)].file_path.values
                ]
                # TODO
                # segsem = [
                #     get_midvision_complexity('segment_semantic', pth, ext=ext)
                #     for pth in group[group.value.isin(rng)].file_path.values
                # ]

                # surf_count = [
                #     get_surf_complexity(pth)
                #     for pth in group[group.value.isin(rng)].file_path.values
                # ]

                df_binned = pd.concat([
                    df_binned,
                    pd.DataFrame({
                        'wheel_id': [name[0]],
                        'radius': [name[1]],
                        'bin_left': [bin_],
                        'mean_error': [mean_error],
                        'mean_disk_size': [mean_disk_size],
                        'vae_beta0.01': np.mean(vae),
                        'countr_estimate': np.mean(count_estimates),
                        'keypoints_2d': np.mean(kp2d),
                        'keypoints_3d': np.mean(kp3d),
                        # 'surf_count': np.mean(surf_count),
                        'seg_2d_mean': np.mean(seg2d),
                        'seg_25d_mean': np.mean(seg25d),
                        'vgg19_l4_mean_abs': np.mean(vgg19_mean_abs[4]),
                        'vgg19_l8_mean_abs': np.mean(vgg19_mean_abs[8]),
                        'vgg19_l17_mean_abs': np.mean(vgg19_mean_abs[17]),
                        'vgg19_l26_mean_abs': np.mean(vgg19_mean_abs[26]),
                        'vgg19_l33_mean_abs': np.mean(vgg19_mean_abs[33]),
                        'vgg19_l35_mean_abs': np.mean(vgg19_mean_abs[35]),
                        'vgg19_l4_spatial_entropy': np.mean(vgg19_spat_ent[4]),
                        'vgg19_l8_spatial_entropy': np.mean(vgg19_spat_ent[8]),
                        'vgg19_l17_spatial_entropy': np.mean(vgg19_spat_ent[17]),
                        'vgg19_l26_spatial_entropy': np.mean(vgg19_spat_ent[26]),
                        'vgg19_l33_spatial_entropy': np.mean(vgg19_spat_ent[33]),
                        'vgg19_l35_spatial_entropy': np.mean(vgg19_spat_ent[35]),
                    })
                ])
        df_binned.to_csv(df_pth)
        print(f'Saved dataframe to {df_pth}')
    return df_binned


def stimulus_difficulty_analysis(
    bin_size=30, ext='jpg', reg_type='linear', shuffle_baseline=False
):
    def get_plot_data(df, measures_set, percentiles, dnn_models=[]):
        pdata = {}
        for measure in measures_set:
            bins = np.percentile(
                df.human_mean_error.values.astype(float), percentiles
            )
            bins[-1] += 1e-6  # Hack to make sure last bin handled correctly
            bin_ids = np.digitize(df.human_mean_error, bins)
            pdata[measure] = [
                # Bin mean of complexity measure
                [
                    df[(bin_ids == idx) & (~np.isnan(df[measure]))][measure].mean()
                    for idx in range(1, len(percentiles))
                ],
                # Human mean error
                [
                    df[(bin_ids == idx) & (~np.isnan(df[measure]))].human_mean_error.mean()
                    for idx in range(1, len(percentiles))
                ],
            ] + [
                # Model errors
                [
                    df[(bin_ids == idx) & (~np.isnan(df[measure]))][f'{dnn}_mean_error'].mean()
                    for idx in range(1, len(percentiles))
                ]
                for dnn in dnn_models
            ]
        return pdata

    def _do_sklearn_regression(df, df_train, min_feat=1, iter_=0):        
        from sklearn.linear_model import LinearRegression
        from sklearn.feature_selection import RFECV
        model = LinearRegression()
        rfe = RFECV(model, min_features_to_select=min_feat)
        idxs = df_train.groupby(['wheel_id', 'radius']).sample(
            frac=0.5, replace=False).index
        X_train = df_train.loc[idxs, measures_set]
        y_train = df_train.loc[idxs].mean_error
        # Convert to log-odds in order for target to be in (-inf, inf)
        y_train = np.log(y_train / (180 - y_train))  # 180 deg is max error
        rfe.fit(X_train, y_train)
        X = df.loc[~df.index.isin(idxs), measures_set]
        pred = rfe.predict(X)
        pred1 = np.exp(pred) / (1 + np.exp(pred)) * 180
        df.loc[~df.index.isin(idxs), f'reg_preds_iter{iter_}'] = pred1
        return idxs, rfe.ranking_

    def _plot_reg_results(
        out_pth,
        df,
        radii=[2, 4, 8, 16, 32],
        num_iters=1,
        dnn_models=['clip_RN50', 'clip_ViT-B16', 'vgg19'],
    ):
        # Plotting binned by percentile
        out_pth.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(3 * 3, 3 * 2))
        for idx, (irow, icol) in enumerate(product(range(2), range(3))):
            if idx > 4:
                break
            ax = axes[irow, icol]
            rad_set = radii[idx:idx + 1]
            dfr = df[df.radius.isin(rad_set)]
            pdatas = [
                get_plot_data(
                    dfr,
                    [f'reg_preds_iter{it}'],
                    percentiles,
                    dnn_models=dnn_models,
                )
                for it in range(num_iters)
            ]
            human_mean_err = np.array([list(pdata.values())[0][1] for pdata in pdatas])
            reg_mean_err = np.array([list(pdata.values())[0][0] for pdata in pdatas])
            model_mean_errs = [
                np.array([list(pdata.values())[0][2 + i_mod] for pdata in pdatas])
                for i_mod in range(len(dnn_models))
            ]
            h_mean = human_mean_err.mean(axis=0)
            h_stderrs = human_mean_err.std(axis=0) / np.sqrt(num_iters)
            reg_mean = reg_mean_err.mean(axis=0)
            reg_stderrs = reg_mean_err.std(axis=0) / np.sqrt(num_iters)
            ax.errorbar(
                range(1, 5),
                reg_mean,
                yerr=reg_stderrs,
                label='Reg. model'
            )
            ax.errorbar(
                range(1, 5),
                h_mean,
                yerr=h_stderrs,
                label='Human',
                color='black',
                linestyle='--'
            )
            for i_mod in range(len(dnn_models)):
                m_mean = model_mean_errs[i_mod].mean(axis=0)
                m_stderrs = model_mean_errs[i_mod].std(axis=0) / np.sqrt(num_iters)
                ax.errorbar(
                    range(1, 5),
                    m_mean,
                    yerr=m_stderrs,
                    label=f'{dnn_models[i_mod]}',
                )
            axes[1, 2].errorbar(
                range(1, 5),
                reg_mean,
                yerr=reg_stderrs,
                label=f'radius={radii[idx]}'
            )
            ax.set_title(f'Radius: {radii[idx]}')
        axes[0, 0].legend()
        axes[1, 0].set_xlabel('Human-mean-error quartile')
        axes[1, 0].set_ylabel('Mean error (deg)')
        axes[1, 2].legend()
        plt.tight_layout()
        plt.savefig(out_pth)

    df = get_visual_complexity_data(bin_size=bin_size, ext=ext)
    tcc = TCCSceneWheel()
    human_errs = tcc.get_human_errors_by_key()
    bins = np.arange(0, 360 + 1, bin_size)

    model_errs = {
        'clip_RN50': tcc.get_or_load_TCC_samples(
            np.argmax(tcc.models.name == 'clip_RN50_l24'),
            4.23404255319149
        )[1],
        'clip_ViT-B16': tcc.get_or_load_TCC_samples(
            np.argmax(tcc.models.name == 'clip_ViT-B16_l12'),
            3.8191489361702127
        )[1],
        'vgg19': tcc.get_or_load_TCC_samples(
            np.argmax(tcc.models.name == 'vgg19_l30'),
            4.648936170212766
        )[1],
    }

    binaves = {}
    for wid in range(1, 6):
        for r in [2, 4, 8, 16, 32]:
            binaves[(wid, r)] = {
                'clip_RN50': _get_binned_averages(
                    {
                        key: val for key, val in model_errs['clip_RN50'].items()
                        if key[0] == wid and key[1] == r
                    },
                    bins
                ),
                'clip_ViT-B16': _get_binned_averages(
                    {
                        key: val for key, val in model_errs['clip_ViT-B16'].items()
                        if key[0] == wid and key[1] == r
                    },
                    bins
                ),
                'vgg19': _get_binned_averages(
                    {
                        key: val for key, val in model_errs['vgg19'].items()
                        if key[0] == wid and key[1] == r
                    },
                    bins
                ),
                'human': _get_binned_averages(
                    {
                        key: val for key, val in human_errs.items()
                        if key[0] == wid and key[1] == r
                    },
                    bins
                ),
            }
    df['clip_RN50_mean_error'] = None
    df['clip_ViT-B16_mean_error'] = None
    df['vgg19_mean_error'] = None
    df['human_mean_error'] = None
    for wid in range(1, 6):
        for r in [2, 4, 8, 16, 32]:
            for i, bin_left in enumerate(bins[:-1]):
                df.loc[
                    (df.bin_left == bin_left) & (df.wheel_id == wid) & (df.radius == r),
                    'clip_RN50_mean_error'
                ] = binaves[(wid, r)]['clip_RN50'][i + 1]
                df.loc[
                    (df.bin_left == bin_left) & (df.wheel_id == wid) & (df.radius == r),
                    'clip_ViT-B16_mean_error'
                ] = binaves[(wid, r)]['clip_ViT-B16'][i + 1]
                df.loc[
                    (df.bin_left == bin_left) & (df.wheel_id == wid) & (df.radius == r),
                    'vgg19_mean_error'
                ] = binaves[(wid, r)]['vgg19'][i + 1]
                df.loc[
                    (df.bin_left == bin_left) & (df.wheel_id == wid) & (df.radius == r),
                    'human_mean_error'
                ] = binaves[(wid, r)]['human'][i + 1]

    percentiles = [0, 25, 50, 75, 100]
    measures_set_full = [
        'radius',
        'mean_disk_size',
        'vae_beta0.01',
        'countr_estimate',
        'keypoints_2d',
        'keypoints_3d',
        'seg_2d_mean',
        'seg_25d_mean',
        'vgg19_l4_mean_abs',
        'vgg19_l8_mean_abs',
        'vgg19_l17_mean_abs',
        'vgg19_l26_mean_abs',
        'vgg19_l33_mean_abs',
        'vgg19_l35_mean_abs',
        'vgg19_l4_spatial_entropy',
        'vgg19_l8_spatial_entropy',
        'vgg19_l17_spatial_entropy',
        'vgg19_l26_spatial_entropy',
        'vgg19_l33_spatial_entropy',
        'vgg19_l35_spatial_entropy',
    ]
    measures_set = list(measures_set_full)  # Make a copy

    # Heat-map of pairwise correlations between measures
    y = np.empty((len(measures_set), len(measures_set)))
    for i, m1 in enumerate(measures_set):
        for j, m2 in enumerate(measures_set):
            y[i, j] = spearmanr(df[m1], df[m2])[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks(range(len(measures_set)))
    ax.set_yticklabels(measures_set)
    handle = ax.imshow(y, cmap='Blues')
    plt.subplots_adjust(left=0.3)
    fig.colorbar(handle, ax=ax)
    plot_pth = Path('figures').joinpath(
        'summary',
        'scene_wheels',
        f'feature_correlations.pdf'
    )
    plt.savefig(plot_pth)

    # Multiple regression
    radii = [2, 4, 8, 16, 32]  # Fit regression model to these radii
    df_train = df[df.radius.isin(radii)]
    regression_data_pth = Path(
        f'{TRIAL_DIFFICULTY_PATH}/regression_results_{reg_type}.json'
    )
    min_feat = 1
    num_iters = 1000
    reg_data = [
        _do_sklearn_regression(df, df_train, min_feat=min_feat, iter_=i)
        for i in range(num_iters)
    ]
    sample_idxs, rankings = list(zip(*reg_data))
    rankings = np.array(rankings)
    fig, ax = plt.subplots()
    ax.bar(np.arange(rankings.shape[1]), (rankings == 1).sum(0) / num_iters)
    ax.set_xticks(np.arange(rankings.shape[1]))
    ax.set_xticklabels(measures_set, rotation=45, ha='right')
    plot_pth = Path('figures').joinpath(
        'summary',
        'scene_wheels',
        f'feature_keep_frequencies_min_feat{min_feat}.pdf'
    )
    plt.tight_layout()
    plt.savefig(plot_pth)
    plot_pth = Path('figures').joinpath(
        'summary',
        'scene_wheels',
        f'load_sklearn_RFECV_min_feat{min_feat}.pdf'
    )
    _plot_reg_results(
        plot_pth,
        df,
        num_iters=num_iters,
    )
    # print(rfe.ranking_)
    # print([spearmanr(df[df.radius == r].reg_preds_iter0, df[df.radius == r].mean_error) for r in radii])
    return


def collate_summaries_scene_wheels(models):
    root = Path(TCC_PATH).joinpath('scene_wheels_analysis')
    dfs = []
    for model in models:
        dfs.extend([
            pd.read_csv(pth)
            for pth in root.glob(f'summary_{model}_*.csv')
        ])
    return pd.concat(dfs, ignore_index=True)


def collate_summaries_bays(models, setsizes=[1, 2, 4, 8]):
    root = Path(TCC_PATH).joinpath('setsize_analysis')
    dfs = []
    for ss in setsizes:
        subroot = root.joinpath(f'bays2014_setsize{ss}')
        for model in models:        
            summaries = [
                pd.read_csv(pth)
                for pth in subroot.glob(f'summary_{model}_*.csv')
                if 'signed_error' not in str(pth)  # Holdover from before
            ]
            dfs.extend(summaries)
    return pd.concat(dfs, ignore_index=True)


def collate_summaries_brady(models):
    root = Path(TCC_PATH).joinpath('setsize_analysis', 'brady_alvarez')
    dfs = []
    for model in models:
        dfs.extend([
            pd.read_csv(pth)
            for pth in root.glob(f'summary_{model}_*.csv')
        ])
    return pd.concat(dfs, ignore_index=True)


def plot_taylor_bays_setsize(
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
):
    best_layers, best_dps = get_best_fit_models(
        'bays2014_all',
        model_classes=model_classes,
    )
    df = collate_summaries_bays(model_classes + ['human'])
    df = df.sort_values(by=['setsize']).sort_index()
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    dfh = df[df.model_name == 'human']
    fig, ax = plt.subplots(figsize=(3, 3))  # Single plot
    df_plot = pd.DataFrame({
        'label': ['human'] * len(dfh.setsize),
        'x': dfh.setsize,
        'y': dfh.mean_abs_err,
    })
    ax.plot(dfh.setsize, dfh.mean_abs_err, label='human', linestyle='--')
    for i, mclass in enumerate(model_classes):
        dfm = df[df.model_class == mclass]
        dfm_best = dfm[(dfm.dprime == best_dps[i]) & (dfm.model_name == best_layers[i])]
        ax.plot(
            dfm_best.setsize,
            dfm_best.mean_abs_err,
            label=f"{mclass}"
        )
        df_plot = pd.concat([
            df_plot,
            pd.DataFrame({
                'label': [mclass] * len(dfm_best.setsize),
                'x': dfm_best.setsize,
                'y': dfm_best.mean_abs_err,
            })
        ])
    ax.legend()
    ax.set_xlabel('Set size')
    ax.set_ylabel('Mean abs error')
    plt.tight_layout()
    plt.savefig(f'figures/summary/taylor_bays/error_per_setsize.pdf')
    return df_plot


def plot_taylor_bays_sim_curves(
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
):
    best_layers, best_dps = get_best_fit_models(
        'bays2014_all',
        model_classes=model_classes,
    )
    # best_layers[-1] = 'vgg19_l19'  # DEBUG
    fig, ax = plt.subplots(
        ncols=len(model_classes),
        figsize=(2.5 * len(model_classes), 3)
    )
    df_plot = pd.DataFrame({
        'ss': [],
        'model_class': [],
        'x': [],
        'y': [],
    })
    setsizes = [1, 2, 4, 8]
    sims = {}
    for i, mclass in enumerate(model_classes):
        sims[mclass] = {}
        for ss in setsizes:
            item_locs = None  # Even spacing
            tcc = TCCBays(
                ss,
                model_classes=[mclass],
                just_in_time_loading=True,
                inherit_from=f'bays2014_setsize1',  # Shouldn't actually matter
            )
            layer_idx = np.argmax(tcc.models.name == best_layers[i])
            targets = [0] * ss
            responses = [[0] * (ss - 1) + [i] for i in range(180)]
            target_id = (tuple(targets), item_locs)
            resp_ids = [(tuple(r), item_locs) for r in responses]
            # Get model info
            if mclass.startswith('clip_RN50'):
                embed_func = clip_embed
                img_size = 224
                embed_args = (tcc.models.specs.values[layer_idx], 'RN50')
            elif mclass.startswith('clip_ViT-B16'):
                embed_func = clip_embed
                img_size = 224
                embed_args = (tcc.models.specs.values[layer_idx], 'ViT-B16')
            elif mclass in ['vgg19', 'resnet50']:
                embed_func = torchvision_embed
                img_size = 224
                embed_args = (tcc.models.specs.values[layer_idx], mclass)
            else:
                raise NotImplementedError()
            # Compute model embeddings
            # (Note: Averaging over samples, because stimuli are colored lines
            # , where colors are randomly sampled each time they are created.
            # The similarity curves look notably different per sample.)
            Z = []
            for k in range(25):
                Z.append(tcc._embed(
                    best_layers[i],
                    target_id,
                    resp_ids,
                    embed_func,
                    img_size,
                    *embed_args
                ))
            Z = np.mean(Z, axis=0)
            print(f'Norm ({mclass}, ss={ss}): {np.linalg.norm(Z[0])}')
            # Compute similarities
            sims[mclass][ss] = cossim_torch(Z[0], Z[1:], dev=device).detach().cpu().numpy()
            ax[i].plot(sims[mclass][ss], label=f'ss={ss}')
            ax[i].set_title(f'{mclass}')
            ax[i].set_xlabel(f'Response angle')
            df_plot = pd.concat([
                df_plot,
                pd.DataFrame({
                    'ss': [ss] * 180,
                    'model_class': [mclass] * 180,
                    'x': np.arange(0, 180),
                    'y': sims[mclass][ss],
                })
            ])
    ax[0].set_ylabel(f'Cosine similarity to target')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(f'figures/summary/taylor_bays/similarity_curves_by_setsize.pdf')
    return df_plot


def plot_taylor_bays_spearman_full(
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
):
    df = collate_summaries_bays(model_classes)
    df = df.sort_values(by=['setsize']).sort_index()
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    fig, ax = plt.subplots(
        nrows=len(model_classes),
        ncols=len(np.unique(df.setsize)),
        figsize=(len(np.unique(df.setsize)) * 2.5, 4.5 * len(model_classes)),
        squeeze=False,
    )
    min_rho = df[f'spearman_r'].min()
    for i, mclass in enumerate(model_classes):
        dfm = df[df.model_class == mclass]
        # Take best d' for each layer
        best_dps = {
            mname: dfm.groupby(['dprime'])['loglik'].mean().idxmax()
            for mname, group in dfm.groupby('model_name')
        }
        dfm_best = pd.concat([
            dfm[(dfm.model_name == mname) & (dfm.dprime == dp)]
            for mname, dp in best_dps.items()
        ])

        # for j, ss in enumerate(['1', '2', '4', '8', 'all']):
        for j, (ss, group) in enumerate(dfm_best.groupby('setsize')):
            group = dfm_best[dfm_best.setsize == ss]
            colors = [
                'gray' if pval > 0.05 else 'b'
                for pval in group[f'spearman_pval']
            ]
            ax[i][j].bar(
                range(len(group)),
                group[f'spearman_r'],
                color=colors
            )
            ax[i][j].set_xticks([])
            ax[i][j].set_xticklabels([])
            ax[i][j].set_xlabel(f'{mclass} layers')
            ax[i][j].set_ylim(min_rho - min_rho * 0.05, 1)
            if i == 0:
                ax[i][j].set_title(f'Set-size={int(ss)}')
    ax[0, 0].set_ylabel('Spearman rho')
    plt.tight_layout()
    plt.savefig(f'figures/summary/taylor_bays/spearman_bars_orientation.pdf')


def plot_taylor_bays_spearman_condensed(
    data_pth,
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
    pca=1,
):
    pca_str = f'_pca{pca}' if pca < 1 else ''
    df = pd.read_csv(data_pth)
    df = df.sort_values(by=['setsize']).sort_index()
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    setsizes = [1, 2, 4, 8]
    fig, ax = plt.subplots(
        ncols=len(setsizes), figsize=(1.5 * len(setsizes), 4)
    )
    for i, ss in enumerate(setsizes):
        ax[i].set_ylim(-0.5, 0.5)
        ax[i].set_title(f'Set-size: {ss}')
        ax[i].set_xticks(np.arange(len(model_classes)) * 0.5)
        ax[i].set_xticklabels(model_classes, ha='right', rotation=45)
        ax[i].spines[['right', 'top']].set_visible(False)
        if i != 0:
            ax[i].set_yticklabels([])
    dfp = get_spearman_condensed_ss_plot_data(
        df,
        setsizes,
        model_classes=model_classes,
    )

    for i in range(len(ax)):
        dfpi = dfp[dfp.axis == i]
        colors = [
            'gray' if pval > 0.05 else 'b'
            for pval in dfpi.pval
        ]
        ax[i].bar(dfpi.x, dfpi.y, width=0.25, label=dfpi.name, color=colors)

    ax[0].set_ylabel('Spearman rho')
    plt.tight_layout()
    plt.savefig(f'figures/summary/taylor_bays/spearman_bars_condensed.pdf')


def plot_taylor_bays_quantiles(
    data_pth,
    dataset,
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
    num_boot=1000,
    pca=1,
):
    def bootstrap(dfh, num_boot, mclass):
        model_bin_means = [
            dfh.groupby('bins').sample(frac=1, replace=True).groupby('bins')[f'{mclass}_err'].mean()
            for iboot in range(num_boot)
        ]
        human_bin_means = [
            dfh.groupby('bins').sample(frac=1, replace=True).groupby('bins')['abs_error'].mean()
            for iboot in range(num_boot)
        ]
        # Confidence intervals
        m_boot_means = np.mean(model_bin_means, axis=0)
        m_boot_lower = np.quantile(model_bin_means, q=0.05, axis=0)
        m_boot_upper = np.quantile(model_bin_means, q=0.95, axis=0)
        h_boot_means = np.mean(human_bin_means, axis=0)
        h_boot_lower = np.quantile(human_bin_means, q=0.05, axis=0)
        h_boot_upper = np.quantile(human_bin_means, q=0.95, axis=0)
        return (
            m_boot_means,
            m_boot_lower,
            m_boot_upper,
            h_boot_means,
            h_boot_lower,
            h_boot_upper,
        )

    pca_str = f'_pca{pca}' if pca < 1 else ''
    df = pd.read_csv(data_pth)
    df = df.sort_values(by=['setsize']).sort_index()
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    setsizes = [1, 2, 4, 8]
    num_quantiles = 4
    dfh_all = []  # Collect data from all set-sizes
    for ss in setsizes:
        tcc = TCCBays(
            ss,
            model_classes=model_classes,
            just_in_time_loading=True,
            # max_mem=370,
            inherit_from=f'{dataset}_setsize1',
        )
        fig, ax = plt.subplots(
            ncols=len(model_classes),
            figsize=(2.5 * len(model_classes), 3)
        )
        if not hasattr(ax, '__len__'):
            ax = [ax]
        dfh = tcc.human_data
        dfh['abs_error'] = dfh.error.abs()
        dfh['bins'] = pd.qcut(dfh.abs_error, num_quantiles)
        # Get model errors for each trial and compute corresponding means
        for i, mclass in enumerate(model_classes):
            # dfm = df[df.model_name.str.contains(mclass + '_l')]
            dfm = df[df.model_class == mclass]
            best_layer, best_dp = dfm.groupby(['model_name', 'dprime'])['loglik'].mean().idxmax()
            layer_idx = np.argmax(tcc.models.name == best_layer)
            errs = tcc.sample_model_errors(best_dp, idxs=[layer_idx])[best_layer]
            mean_abs_model_error = []
            for j in range(len(dfh)):
                key = tcc._get_key_from_idx(j)
                mean_abs_model_error.append(np.abs(errs[key]).mean())
            dfh[f'{mclass}_err'] = mean_abs_model_error
            dfh_all.append(dfh)
            x = bootstrap(dfh, num_boot, mclass)
            (
                m_boot_means,
                m_boot_lower,
                m_boot_upper,
                h_boot_means,
                h_boot_lower,
                h_boot_upper,
            ) = x
            # print(f'Model: mean={m_boot_means}, lower={m_boot_lower}, upper={m_boot_upper}')
            # print(f'Humans: mean={h_boot_means}, lower={h_boot_lower}, upper={h_boot_upper}')

            # No bootstrap
            # ax[i].plot(dfh.groupby('bins')[f'{mclass}_err'].mean(), human_qmeans)
            # With bootstrap
            ax[i].errorbar(
                m_boot_means,
                h_boot_means,
                xerr=[m_boot_means - m_boot_lower, m_boot_upper - m_boot_means],
                yerr=[h_boot_means - h_boot_lower, h_boot_upper - h_boot_means],
            )
            ax[i].set_title(f"{mclass}, d'={best_dp}")
            print(f'Setsize: {ss}, ', best_layer, dfh[f'{mclass}_err'].mean())
        ax[0].set_xlabel('Model mean abs error (deg)')
        ax[0].set_ylabel('Human mean abs error (deg)')
        plt.tight_layout()
        plt.savefig(f'figures/summary/taylor_bays/error_quantiles_ss{ss}.pdf')

    # Now bootstrap all set-sizes together
    dfh_all = pd.concat(dfh_all)
    dfh_all['bins'] = pd.qcut(dfh_all.abs_error, num_quantiles)
    fig, ax = plt.subplots(
        ncols=len(model_classes),
        figsize=(2.5 * len(model_classes), 3)
    )
    if not hasattr(ax, '__len__'):
        ax = [ax]
    for i, mclass in enumerate(model_classes):
        x = bootstrap(dfh_all, num_boot, mclass)
        (
            m_boot_means,
            m_boot_lower,
            m_boot_upper,
            h_boot_means,
            h_boot_lower,
            h_boot_upper,
        ) = x
        ax[i].errorbar(
            m_boot_means,
            h_boot_means,
            xerr=[m_boot_means - m_boot_lower, m_boot_upper - m_boot_means],
            yerr=[h_boot_means - h_boot_lower, h_boot_upper - h_boot_means],
        )
        ax[i].set_title(f"{mclass}")
    ax[0].set_xlabel('Model mean abs error (deg)')
    ax[0].set_ylabel('Human mean abs error (deg)')
    plt.tight_layout()
    plt.savefig(f'figures/summary/taylor_bays/error_quantiles_all_ss.pdf')


def plot_taylor_bays_biases_by_layer(
    model_class='vgg19',
    fixed_dprime=None,
    layers=[8, 9, 10, 11],
    setsize=1,
    binsize=12,
):
    """
    E.g., examine bias in early layers of VGG-19, which show human-like
    repulsion, even if they are not the overall best-fit across all set sizes.
    """
    df = collate_summaries_bays([model_class])
    df = df.sort_values(by=['setsize']).sort_index()
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    tcc = TCCBays(
        setsize,
        model_classes=[model_class],
        just_in_time_loading=True,
        inherit_from=f'bays2014_setsize1',
    )
    best_dps = []
    for layer in layers:
        dfm = df[df.model_name == model_class + f'_l{layer}']
        best_dp = dfm.groupby('dprime')['loglik'].mean().idxmax()
        best_dps.append(best_dp)
    plot_error_bias(
        tcc,
        [model_class + f'_l{layer}' for layer in layers],
        [fixed_dprime] * len(layers) if fixed_dprime else best_dps,
        title_full_name=True,
        tag='_layers_' + '_'.join([str(l) for l in layers])
    )


def plot_taylor_bays_biases(
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
    fit_across_setsize=False,
    abs_error=False,
):
    if fit_across_setsize:
        best_layers, best_dps, df = get_best_fit_models(
            'bays2014_all',
            model_classes=model_classes,
            return_df=True,
        )
    else:
        # Fit to setsize 1 and generalize to larger setsizes
        best_layers, best_dps, df = get_best_fit_models(
            f'bays2014_setsize1',
            model_classes=model_classes,
            return_df=True,
            verbose=True,
        )
        print(list(zip(model_classes, best_layers, best_dps)))
    setsizes = [1, 2, 4, 8]
    dfs_plot = []
    for ss in setsizes:
        tcc = TCCBays(
            ss,
            model_classes=model_classes,
            just_in_time_loading=True,
            inherit_from=f'bays2014_setsize1',
        )
        df_plot = plot_error_bias(
            tcc, best_layers,
            best_dps,
            abs_error=abs_error,
            fit_curve=True,
            tag='_fit_all_ss' if fit_across_setsize else '',
        )
        dfs_plot.append(df_plot)
    return dfs_plot


def plot_taylor_bays(
    model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19'],
    binsize=12,
    pca=1,
    num_boot=1000,
    analyses=[
        'biases',
        'spearman_full',
        'spearman_condensed',
        'quantiles',
        'setsize',
        'similarity_curves',
        'setsize_and_similarity_curves',
    ],
):

    # SETSIZE EFFECTS
    # (Mean abs error per set-size)
    if 'setsize' in analyses or 'setsize_and_similarity_curves' in analyses:
        df_plot_ss = plot_taylor_bays_setsize(
            model_classes=model_classes,
            binsize=binsize,
        )

    # Interrogate set-size effects further by comparing similarity curves for
    # different set sizes
    if 'similarity_curves' in analyses or 'setsize_and_similarity_curves' in analyses:
        df_plot_sc = plot_taylor_bays_sim_curves(
            model_classes=model_classes,
            binsize=binsize,
        )

    if 'setsize_and_similarity_curves' in analyses:
        # Combine data into one plot
        fig, ax = plt.subplots(
            ncols=len(model_classes) + 1,
            figsize=(3 * (len(model_classes) + 1) + 0.1, 3)
        )
        # Setsize in first plot
        dfph = df_plot_ss[df_plot_ss.label == 'human']
        ax[0].plot(dfph.x, dfph.y, label='human', linestyle='--')
        for mc in model_classes:
            dfp = df_plot_ss[df_plot_ss.label == mc]
            ax[0].plot(dfp.x, dfp.y, label=mc)
        ax[0].set_xlabel('Set size')
        ax[0].set_ylabel('Mean abs. error')
        ax[0].legend()
        # Sim curves in remaining plots
        for i, mc in enumerate(model_classes):
            for ss in df_plot_sc.ss.unique():
                dfp = df_plot_sc[
                    (df_plot_sc.model_class == mc) & (df_plot_sc.ss == ss)
                ]
                ax[i + 1].plot(dfp.x, dfp.y, label=f'ss={ss}')
            ax[i + 1].set_xlabel('Response angle')
            ax[i + 1].set_title(f'{mc}')
        ax[1].set_ylabel('Cosine similarity to target')
        ax[2].legend()
        plt.tight_layout()
        plt.savefig(f'figures/summary/taylor_bays/setsize_combined.pdf')

    # Plot biases
    if 'biases' in analyses:
        plot_taylor_bays_biases(
            model_classes=model_classes,
            binsize=binsize,
        )

    # Spearman rank analysis
    if 'spearman_full' in analyses:
        plot_taylor_bays_spearman_full(
            model_classes=model_classes,
            binsize=binsize,
        )

    # # Condensed version
    # if 'spearman_condensed' in analyses:
    #     # TODO: Update
    #     raise NotImplementedError()
    #     plot_taylor_bays_spearman_condensed(
    #         data_pth,
    #         model_classes=model_classes,
    #         binsize=binsize,
    #         pca=pca,
    #     )

    # # Re-plot using error quantiles
    # if 'quantiles' in analyses:
    #     # TODO: Update
    #     raise NotImplementedError()
    #     plot_taylor_bays_quantiles(
    #         data_pth,
    #         dataset,
    #         model_classes=model_classes,
    #         binsize=binsize,
    #         num_boot=num_boot,
    #         pca=pca,
    #     )
    return


def plot_error_bias(
    tcc,
    model_names,
    dprimes,
    title_full_name=False,
    tag='',
    abs_error=False,
    fit_curve=False,
):
    def _plot_curve_fit(ax, targets, errs_mean):
        from scipy.optimize import curve_fit
        targets = np.array(targets)
        lowest_err = np.inf
        def make_func(freq):
            return lambda x, a, b: a * np.sin(freq * x * np.pi / 180 + b)
        for freq in range(2, 7, 2):
            test_func = make_func(freq)
            fit_output = curve_fit(
                test_func, targets, errs_mean, p0=(5, 0), full_output=True
            )
            params = fit_output[0]
            infodict, mesg = fit_output[2:4]
            mean_abs_err = np.abs(infodict["fvec"]).mean()
            if mean_abs_err < lowest_err:
                lowest_err = mean_abs_err.tolist()
                best_freq = freq
                best_params = params.copy()
                best_test_func = make_func(freq)
        isort = np.argsort(targets)
        sine_fit = best_test_func(
            targets, best_params[0], best_params[1]
        )
        ax.plot(
            targets[isort],
            sine_fit[isort],
            color='red',
            linewidth=3,
        )
        pdict = {
            'freq': best_freq,
            'amplitude': best_params[0],
            'phase': best_params[1] * 180 / np.pi,
        }
        return pdict, sine_fit

    def _get_mean_errs(errs_dict, abs_error=False):
        errs_per_targ = {}  # Bin by angle
        for key, errs_stim_i in errs_dict.items():
            target = key[-1]
            if target not in errs_per_targ:
                errs_per_targ[target] = np.array(errs_stim_i).reshape(-1)
            else:
                errs_per_targ[target] = np.concatenate([
                    errs_per_targ[target],
                    np.array(errs_stim_i).reshape(-1)
                ])
        targets = [k[-1] for k in errs_dict.keys()]
        if abs_error:
            errs_mean = [np.mean(np.abs(errs_per_targ[t])) for t in targets]
        else:
            errs_mean = [np.mean(errs_per_targ[t]) for t in targets]
        return targets, errs_mean

    sine_fit = None
    herrs_dict = tcc.get_human_errors_by_key()
    fig, axs = plt.subplots(
        nrows=len(dprimes) + 1, figsize=(6, 2 * (len(dprimes) + 1))
    )
    if not hasattr(axs, '__len__'):
        axs = [axs]
    df_plot = pd.DataFrame({
        'dataset': [],
        'model_class': [],
        'x': [],
        'y': [],
        'colors': [],
        'sine_fit': []
    })
    pdicts = {}
    df_curve_fit = pd.DataFrame({
        'Model': [],
        'Amplitude': [],
        'Phase': [],
        'Frequency': [],
    })
    for ax, mname, dp in zip(axs, model_names, dprimes):
        mclass = f"{mname.split('_l')[0]}"
        idx = np.argmax(tcc.models.name.values == mname)
        errs_dict = tcc.sample_model_errors(dp, idxs=[idx])[mname]
        targets, errs_mean = _get_mean_errs(errs_dict, abs_error)
        if tcc.dataset == 'brady_alvarez':
            colors = [
                rgb_from_angle(
                    hue * np.pi / 180,
                    lab_center=tcc.lab_center,
                    radius=tcc.lab_radius
                )
                for hue in targets
            ]
        else:
            colors = None
        ax.scatter(targets, errs_mean, c=colors)

        if fit_curve:
            pdict, sine_fit = _plot_curve_fit(ax, targets, errs_mean)
            df_curve_fit = pd.concat([
                df_curve_fit,
                pd.DataFrame({
                    'Model': [mname],
                    'Amplitude': [pdict['amplitude']],
                    'Phase': [pdict['phase']],
                    'Frequency': [pdict['freq']],
                })
            ])

        if title_full_name:
            ax.set_title(f'{mname}')
        else:
            ax.set_title(mclass)
        df_plot = pd.concat([
            df_plot,
            pd.DataFrame({
                'dataset': [tcc.dataset] * len(targets),
                'model_class': [mclass] * len(targets),
                'x': targets,
                'y': errs_mean,
                'colors': colors or [None] * len(targets),
                'sine_fit': sine_fit if sine_fit is not None else np.zeros(len(targets)) * np.nan,
            })
        ])
    htargets, herrs_mean = _get_mean_errs(herrs_dict, abs_error)
    if fit_curve:
        pdict, sine_fit = _plot_curve_fit(axs[-1], htargets, herrs_mean)
        df_curve_fit = pd.concat([
            df_curve_fit,
            pd.DataFrame({
                'Model': ['human'],
                'Amplitude': [pdict['amplitude']],
                'Phase': [pdict['phase']],
                'Frequency': [pdict['freq']],
            })
        ])
    df_plot = pd.concat([
        df_plot,
        pd.DataFrame({
            'dataset': [tcc.dataset] * len(htargets),
            'model_class': ['human'] * len(htargets),
            'x': htargets,
            'y': herrs_mean,
            'colors': colors or [None] * len(targets),
            'sine_fit': sine_fit if sine_fit is not None else np.zeros(len(targets)) * np.nan,
        })
    ])
    axs[-1].scatter(htargets, herrs_mean, c=colors)
    axs[-1].set_title('human')
    axs[-1].set_xlabel('Stimulus angle (deg)')
    axs[-1].set_ylabel('Error (deg)')
    plt.tight_layout()
    if 'bays' in tcc.dataset:
        dataset = 'taylor_bays'
    else:
        dataset = 'brady_alvarez'
    save_pth = Path(
        f'figures/summary/{dataset}/{tcc.item_type}_biases_ss{tcc.setsize}{tag}.pdf'
    )
    save_pth.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_pth)

    if fit_curve:
        print(df_curve_fit.to_latex(escape=False, index=False))
    return df_plot


def get_spearman_condensed_ss_plot_data(
    experiment,
    setsizes,
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    fit_across_setsize=False,
    binsize=12,
):
    dfp = pd.DataFrame({
        'axis': [],
        'name': [],
        'x': [],
        'y': [],
        'pval': [],
    })

    if experiment == 'bays2014':
        if fit_across_setsize:
            # Fit to all at once
            best_layers, best_dps, df = get_best_fit_models(
                f'bays2014_all', model_classes=model_classes, return_df=True
            )
        else:
            # Fit to setsize 1 and generalize to larger setsizes
            best_layers, best_dps = get_best_fit_models(
                f'bays2014_setsize1', model_classes=model_classes
            )
            df = collate_summaries_bays(model_classes)
    else:
        best_layers, best_dps, df = get_best_fit_models(
            experiment, model_classes=model_classes, return_df=True
        )
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    for i, ss in enumerate(setsizes):
        for j, mclass in enumerate(model_classes):
            dfm = df[df.model_class == mclass]
            dfm_best = dfm[(dfm.dprime == best_dps[j]) & (dfm.model_name == best_layers[j])]
            x = [j * 0.5]
            y = dfm_best[dfm_best.setsize == ss].spearman_r
            pval = dfm_best[dfm_best.setsize == ss].spearman_pval
            dfp = pd.concat([
                dfp,
                pd.DataFrame({
                    'axis': [i],
                    'name': best_layers[j],
                    'x': x,
                    'y': y,
                    'pval': pval,
                })
            ])
    return dfp


def plot_brady_spearman_full(
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
):
    df = collate_summaries_brady(model_classes)
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    fig, ax = plt.subplots(
        ncols=len(model_classes),
        figsize=(4.5 * len(model_classes), 5)
    )
    if not hasattr(ax, '__len__'):
        ax = [ax]
    for i, mclass in enumerate(model_classes):
        dfm = df[df.model_class == mclass]
        colors = [
            'gray' if pval > 0.05 else 'b' for pval in dfm.spearman_pval
        ]
        # Get best-fit d' for each layer
        dfm = dfm.loc[dfm.groupby('model_name')['loglik'].idxmax()]
        layer_idxs = np.array([
            int(n.split('_')[-1][1:]) for n in dfm.model_name.values
        ])
        isort = np.argsort(layer_idxs)
        ax[i].bar(
            range(len(dfm)),
            dfm.spearman_r.values[isort],
            color=colors,
        )
        ax[i].set_xticks([])
        ax[i].set_xticklabels([])
        ax[i].set_xlabel(f'layers')
        ax[i].set_ylim(-0.45, 0.8)
        ax[i].set_title(f'{mclass}')
    ax[0].set_ylabel('Spearman rho')
    plt.tight_layout()
    plt.savefig(f'figures/summary/brady_alvarez/spearman_bars_colors.pdf')


def plot_bias_color_orientation(
    binsize=12,
    model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19'],
    plot_by_layer=False,
    abs_error=False,
):
    # Plot specific layers, rather than best-fit layers
    if plot_by_layer:
        plot_taylor_bays_biases_by_layer(
            model_class='vgg19',
            binsize=binsize,
        )

    # Produce separate plot for Supplementary for orientation bias when fitting
    # models to all set-sizes at once
    plot_taylor_bays_biases(
        model_classes=model_classes,
        binsize=binsize,
        abs_error=abs_error,
        fit_across_setsize=True,
    )[0]
    # Just fit to set-size 1 for combined plot
    df_plot_orient = plot_taylor_bays_biases(
        model_classes=model_classes,
        binsize=binsize,
        abs_error=abs_error,
    )[0]
    df_plot_color = plot_brady_biases(
        model_classes=model_classes,
        binsize=binsize,
        abs_error=abs_error,
    )
    fig, ax = plt.subplots(
        nrows=len(model_classes) + 1,
        ncols=2,
        figsize=(9, 2.2 * (len(model_classes) + 1))
    )
    for i, mc in enumerate(model_classes + ['human']):
        dfpo = df_plot_orient[df_plot_orient.model_class == mc]
        if dfpo.colors.values[0] is None:
            colors = None
        else:
            colors = dfpo.colors
        ax[i][0].scatter(dfpo.x, dfpo.y, c=colors)
        ax[i][0].set_title(mc)
        isort = np.argsort(dfpo.x)
        ax[i][0].plot(
            dfpo.x[isort],
            dfpo.sine_fit[isort],
            # linestyle='dotted',
            c='r',
            linewidth=3,
        )

    for i, mc in enumerate(model_classes + ['human']):
        dfpc = df_plot_color[df_plot_color.model_class == mc]
        ax[i][1].scatter(dfpc.x, dfpc.y, c=dfpc.colors)
        ax[i][1].set_title(mc)
    ax[-1][0].set_xlabel('Response angle (deg)')
    ax[-1][0].set_ylabel('Mean error (deg)')
    plt.tight_layout()
    abserr_str = '_abs_error' if abs_error else ''
    save_pth = Path(f'figures/summary/biases_combined{abserr_str}.pdf')
    save_pth.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_pth)
    return


def plot_ss_spearman_condensed(
    binsize=12,
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
):
    """
    Plot both color and orientation results in single figure
    """
    # Orientation
    dfh0 = collate_summaries_bays(['human'])
    cil0 = dfh0.conf_int95_lower.values.tolist()
    ciu0 = dfh0.conf_int95_upper.values.tolist()
    dfp0 = get_spearman_condensed_ss_plot_data(
        'bays2014', [1, 2, 4, 8], model_classes=model_classes,
    )
    # Color
    dfh1 = collate_summaries_brady(['human'])
    cil1 = dfh1.conf_int95_lower.values.tolist()
    ciu1 = dfh1.conf_int95_upper.values.tolist()
    dfp1 = get_spearman_condensed_ss_plot_data(
        'brady_alvarez', [3], model_classes=model_classes,
    )
    fig, ax = plt.subplots(ncols=5, figsize=(1.5 * 5, 4))
    dfp1['axis'] += 4
    dfp = pd.concat([dfp0, dfp1])
    cil = cil0 + cil1
    ciu = ciu0 + ciu1
    exp_names = ['Orient.'] * 4 + ['Color']
    for i, ss in enumerate([1, 2, 4, 8, 3]):
        ax[i].set_ylim(-1.0, 1.0)
        ax[i].set_title(f'{exp_names[i]} SS {ss}')
        ax[i].set_xticks(np.arange(len(model_classes)) * 0.5)
        ax[i].set_xticklabels(model_classes, ha='right', rotation=45)
        ax[i].axhline(y=0, color='black', linewidth=0.8)
        ax[i].spines[['right', 'top']].set_visible(False)
        if i != 0:
            ax[i].set_yticklabels([])
    for i in range(len(ax)):
        dfpi = dfp[dfp.axis == i]
        colors = [
            'gray' if pval > 0.05 else 'b'
            for pval in dfpi.pval
        ]
        ax[i].bar(dfpi.x, dfpi.y, width=0.25, color=colors)
        ax[i].hlines(
            cil[i], dfpi.x.min(), dfpi.x.max(), linestyle='dotted', colors='gray'
        )
        ax[i].hlines(
            ciu[i], dfpi.x.min(), dfpi.x.max(), linestyle='dotted', colors='gray'
        )
    plt.tight_layout()
    plt.savefig(f'figures/summary/spearman_bars_condensed_artificial.pdf')


# def plot_brady_spearman_condensed(
#     data_pth,
#     model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
#     binsize=12,
# ):
#     # Need to update to new version of get_spearman_condensed_ss_plot_data
#     raise NotImplementedError()
#     df = pd.read_csv(data_pth)
#     df = df.rename(
#         columns={
#             f'spearman_r_binsize{binsize}': 'spearman_r',
#             f'spearman_pval_binsize{binsize}': 'spearman_pval',
#         }
#     )

#     setsizes = [3]
#     dfp = get_spearman_condensed_ss_plot_data(
#         df, setsizes, model_classes=model_classes,
#     )
#     fig, ax = plt.subplots(
#         ncols=len(setsizes), figsize=(1.5 * len(setsizes), 4)
#     )
#     if not hasattr(ax, '__len__'):
#         ax = [ax]
#     for i, ss in enumerate(setsizes):
#         ax[i].set_ylim(-0.5, 0.5)
#         ax[i].set_title(f'Set-size: {ss}')
#         ax[i].set_xticks(np.arange(len(model_classes)) * 0.5)
#         ax[i].set_xticklabels(model_classes, ha='right', rotation=45)
#         ax[i].spines[['right', 'top']].set_visible(False)
#         if i != 0:
#             ax[i].set_yticklabels([])
#     for i in range(len(ax)):
#         dfpi = dfp[dfp.axis == i]
#         colors = [
#             'gray' if pval > 0.05 else 'b'
#             for pval in dfpi.pval
#         ]
#         ax[i].bar(dfpi.x, dfpi.y, width=0.25, label=dfpi.name, color=colors)
#     plt.tight_layout()
#     plt.savefig(f'figures/summary/brady_alvarez/spearman_bars_condensed.pdf')


def plot_brady_biases(
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
    abs_error=False,
):
    tcc = TCCBrady(
        3,
        model_classes=model_classes,
        just_in_time_loading=True,
        max_mem=370,
    )
    best_layers, best_dps = get_best_fit_models(
        'brady_alvarez',
        model_classes=model_classes,
        binsize=12,
        verbose=False,
    )
    df_plot = plot_error_bias(tcc, best_layers, best_dps, abs_error=abs_error)
    return df_plot


def plot_brady_sim_curves(
    data_pth,
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
):
    df = pd.read_csv(data_pth)
    df = df.sort_values(by=['setsize']).sort_index()
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    fig, ax = plt.subplots(
        ncols=len(model_classes),
        figsize=(2.5 * len(model_classes), 3)
    )
    df_plot = pd.DataFrame({
        'ss': [],
        'model_class': [],
        'x': [],
        'y': [],
    })
    setsizes = [3]
    sims = {}
    for i, mclass in enumerate(model_classes):
        # dfm = df[df.model_name.str.contains(mclass + '_l')]
        dfm = df[df.model_class == mclass]
        best_layer, best_dp = dfm.groupby(['model_name', 'dprime'])['loglik'].mean().idxmax()
        dfm_best = dfm[(dfm.dprime == best_dp) & (dfm.model_name == best_layer)]
        sims[mclass] = {}
        for ss in setsizes:
            item_locs = None  # Even spacing
            tcc = TCCBrady(
                ss,
                model_classes=model_classes,
                just_in_time_loading=True,
            )
            layer_idx = np.argmax(tcc.models.name == best_layer)
            targets = [0] * ss
            responses = [[0] * (ss - 1) + [i] for i in range(360)]
            target_id = (tuple(targets), item_locs)
            resp_ids = [(tuple(r), item_locs) for r in responses]
            # Get model info
            if mclass.startswith('clip_RN50'):
                embed_func = clip_embed
                img_size = 224
                embed_args = (tcc.models.specs.values[layer_idx], 'RN50')
            elif mclass.startswith('clip_ViT-B16'):
                embed_func = clip_embed
                img_size = 224
                embed_args = (tcc.models.specs.values[layer_idx], 'ViT-B16')
            elif mclass in ['vgg19', 'resnet50']:
                embed_func = torchvision_embed
                img_size = 224
                embed_args = (tcc.models.specs.values[layer_idx], mclass)
            else:
                raise NotImplementedError()
            # Compute model embeddings
            Z = tcc._embed(
                best_layer,
                target_id,
                resp_ids,
                embed_func,
                img_size,
                *embed_args
            )
            # Compute similarities
            sims[mclass][ss] = cossim_torch(Z[0], Z[1:], dev=device).detach().cpu().numpy()
            ax[i].plot(sims[mclass][ss], label=f'ss={ss}')
            ax[i].set_title(f'{mclass}')
            ax[i].set_xlabel(f'Response angle')
            df_plot = pd.concat([
                df_plot,
                pd.DataFrame({
                    'ss': [ss] * 360,
                    'model_class': [mclass] * 360,
                    'x': np.arange(0, 360),
                    'y': sims[mclass][ss],
                })
            ])
    ax[0].set_ylabel(f'Cosine similarity to target')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(f'figures/summary/brady_alvarez/similarity_curves_by_setsize.pdf')
    return df_plot


def plot_brady_quantiles(
    data_pth,
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    num_boot=1000,
    binsize=12,
):
    df = pd.read_csv(data_pth)
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    tcc = TCCBrady(
        3,
        model_classes=model_classes,
        just_in_time_loading=True,
        max_mem=370,
    )

    fig, ax = plt.subplots(
        ncols=len(model_classes),
        figsize=(2.5 * len(model_classes), 3)
    )
    if not hasattr(ax, '__len__'):
        ax = [ax]
    num_quantiles = 4
    dfh = tcc.human_data
    dfh['abs_error'] = dfh.error.abs()
    dfh['bins'] = pd.qcut(dfh.abs_error, num_quantiles)
    # human_qmeans = dfh.groupby('bins')['abs_error'].mean()
    # Get model errors for each trial and compute corresponding means
    for i, mclass in enumerate(model_classes):
        # dfm = df[df.model_name.str.contains(mclass + '_l')]
        dfm = df[df.model_class == mclass]
        best_layer, best_dp = dfm.groupby(['model_name', 'dprime'])['loglik'].mean().idxmax()
        layer_idx = np.argmax(tcc.models.name == best_layer)
        errs = tcc.sample_model_errors(best_dp, idxs=[layer_idx])[best_layer]

        mean_abs_model_error = []
        for j in range(len(dfh)):
            key = tcc._get_key_from_idx(j)
            mean_abs_model_error.append(np.abs(errs[key]).mean())
        dfh[f'{mclass}_err'] = mean_abs_model_error

        model_bin_means = [
            dfh.groupby('bins').sample(frac=1, replace=True).groupby('bins')[f'{mclass}_err'].mean()
            for iboot in range(num_boot)
        ]
        human_bin_means = [
            dfh.groupby('bins').sample(frac=1, replace=True).groupby('bins')['abs_error'].mean()
            for iboot in range(num_boot)
        ]
        # Confidence intervals
        m_boot_means = np.mean(model_bin_means, axis=0)
        m_boot_lower = np.quantile(model_bin_means, q=0.05, axis=0)
        m_boot_upper = np.quantile(model_bin_means, q=0.95, axis=0)
        h_boot_means = np.mean(human_bin_means, axis=0)
        h_boot_lower = np.quantile(human_bin_means, q=0.05, axis=0)
        h_boot_upper = np.quantile(human_bin_means, q=0.95, axis=0)
        print(f'Model: mean={m_boot_means}, lower={m_boot_lower}, upper={m_boot_upper}')
        print(f'Humans: mean={h_boot_means}, lower={h_boot_lower}, upper={h_boot_upper}')

        # No bootstrap
        # ax[i].plot(dfh.groupby('bins')[f'{mclass}_err'].mean(), human_qmeans)
        # With bootstrap
        ax[i].errorbar(
            m_boot_means,
            h_boot_means,
            xerr=[m_boot_means - m_boot_lower, m_boot_upper - m_boot_means],
            yerr=[h_boot_means - h_boot_lower, h_boot_upper - h_boot_means],
        )
        # ax[i].set_title(f"{mclass}, d'={best_dp}")
        ax[i].set_title(f"{mclass}")
    ax[0].set_xlabel('Model mean abs error (deg)')
    ax[0].set_ylabel('Human mean abs error (deg)')
    plt.tight_layout()
    plt.savefig('figures/summary/brady_alvarez/error_quantiles.pdf')


def do_brady_hierarchical(
    data_pth,
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
):
    df = pd.read_csv(data_pth)
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    tcc = TCCBrady(
        3,
        model_classes=model_classes,
        just_in_time_loading=True,
        max_mem=370,
    )
    # fig, ax = plt.subplots(
    #     ncols=len(model_classes),
    #     figsize=(2.5 * len(model_classes), 3)
    # )
    # if not hasattr(ax, '__len__'):
    #     ax = [ax]
    rhos = {m: [] for m in model_classes}
    num_stims = 50  # Num random stimuli to generate
    num_samp = 100  # Num responses sampled per stimulus
    for i, mclass in enumerate(model_classes):
        # dfm = df[df.model_name.str.contains(mclass + '_l')]
        dfm = df[df.model_class == mclass]
        best_layer, best_dp = dfm.groupby(['model_name', 'dprime'])['loglik'].mean().idxmax()
        layer_idx = np.argmax(tcc.models.name == best_layer)
        item_locs = (0. + 90., 120. + 90., 240. + 90.)  # Must be type float
        target_hues_list = []
        resp_hues_list = []
        for _ in tqdm(range(num_stims)):
            # Sample random stimulus values
            target_hues = np.random.choice(360, size=3)
            # Get response options that fit criteria (i.e. large errors on all
            # three items)
            responses0 = product(np.arange(360)[::16], repeat=3)
            responses = []
            for resp in responses0:
                diff = np.abs(standardize_angles(
                    np.array(resp) - target_hues, deg=True
                ))
                if np.all(diff > 45):  # Error threshold of 45 deg
                    responses.append(resp)

            # Get model info
            if mclass.startswith('clip_RN50'):
                embed_func = clip_embed
                img_size = 224
                embed_args = (tcc.models.specs.values[layer_idx], 'RN50')
            elif mclass in ['vgg19', 'resnet50']:
                embed_func = torchvision_embed
                img_size = 224
                embed_args = (tcc.models.specs.values[layer_idx], mclass)
            elif mclass.startswith('clip_ViT-B16'):
                embed_func = clip_embed
                img_size = 224
                embed_args = (tcc.models.specs.values[layer_idx], 'ViT-B16')
            else:
                raise NotImplementedError()
            # Compute keys (unique identifiers) for each stimulus to pass to
            # functions for computing similarities
            target_id = (tuple(target_hues), item_locs)
            resp_ids = [(tuple(r), item_locs) for r in responses]
            # Compute model embeddings
            Z = tcc._embed(
                best_layer,
                target_id,
                resp_ids,
                embed_func,
                img_size,
                *embed_args
            )
            # Compute similarities
            sims = cossim_torch(Z[0], Z[1:], dev='cpu').detach().cpu().numpy()
            # Sample responses
            resp_idx = sample_TCC(sims, best_dp, N=num_samp)
            resp_hues_list.extend(np.array([responses[r] for r in resp_idx]))
            target_hues_list.extend(
                [target_hues.copy() for n in range(num_samp)]
            )
        # Get correlation between variances
        resp_hues_vars = [circvar(r) for r in resp_hues_list]
        target_hues_vars = [circvar(t) for t in target_hues_list]
        np.save(
            f'{DATA_STORAGE}/data_tcc/setsize_analysis/brady_alvarez/{mclass}_resp_hues_vars.npy',
            np.array(resp_hues_vars)
        )
        np.save(
            f'{DATA_STORAGE}/data_tcc/setsize_analysis/brady_alvarez/{mclass}_target_hues_vars.npy',
            np.array(target_hues_vars)
        )
        print(
            f'Response hues vs. stimulus hues variances ({mclass}):'
            f' {spearmanr(resp_hues_vars, target_hues_vars)}'
        )
        tcc.embeddings_cache = {}

        # ax[i].scatter(target_hues_vars, resp_hues_vars)
        # ax[i].set_title(f'{mclass}')
        # ax[i].set_xlabel('Circ. var. of stimulus hues')
        # ax[i].set_ylabel('Circ. var. of response hues')
    # plt.tight_layout()
    # out_pth = Path(
    #     'figures/summary/brady_alvarez/resp_var_vs_stim_var_scatter.pdf'
    # )
    # out_pth.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(out_pth)


def get_best_fit_models(
    experiment,
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    binsize=12,
    verbose=False,
    return_df=False,
):
    if experiment == 'scene_wheels':
        df = collate_summaries_scene_wheels(model_classes + ['human'])
    elif experiment == 'brady_alvarez':
        df = collate_summaries_brady(model_classes + ['human'])
    elif experiment == 'bays2014_setsize1':
        df = collate_summaries_bays(model_classes + ['human'], setsizes=[1])
    elif experiment == 'bays2014_setsize2':
        df = collate_summaries_bays(model_classes + ['human'], setsizes=[2])
    elif experiment == 'bays2014_setsize4':
        df = collate_summaries_bays(model_classes + ['human'], setsizes=[4])
    elif experiment == 'bays2014_setsize8':
        df = collate_summaries_bays(model_classes + ['human'], setsizes=[8])
    elif experiment == 'bays2014_all':
        df = collate_summaries_bays(model_classes + ['human'], setsizes=[1, 2, 4, 8])

    if 'radius' in df.columns:
        df = df.loc[df.radius == 'all']
    df = df.rename(
        columns={
            f'spearman_r_binsize{binsize}': 'spearman_r',
            f'spearman_pval_binsize{binsize}': 'spearman_pval',
        }
    )
    dfm = df[df.model_name != 'human']

    # Check that grid search over d' was big enough
    for mname in dfm.model_name.unique():
        dfi = df[df.model_name == mname]
        if experiment == 'scene_wheels':
            max_dp = 20
        elif experiment == 'brady_alvarez':
            max_dp = 20
        elif experiment.startswith('bays'):
            if mname.startswith('clip_RN50'):
                max_dp = 1000
            elif mname.startswith('clip_ViT-B16'):
                max_dp = 1000
            elif mname.startswith('vgg19'):
                max_dp = 100
        else:
            raise NotImplementedError()
        best_dp = dfi.iloc[dfi.loglik.argmax()].dprime
        if best_dp >= max_dp:
            print(f'Best d-prime not found for {mname} in {experiment}')

    best_layers = []
    best_dps = []
    for i, mclass in enumerate(model_classes):
        dfm = df[df.model_class == mclass]
        best_layer, best_dp = dfm.groupby(
            ['model_name', 'dprime']
        )['loglik'].mean().idxmax()
        best_layers.append(best_layer)
        best_dps.append(best_dp)
        if verbose:
            print(f'Best model for {mclass}: layer {best_layer}, dprime={best_dp}')
    if return_df:
        return best_layers, best_dps, df
    else:
        return best_layers, best_dps


def plot_brady(
    model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    num_boot=1000,
    binsize=12,
    analyses=[
        'biases',
        'spearman_full',
        'spearman_condensed',
        'quantiles',
        'hierarchical',
        'similarity_curves',
    ],
):
    if 'spearman_full' in analyses:
        # Spearman rank analysis
        plot_brady_spearman_full(
            model_classes=model_classes,
            binsize=12,
        )

    # Condensed version
    # if 'spearman_condensed' in analyses:
    #     plot_brady_spearman_condensed(
    #         data_pth,
    #         model_classes=model_classes,
    #         binsize=binsize,
    #     )
    
    # if 'quantiles' in analyses:
    #     # Re-plot using error quantiles
    #     plot_brady_quantiles(
    #         data_pth,
    #         model_classes=model_classes,
    #         num_boot=num_boot,
    #         binsize=binsize,
    #     )

    # if 'biases' in analyses:
    #     plot_brady_biases(
    #         data_pth,
    #         model_classes=model_classes,
    #         binsize=binsize,
    #     )

    # if 'similarity_curves' in analyses:
    #     plot_brady_sim_curves(
    #         data_pth,
    #         model_classes=model_classes,
    #         binsize=binsize,
    #     )

    # Humans respond with higher-variance choices when display is higher
    # variance, even when they are completely wrong. Do models do the same?
    # Approach 1: Replicate analysis in paper. For each unique display,
    # enumerate all response options for the three items, and consider the
    # subset for which error on all three items is at least 45 degrees. Then
    # take samples and measure the correlation between the variance across the
    # three hues in responses vs target display.
    if 'hierarchical' in analyses:
        do_brady_hierarchical(
            data_pth,
            model_classes=model_classes,
            binsize=binsize,
        )

    # # Humans respond with higher-variance choices when display is higher
    # # variance, even when they are completely wrong. Do models do the same?
    # # Approach 2: Ask first-order question: Do model choices for a particular
    # # probed item vary as a function of choices for the other items? Measure
    # # this by computing spearman rank correlations between similarity scores of
    # # the response options, while varying the values of the non-probed items.
    # # A high correlation means that the model will tend to still prefer the
    # # same responses, independent of its choices for the other items.
    # fig, ax = plt.subplots(
    #     ncols=len(model_classes),
    #     figsize=(2.5 * len(model_classes), 3)
    # )
    # if not hasattr(ax, '__len__'):
    #     ax = [ax]
    # rhos = {m: [] for m in model_classes}
    # for i, mclass in enumerate(model_classes):
    #     dfm = df[df.model_name.str.contains(mclass + '_l')]
    #     best_layer, best_dp = dfm.groupby(['model_name', 'dprime'])['loglik'].mean().idxmax()
    #     layer_idx = np.argmax(tcc.models.name == best_layer)
    #     item_locs = (0. + 90., 120. + 90., 240. + 90.)  # Must be type float
    #     num_stims = 100
    #     for _ in tqdm(range(num_stims)):
    #         # Sample random stimulus values
    #         target_hues = np.random.choice(360, size=3)
    #         idx_probe = np.random.choice(3)  # Probed location
    #         idx_nonprobe = [j for j in range(3) if j != idx_probe]  # Other locations
    #         # Get response choice scores assuming non-probed items are set to
    #         # their correct values
    #         choices_matched = []
    #         for a in range(360):
    #             c = target_hues.copy()
    #             c[idx_probe] = a
    #             choices_matched.append(c)
    #         # Get response choice scores assuming non-probed items are set to
    #         # randomly chosen values
    #         choices_rand = []
    #         c = np.random.choice(360, size=3)
    #         for a in range(360):
    #             c[idx_probe] = a
    #             choices_rand.append(c.copy())
    #         # Get model info
    #         if mclass.startswith('clip_RN50'):
    #             embed_func = clip_embed
    #             img_size = 224
    #             embed_args = (tcc.models.specs.values[layer_idx], 'RN50')
    #         elif mclass in ['vgg19', 'resnet50']:
    #             embed_func = torchvision_embed
    #             img_size = 224
    #             embed_args = (tcc.models.specs.values[layer_idx], mclass)
    #         else:
    #             raise NotImplementedError()
    #         # Compute keys (unique identifiers) for each stimulus to pass to
    #         # functions for computing similarities
    #         target_id = (tuple(target_hues), item_locs)
    #         choice_ids_matched = [(tuple(c), item_locs) for c in choices_matched]
    #         choice_ids_rand = [(tuple(c), item_locs) for c in choices_rand]
    #         # Compute model embeddings
    #         Z_matched = tcc._embed(
    #             mclass + f'_l{best_layer}',
    #             target_id,
    #             choice_ids_matched,
    #             embed_func,
    #             img_size,
    #             *embed_args
    #         )
    #         Z_rand = tcc._embed(
    #             mclass + f'_l{best_layer}',
    #             target_id,
    #             choice_ids_rand,
    #             embed_func,
    #             img_size,
    #             *embed_args
    #         )
    #         # Compute similarities
    #         sims_matched = cossim_torch(
    #             Z_matched[0], Z_matched[1:]
    #         ).detach().cpu().numpy()
    #         sims_rand = cossim_torch(
    #             Z_rand[0], Z_rand[1:]
    #         ).detach().cpu().numpy()
    #         rhos[mclass].append(spearmanr(sims_matched, sims_rand)[0])
    #     print(f'Mean rho (model {mclass}): {np.mean(rhos[mclass])}')
    #     print(f'Min rho (model {mclass}): {np.min(rhos[mclass])}')
    #     print(f'STD rho (model {mclass}): {np.std(rhos[mclass])}')
    #     tcc.embeddings_cache = {}
    # from ipdb import set_trace; set_trace()

    # Approach 3: generate new stimuli which are tailored to assessing this
    # question directly. Specifically, generate set-size 3 stimuli for which
    # the hues are evenly spaced. Then, generate possible response
    # alternatives where two of the hues remain fixed and one is varied. The
    # fixed hues are located exactly equidistant to their two closest target
    # values, while the third one varies all the way around the circle. We can
    # then examine how target-probe similarity varies as the third item moves
    # around. In particular, see whether similarity increases as item three
    # approaches its correct hue, or decreases as a result of getting too close
    # to other items, and thus violating the higher level structure (i.e. that
    # the three hues were all dissimilar). Next, test the case in which the
    # target hues are all close. As before, let just one item vary in the
    # response options. Does the similarity score go up as one item gets closer
    # to its correct value, or is it penalized for violating the higher level
    # structure (all items being similar).
    # fig, ax = plt.subplots(
    #     ncols=len(model_classes),
    #     figsize=(2.5 * len(model_classes), 3)
    # )
    # if not hasattr(ax, '__len__'):
    #     ax = [ax]
    # for i, mclass in enumerate(model_classes):
    #     dfm = df[df.model_name.str.contains(mclass + '_l')]
    #     best_layer, best_dp = dfm.groupby(['model_name', 'dprime'])['loglik'].mean().idxmax()
    #     layer_idx = np.argmax(tcc.models.name == best_layer)
    #     # Dissimilar hues
    #     item_locs = (0. + 90., 120. + 90., 240. + 90.)
    #     sims = []
    #     for rot in np.arange(0, 360, 16):
    #         # Repeat experiment for different rotations of the target hues,
    #         # keeping the original spacing in hue angle space.
    #         target_hues = np.array([0., 120., 240.]) + rot
    #         choice_hues = [
    #             np.array([60., 180., j]) + rot for j in range(1, 360)
    #         ]
    #         if mclass.startswith('clip_RN50'):
    #             embed_func = clip_embed
    #             img_size = 224
    #             embed_args = (tcc.models.specs.values[layer_idx], 'RN50')
    #         elif mclass in ['vgg19', 'resnet50']:
    #             embed_func = torchvision_embed
    #             img_size = 224
    #             embed_args = (tcc.models.specs.values[layer_idx], mclass)
    #         else:
    #             raise NotImplementedError()
    #         target_id = (tuple(target_hues), item_locs)
    #         choice_ids = [(tuple(ph), item_locs) for ph in choice_hues]
    #         Z = tcc._embed(
    #             mclass + f'_l{best_layer}',
    #             target_id,
    #             choice_ids,
    #             embed_func,
    #             img_size,
    #             *embed_args
    #         )
    #         sims.append(cossim_torch(Z[0], Z[1:]).detach().cpu().numpy())
    #     sims = np.mean(sims, axis=0)
    #     # Summarize repulsion by taking differences between mirrored points on
    #     # either side of the true hue value (at index 239). Differences of zero
    #     # indicate no repulsion bias. (A repulsion bias would occur if there is
    #     # a preference for maintaining the higher-level structure of spread
    #     # out hues.) Positive values indicate repulsion.
    #     repulsion_bias = sims[240:290] - sims[189:239]
    #     # ax[i].plot(range(1, 360), sims)
    #     ax[i].plot(list(range(-50, 0) + list(range(1, 51))), repulsion_bias)
    #     tcc.embeddings_cache = {}  # Need to manually reset
    #     # Similar hues
    #     # TODO
    # plt.show()
    return

def plot_scene_wheels_rank_transform(
    model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19']
):
    from sklearn.linear_model import LinearRegression
    best_layers, best_dps = get_best_fit_models(
        'scene_wheels',
        model_classes=model_classes,
    )
    tcc = TCCSceneWheel()
    human_errs_by_stim = tcc.get_human_errors_by_key()
    radii = tcc.radius_vals
    bins = np.arange(0, 360 + 1, 30)
    fig, ax = plt.subplots(
        nrows=len(model_classes),
        ncols=len(radii),
        figsize=(12, 8),
    )
    iterator = list(zip(model_classes, best_layers, best_dps))
    ax[-1, 0].set_xlabel('Human rank transform')
    ax[-1, 0].set_ylabel('Model rank transform')
    # ax[-1, 0].set_xticklabels([0, 0.5, 1])
    # ax[-1, 0].set_yticklabels([0, 0.5, 1])
    for i, rad in enumerate(radii):
        for j, (mclass, best_layer, best_dp) in enumerate(iterator):
            layer_idx = np.argmax(tcc.models.name == best_layer)
            model_errs_by_stim = tcc.get_or_load_TCC_samples(
                layer_idx, best_dp
            )[1]
            stimulus_keys = model_errs_by_stim.keys()
            skeys = [key for key in stimulus_keys if key[1] == rad]
            merrs_bin_ave_flat = tcc._get_values_list_flat(
                skeys, model_errs_by_stim, [rad], bins
            )
            herrs_bin_ave_flat = tcc._get_values_list_flat(
                skeys, human_errs_by_stim, [rad], bins
            )
            n = len(merrs_bin_ave_flat)
            m_rank = np.argsort(np.argsort(merrs_bin_ave_flat)) / n
            h_rank = np.argsort(np.argsort(herrs_bin_ave_flat)) / n
            reg = LinearRegression(fit_intercept=False)
            reg.fit(m_rank[:, None], h_rank)
            ax[j, i].scatter(h_rank, m_rank, label=mclass)
            ax[j, i].plot(
                np.linspace(0, 1, 10),
                reg.predict(np.linspace(0, 1, 10)[:, None]),
                c='red'
            )
            ax[j, i].set_title(f'{mclass}, radius={rad}')
            ax[j, i].set_aspect('equal')
            ax[j, i].set_xticklabels([])
            ax[j, i].set_yticklabels([])
            ax[j, i].set_xticks([0, 0.5, 1])
            ax[j, i].set_yticks([0, 0.5, 1])
    ax[-1, 0].set_xticklabels([0, 0.5, 1])
    ax[-1, 0].set_yticklabels([0, 0.5, 1])
    plt.tight_layout()
    plt.savefig('figures/summary/scene_wheels/rank_scatters.pdf')
    return


def plot_scene_wheels_spearman_full(
    model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19']
):
    df = collate_summaries_scene_wheels(model_classes + ['human'])
    dfh = df[df.model_name == 'human']

    # Spearman rank analysis
    fig, ax = plt.subplots(
        nrows=len(model_classes),
        ncols=6,
        figsize=(6 * 2.5, 4.5 * len(model_classes))
    )
    rad2idx = {'2': 0, '4': 1, '8': 2, '16': 3, '32': 4, 'all': 5}
    for i, mclass in enumerate(model_classes):
        # dfm = df[df.model_name.str.contains(mclass + '_l')]
        dfm = df[df.model_class == mclass]
        for rad, group in dfm.groupby('radius'):
            colors = [
                'gray' if pval > 0.05 else 'b' for pval in group.spearman_pval
            ]
            layer_idxs = np.array([
                int(n.split('_')[-1][1:]) for n in group.model_name.values
            ])
            isort = np.argsort(layer_idxs)
            ax[i][rad2idx[str(rad)]].bar(
                range(len(group)),
                group.spearman_r.values[isort],
                color=colors,
            )
            ax[i][rad2idx[str(rad)]].set_xticks([])
            ax[i][rad2idx[str(rad)]].set_xticklabels([])
            ax[i][rad2idx[str(rad)]].set_xlabel(f'{mclass} layers')
            ax[i][rad2idx[str(rad)]].set_ylim(-0.3, 1)
            if i == 0:
                ax[i][rad2idx[str(rad)]].set_title(f'Radius={rad}')
    ax[0, 0].set_ylabel('Spearman rho')
    plt.tight_layout()
    plt.savefig(f'figures/summary/scene_wheels/spearman_bars_scene_wheels.pdf')


def plot_scene_wheels_spearman_condensed(
    model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19', 'rgb', 'pixels', 'vae']
):
    radii = ['all', 2, 4, 8, 16, 32]
    dfp = pd.DataFrame({
        'axis': [],
        'name': [],
        'x': [],
        'y': [],
    })
    min_y = np.inf

    best_layers, best_dps = get_best_fit_models(
        'scene_wheels',
        model_classes=model_classes,
    )
    df = collate_summaries_scene_wheels(model_classes + ['human'])
    dfh = df[df.model_name == 'human']

    for i, mclass in enumerate(model_classes):
        for j, r in enumerate(radii):
            dfr = df[df.radius == r]
            dfm_best = dfr[dfr.model_name == best_layers[i]]
            x = [i]
            y = dfm_best.spearman_r
            dfp = pd.concat([
                dfp,
                pd.DataFrame({
                    'axis': [j],
                    'name': [mclass],
                    'x': x,
                    'y': y,
                })
            ])
            if np.min(y) < min_y:
                min_y = np.min(y)

    cil = [dfh[dfh.radius == r].conf_int95_lower.values[0] for r in radii]
    ciu = [dfh[dfh.radius == r].conf_int95_upper.values[0] for r in radii]

    fig, ax = plt.subplots(
        ncols=len(radii), figsize=(2 * len(radii), 5)
    )
    for i, r in enumerate(radii):
        ax[i].set_ylim(min_y * 1.05, 0.93)
        ax[i].set_title(f'Radius: {r}')
        ax[i].set_xticks(range(len(model_classes)))
        ax[i].set_xticklabels(model_classes, ha='right', rotation=45)
        ax[i].spines[['right', 'top']].set_visible(False)
        ax[i].axhline(y=0, color='black', linewidth=0.8)
        if i != 0:
            ax[i].set_yticklabels([])
    for i in range(len(ax)):
        dfpi = dfp[dfp.axis == i]
        ax[i].bar(dfpi.x, dfpi.y, width=0.4, label=dfpi.name)
        ax[i].hlines(
            cil[i], dfpi.x.min(), dfpi.x.max(), linestyle='dotted', colors='gray'
        )
        ax[i].hlines(
            ciu[i], dfpi.x.min(), dfpi.x.max(), linestyle='dotted', colors='gray'
        )
    ax[0].set_ylabel('Spearman rho')
    plt.tight_layout()
    plt.savefig(f'figures/summary/scene_wheels/spearman_bars_condensed.pdf')
    plt.savefig(f'figures/summary/scene_wheels/spearman_bars_condensed.png')


def plot_scene_wheels_error_vs_radius(
    model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19', 'rgb', 'pixels', 'vae']
):
    from tqdm import tqdm

    def _boot(num_boot=1000):
        tcc = TCCSceneWheel()
        mean_errors = []
        for k in tqdm(range(num_boot), desc='Bootstrapping...', dynamic_ncols=True):
            # Bootstrap resample within radius
            df_boot = tcc.human_data.groupby([
                'wheel_num', 'radius', pd.cut(tcc.human_data.answer, np.arange(-1, 360, 30))
            ]).sample(frac=1, replace=True)
            df_boot['abs_error'] = np.abs(df_boot.error)
            mean_errors.append(
                df_boot.groupby('radius')['abs_error'].mean().values
            )
        mean_errors = np.array(mean_errors)
        lower, mean, upper = [], [], []
        for i in range(5):
            x = np.sort(mean_errors[:, i])
            lower.append(x[num_boot // 20])
            upper.append(x[-num_boot // 20])
            mean.append(x.mean())
        return np.array(lower), np.array(mean), np.array(upper)

    best_layers, best_dps = get_best_fit_models(
        'scene_wheels',
        model_classes=model_classes,
    )
    df = collate_summaries_scene_wheels(model_classes + ['human'])
    dfh = df[df.model_name == 'human']
    lower, human_mean, upper = _boot()  # Confidence intervals

    fig, ax = plt.subplots(
        ncols=len(model_classes), figsize=(2 * len(model_classes), 2.2)
    )
    radii = [2, 4, 8, 16, 32]

    for i, mclass in enumerate(model_classes):
        y = []
        for j, r in enumerate(radii):
            dfr = df[df.radius == r]
            dfm_best = dfr[dfr.model_name == best_layers[i]]
            y.append(dfm_best.mean_abs_err.values[0])
        ax[i].errorbar(
            radii,
            human_mean,
            yerr=[human_mean - lower, upper - human_mean],
            label='human'
        )
        ax[i].plot(radii, y, label=f"{mclass}")
        ax[i].set_xticks(radii)
        ax[i].set_xticklabels(radii)
        ax[i].legend()
    ax[0].set_xlabel('Radius')
    ax[0].set_ylabel('Mean abs error')
    plt.tight_layout()
    plt.savefig(f'figures/summary/scene_wheels/error_per_radius.pdf')
    plt.savefig(f'figures/summary/scene_wheels/error_per_radius.png')


def plot_scene_wheels_scatters(
    data_pth,
    model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19'],
    radius='all',
):
    # Need to update to use new collate_summaries_scene_wheels function
    raise NotImplementedError()

    fig, ax = plt.subplots(
        ncols=len(model_classes), figsize=(5 * len(model_classes), 5)
    )
    tcc = TCCSceneWheel()
    herrs = tcc.get_human_errors_by_key()
    for i, mclass in enumerate(model_classes):
        if mclass in ['rgb', 'pixels', 'vae_beta0.01']:
            pths = data_pth.glob(f'summary_data_{mclass}_rad_all.csv')
        else:
            pths = data_pth.glob(f'summary_data_{mclass}_l*_rad_all.csv')
        dfm = pd.concat([pd.read_csv(pth) for pth in pths])
        # Get best-fit model according to summary data file
        best_layer, best_dp = dfm.groupby(
            ['model_name', 'dprime']
        )['loglik'].mean().idxmax()
        layer_idx = np.argmax(tcc.models.name == best_layer)
        errs = tcc.get_or_load_TCC_samples(layer_idx, best_dp)[1]
        x = []
        y = []
        for key in herrs.keys():
            if radius == 'all' or key[1] == radius:
                x.append(np.abs(herrs[key]).mean())
                y.append(np.abs(errs[key]).mean())
        ax[i].scatter(x, y, label=mclass, s=1)
        ax[i].legend()
    ax[0].set_xlabel('Human mean abs error')
    ax[0].set_ylabel('Model mean abs error')
    plt.tight_layout()
    plt.savefig(f'figures/summary/scene_wheels/error_scatters_rad_{radius}.pdf')


def plot_scene_wheels_error_hist(
    model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19']
):
    best_layers, best_dprimes, df = get_best_fit_models(
        'scene_wheels',
        model_classes=model_classes,
        return_df=True,
    )
    fig, ax = plt.subplots(ncols=len(model_classes) + 1, figsize=(12, 3))
    tcc = TCCSceneWheel()
    herrs_dict = tcc.get_human_errors_by_key()
    num_bins = 25
    for i, mclass in enumerate(['human'] + model_classes):
        if i == 0:
            errs_dict = herrs_dict
        else:
            layer_idx = np.argmax(tcc.models.name == best_layers[i - 1])
            errs_dict = tcc.get_or_load_TCC_samples(layer_idx, best_dprimes[i - 1])[1]
        ax[i].set_title(f'{mclass}')
        for j, rad in enumerate([2, 4, 8, 16, 32]):
            errs = np.concatenate([
                val for key, val in errs_dict.items() if key[1] == rad
            ])
            hist, bin_edges = np.histogram(
                errs, bins=num_bins, range=(-180, 180), density=True
            )
            bin_means = [
                (bin_edges[k] + bin_edges[k + 1]) / 2
                for k in range(len(bin_edges) - 1)
            ]
            # ax[i].plot(bin_means, hist, 'o', label=f'radius={rad}', fillstyle='none')
            ax[i].plot(bin_means, hist, label=f'radius={rad}', linewidth=1)
            ax[i].set_ylim(0, 0.055)
            if i != 0:
                ax[i].set_yticklabels([])
    plt.legend()
    ax[0].set_ylabel('Probability density')
    ax[0].set_xlabel('Error (deg)')
    plt.tight_layout()
    plt.savefig('figures/summary/scene_wheels/error_hists.pdf')
    return

def plot_scene_wheels(
    analyses=[
        'spearman_full', 'spearman_condensed', 'error_vs_radius', 'scatters',
        'error_spearman_combined', 'error_hist',
    ]
):
    # TODO: Update this function for new loading scheme
    # if 'scatters' in analyses:
    #     for rad in [2, 4, 8, 16, 32, 'all']:
    #         plot_scene_wheels_scatters(
    #             data_pth.parent, radius=rad,
    #         )

    if 'spearman_full' in analyses:
        plot_scene_wheels_spearman_full()

    if 'spearman_condensed' in analyses:
        # Condensed version of previous plot showing only best-fit layers
        plot_scene_wheels_spearman_condensed()

    if 'error_vs_radius' in analyses:
        # Mean abs error per radius (humans vs. models)
        plot_scene_wheels_error_vs_radius()

    if 'error_hist' in analyses:
        plot_scene_wheels_error_hist()

    # Maybe not worth it...(TODO: Automatically make combined plot for paper)
    if 'error_spearman_combined' in analyses:
        # Make combined plot including both error_vs_radius and
        # spearman_condensed
        pth1_png = 'figures/summary/scene_wheels/spearman_bars_condensed.png'
        pth2_png = 'figures/summary/scene_wheels/error_per_radius.png'
        im1 = Image.open(pth1_png)
        im2 = Image.open(pth2_png)
        w = max(im1.width, im2.width)
        h = im1.height + im2.height
        canvas = Image.new('RGB', (w, h))
        canvas.paste(im1)
        canvas.paste(im2, (0, im1.height))
        canvas.save('figures/summary/scene_wheels/scene_wheels_results_combined.pdf')


def make_tables():
    def _make(df):
        df['pval_bins'] = pd.cut(df.spearman_pval, [0, 0.001, 0.01, 0.1, 0.5, 1.0])
        df['pval_thresh'] = df.pval_bins.apply(lambda x: x.right)
        df['Model'] = df.model_name.str.rsplit('_', n=1, expand=True)[0]
        df.Model = df.Model.str.replace('rgb', 'RGB channel means')
        df.Model = df.Model.str.replace('pixels', 'Pixels')
        df.Model = df.Model.str.replace('vae', 'VAE')
        df.Model = df.Model.str.replace('vgg19', 'VGG-19')
        df.Model = df.Model.str.replace('clip_RN50', 'CLIP-RN50')
        df.Model = df.Model.str.replace('clip_ViT-B16', 'CLIP-ViT-B16')
        df['Spearman'] = df.spearman_r.round(2).astype(str) + f' ($p < ' + df.pval_thresh.astype(str) + ' $)'
        df['Log-likelihood'] = df.loglik.round().astype(int)
        print(df.loc[:, ['Model', 'Spearman', 'Log-likelihood']].to_latex(escape=False, index=False))

    print('SCENE WHEEL RESULTS')
    layers_sw, dprimes_sw, df_sw = get_best_fit_models(
        'scene_wheels',
        model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
        binsize=12,
        return_df=True,
        verbose=True,
    )
    df = pd.concat([
        df_sw.loc[(df_sw.model_name == layer) & (df_sw.dprime == dp) & (df_sw.radius == 'all')]
        for layer, dp in zip(layers_sw, dprimes_sw)
    ])
    _make(df)

    print('TAYLOR BAYS RESULTS')
    for tag in ['setsize1', 'setsize2', 'setsize4', 'setsize8', 'all']:
        layers_tb, dprimes_tb, df_tb = get_best_fit_models(
            f'bays2014_{tag}',
            model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
            binsize=12,
            return_df=True,
            verbose=True,
        )
        df = pd.concat([
            df_tb.loc[(df_tb.model_name == layer) & (df_tb.dprime == dp)]
            for layer, dp in zip(layers_tb, dprimes_tb)
        ])
        _make(df)

    print('BRADY ALVAREZ RESULTS')
    layers_brady, dprimes_brady, df_brady = get_best_fit_models(
        'brady_alvarez',
        model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
        binsize=12,
        return_df=True,
        verbose=True,
    )
    df = pd.concat([
        df_brady.loc[(df_brady.model_name == layer) & (df_brady.dprime == dp)]
        for layer, dp in zip(layers_brady, dprimes_brady)
    ])
    _make(df)


def make_regression_features_table():
    table = pd.DataFrame({
        'Feature name': [
            r'radius',
            r'disk\_size',
            r'vae\_beta0.01',
            r'countr\_estimate',
            r'keypoints\_2d',
            r'keypoints\_3d',
            r'seg\_2d',
            r'seg\_25d',
            r'vgg19\_l*\_mean\_abs',
            r'vgg19\_l*\_spatial\_entropy',
        ],
        'Description': [
            'Radius in GAN space. Not a feature of stimulus but of response set. Larger radius means response alternatives are more distinct from stimulus and each other, and therefore result in less difficult trials. Included in regression to allow predictions to account for this stimulus-independent source of variability in human error.',
            'Disk storage size for stimulus images (here we used JPEG).',
            r'$\mathrm{mean}(\mathrm{abs}(z))$, where $z$ are the activation values corresponding to $\mu$ in $\beta$-VAE encoder output layer. This model was the same as the one used to create a baseline TCC model (see Main). Its $\beta$ value was $0.01$ and it was trained on the Places-365 dataset. Due to regularization toward the zero-mean prior, large activations magnitudes should generally be rare, but more complex images may elicit higher magnitudes.',
            'Output of CounTR model, which estimates number of objects in an image.',
            r'$\mathrm{mean}(\mathrm{abs}(y))$, where $y$ is the output activation map from the 2D keypoints model in the Midlevel-vision repository. Higher values indicate higher confidence in the presence of a 2D keypoint. Images with more points of interest may be more difficult to remember. (Based on SURF features.)',
            r'Same as keypoints\_2d except with 3D keypoints. (Based on NARF features.)',
            'Disk size of image output from Midlevel-vision unsupervised segmentation (2D) model, which is based on gestalt principles. Images with greater gestalt complexity may be more difficult to remember. (Note that PCA is applied to raw outputs to reduce to 3 image channels, following method by authors).',
            r'Same as seg\_2d, except with 2.5D gestalt features.',
            r'$\mathrm{mean}(\mathrm{abs}(y))$, where $y$ are hidden activations from VGG19 trained on ImageNet-1k at layer k. Due to regularization, activations are sparse. Images with higher visual load may elicit larger activations.',
            r'$H(y)$, where $y$ are hidden activations from VGG19 trained on ImageNet-1k at layer k, and H is the "spatial" entropy. This measure increases to the extent that points of interest (as indicated by non-zero activations) are more evenly distributed across the image. This may be a relevant factor if people prefer focal attention over diffuse.',
        ],
    })
    with pd.option_context("max_colwidth", 1000):
        # table.style.to_latex(
        #     buf='figures/summary/scene_wheels/regression_features.txt',
        #     escape=False,
        #     index=False,
        #     caption='Description of features used in trial-difficulty regression analysis',
        #     column_format='p{2cm}|p{5cm}',
        #     label='regression_features'
        # )
        print(
            table.style.format_index(escape="latex").hide(axis='index').to_latex(
                caption='Description of features used in trial-difficulty regression analysis',
                # column_format='p{2cm}|p{5cm}',
                label='regression_features'
            )
        )


def dnn_arch_comparison():
    summary_pths = Path(
        f'{TCC_PATH}/scene_wheels_analysis/'
    ).glob('summary_data_*.csv')
    data = pd.concat(
        [pd.read_csv(pth) for pth in summary_pths], ignore_index=True
    )
    data = data[data.radius == 'all']
    all_models = [
        'vgg19',
        'resnet50',
        'harmonized_RN50',
        'convnext_base',
        'convnext_base_1k',
        'convnext_large',
        'convnext_large_1k',
        'clip_RN50',
        'clip_RN50x4',
        'clip_RN50x16',
        'clip_RN101',
        'clip_ViT-B16',
    ]
    df = pd.DataFrame({
        'name': [],
        'model_class': [],
        'label': [],
        'num_params': [],
        'num_training_images': [],
        'max_spearman': [],
        'max_LL': [],
        'r_at_max_LL': [],
    })
    for name in all_models:
        if 'convnext' in name:
            model_class = 'convnext'
            label = 'ConvNext'
            if '1k' in name:
                num_training_images = 1.28e6
            else:
                num_training_images = 1.42e7
        elif 'clip' in name:
            model_class = 'clip'
            label = 'CLIP'
            num_training_images = 4e8
        else:
            label = name
            if 'places' in name:
                model_class = 'places'
                num_training_images = 1e7
            if 'vae' in name:
                model_class = 'vae'
                num_training_images = 1e7
            else:
                model_class = 'torchvision'
                num_training_images = 1.28e6
        data_i = data[data.model_name.str.startswith(name)]
        # Get spearman rho for model with maximum likelihood against human data
        r_at_max_LL = data_i.loc[data_i.loglik.idxmax()].spearman_r
        dfi = pd.DataFrame({
            'name': [name],
            'model_class': [model_class],
            'label': [label],
            'num_params': [count_model_params(name)],
            'num_training_images': [num_training_images],
            'max_spearman': [data_i.spearman_r.max()],
            'max_LL': [data_i.loglik.max()],
            'r_at_max_LL': [r_at_max_LL],
        })
        df = pd.concat([df, dfi])

    max_params = df.num_params.max()

    fig, ax = plt.subplots()
    for lab, dfg in df.groupby('label'):
        ax.scatter(
            dfg.num_training_images,
            # dfg.max_spearman,
            dfg.r_at_max_LL,
            label=lab,
            s=dfg.num_params / max_params * 1200,
            alpha=0.5,
        )

    handles = [
        ax.annotate(name, (x, y), fontsize=12)
        for name, x, y in zip(df.name, df.num_training_images, df.r_at_max_LL)
        # for name, x, y in zip(df.name, df.num_training_images, df.max_spearman)
    ]
    from adjustText import adjust_text
    adjust_text(handles, avoid_points=False, avoid_text=True, only_move={'text': 'y'})
    ax.set_xscale('log')
    ax.set_xlabel('# training images')
    ax.set_ylabel('Max Spearman rho (all trials)')
    plt.tight_layout()
    plt.savefig('figures/summary/scene_wheels/dnn_comparison_all.pdf')


if __name__ == '__main__':
    import sys

    # PRINT RESULTS TABLES FOR MODEL FITS (ALL EXPERIMENTS)
    # make_tables()

    # PRINT INFO ABOUT BEST-FIT MODELS
    # exps = [
    #     'scene_wheels', 'brady_alvarez', 'bays2014_setsize1',
    #     'bays2014_setsize2', 'bays2014_setsize4', 'bays2014_setsize8',
    #     'bays2014_all',
    # ]
    # exps = ['scene_wheels', 'bays2014_all', 'brady_alvarez']
    # for exp in exps:
    #     print(exp)
    #     get_best_fit_models(
    #         exp,
    #         model_classes=['vgg19', 'clip_RN50', 'clip_ViT-B16'],
    #         binsize=12,
    #         verbose=True,
    #         return_df=False,
    #     )

    # SCENE WHEELS: DNN SELECTED MODELS
    # plot_scene_wheels(
    #     analyses=[
    #         # 'spearman_full',
    #         # 'spearman_condensed',
    #         # 'error_vs_radius',
    #         # 'error_spearman_combined',
    #         'error_hist',
    #     ],
    # )

    # SCENE WHEELS: DNN FULL MODEL COMPARISON
    # dnn_arch_comparison()

    # SCENE WHEELS: TRIAL DIFFICULTY REGRESSION ANALYSIS
    # if len(sys.argv) > 1:
    #     reg_type = sys.argv[1]
    # else:
    #     reg_type = 'linear'
    # print(f'REGRESSION TYPE: {reg_type}')
    # stimulus_difficulty_analysis(
    #     ext='jpg', reg_type=reg_type, shuffle_baseline=False
    # )
    # Print feature descriptions as latex table
    # make_regression_features_table()

    # SCENE WHEELS: RANK TRANSFORMATION SCATTERS
    # plot_scene_wheels_rank_transform()

    # ORIENTATION WM
    plot_taylor_bays(
        model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19'],
        analyses=[
            # 'biases',
            # 'spearman_full',
            # 'spearman_condensed',
            # 'quantiles',
            # 'setsize',
            # 'similarity_curves',
            'setsize_and_similarity_curves',
        ],
    )

    # COLOR WM
    # plot_brady(
    #     model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19'],
    #     # model_classes=['finetuned_clip_RN50', 'finetuned_clip_ViT-B16'],
    #     analyses=[
    #         # 'similarity_curves',
    #         # 'biases',
    #         'spearman_full',
    #         # 'spearman_condensed',
    #         # 'quantiles',
    #         # 'hierarchical',
    #     ],
    # )

    # COMBINED COLOR AND ORIENTATION BIAS PLOT
    # plot_bias_color_orientation(
    #     binsize=12,
    #     model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19'],
    # )

    # COMBINED COLOR AND ORIENTATION BAR PLOT
    # plot_ss_spearman_condensed(
    #     binsize=12,
    #     model_classes=['clip_RN50', 'clip_ViT-B16', 'vgg19'],
    # )
