from PIL import Image, ImageDraw
import numpy as np
from itertools import product
from skimage import color as skcolor
from skimage.filters import gabor_kernel


def angle_from_rgb(rgb, lab_center=[60., 22., 14.], radius=52.):
    """
    Convert single RGB pixel to CIELAB and then get angle w.r.t. the center
    point (given by lab_center).
    """
    rgb = np.array(rgb)
    L0, a0, b0 = lab_center
    lab = skcolor.rgb2lab(rgb / 255.)
    # angle = np.arctan2(lab[1] - a0, lab[2] - b0)
    angle = np.arctan2(lab[2] - b0, lab[1] - a0)
    return angle


def rgb_from_angle(angle, lab_center=[60., 22., 14.], radius=52.):
    """
    Default values from:
    https://www.nature.com/articles/s41467-019-11298-3
    """
    L, a, b = lab_center
    lab = [L, np.cos(angle) * radius + a, np.sin(angle) * radius + b]
    rgb = skcolor.lab2rgb(lab)
    return rgb


def make_colors_ring(
    hues,
    locs=None,
    size=64,
    eccentricity=0.25,
    sqsize=2,
    spatial_jitter=((0, 0), (0, 0)),
    angle_jitter=(0, 0),
    color=True,
    lab_center=[60., 22., 14.],
    radius=52.,
):
    """
    VWM stimulus with annulus and colored squares

    spatial_jitter shifts the annulus location randomly.
    angle_jitter shifts rotates the annulus randomly, after placing items.
    """
    if len(hues) > 8:
        raise Exception(
            f'Max number of locations is 8, but {len(hues)} values given.'
        )
    rad = size * eccentricity
    if locs is None:
        # By default, space items evenly
        angles = np.linspace(0, 2 * np.pi, len(hues), endpoint=False)
    elif type(locs[0]) is int:
        # Locations get passed in as indexes. Convert to angles.
        angles_set = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angles = angles_set[np.array(locs)]
    elif type(locs[0]) is float:
        # Assume locs given as angles in degrees
        angles = np.array(locs) * np.pi / 180
    else:
        raise Exception('Unexpected data type for variable "locs"')
    angle_jit = np.random.uniform(
        angle_jitter[0] - angle_jitter[1],
        angle_jitter[0] + angle_jitter[1],
    )
    angles += angle_jit
    xjit = np.random.choice(
        np.arange(
            spatial_jitter[0][0] - spatial_jitter[1][0],
            spatial_jitter[0][0] + spatial_jitter[1][0] + 1
        )
    )
    yjit = np.random.choice(
        np.arange(
            spatial_jitter[0][1] - spatial_jitter[1][1],
            spatial_jitter[0][1] + spatial_jitter[1][1] + 1
        )
    )
    xcoords = np.array([rad * np.cos(a) for a in angles]) + size / 2 + xjit - sqsize / 2
    xcoords = np.round(xcoords).astype(int)
    ycoords = np.array([rad * np.sin(a) for a in angles]) + size / 2 + yjit - sqsize / 2
    ycoords = np.round(ycoords).astype(int)
    sqgrid = np.meshgrid(
        np.arange(0, sqsize).astype(int), np.arange(0, sqsize).astype(int)
    )
    canvas = np.ones((size, size, 3)) * 255
    for hue, x0, y0 in zip(hues, xcoords, ycoords):
        if x0 < 0 or x0 > size - 1 or y0 < 0 or y0 > size - 1:
            raise Exception()
        if color:
            hue = rgb_from_angle(hue, lab_center=lab_center, radius=radius)
        canvas[sqgrid[0] + x0, sqgrid[1] + y0] = hue * 255
    img = Image.fromarray(np.uint8(canvas))
    return img, xjit, yjit, angle_jit, xcoords, ycoords


def make_colored_lines_ring(
    thetas,
    locs,
    line_length=20,
    line_thickness=2,
    size=224,
    eccentricity=0.25,
    spatial_jitter=((0, 0), (0, 0)),
    angle_jitter=(0, 0),
):
    """
    Stimuli from Bays (2014). Oriented colored lines.
    """
    rad = size * eccentricity
    # angles = np.linspace(0, 2 * np.pi, len(thetas), endpoint=False)
    if locs is None:
        # By default, space items evenly
        angles = np.linspace(0, 2 * np.pi, len(thetas), endpoint=False)
    else:
        # Locations get passed in as indexes. Convert to angles.
        angles_set = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angles = angles_set[np.array(locs)]
    angle_jit = np.random.uniform(
        angle_jitter[0] - angle_jitter[1],
        angle_jitter[0] + angle_jitter[1],
    )
    angles += angle_jit
    xjit = np.random.choice(
        np.arange(
            spatial_jitter[0][0] - spatial_jitter[1][0],
            spatial_jitter[0][0] + spatial_jitter[1][0] + 1
        )
    )
    yjit = np.random.choice(
        np.arange(
            spatial_jitter[0][1] - spatial_jitter[1][1],
            spatial_jitter[0][1] + spatial_jitter[1][1] + 1
        )
    )
    xcoords = np.array([rad * np.cos(a) for a in angles]) + size / 2 + xjit
    xcoords = xcoords.astype(int)
    ycoords = np.array([rad * np.sin(a) for a in angles]) + size / 2 + yjit
    ycoords = ycoords.astype(int)

    colors_set = [
        tuple((rgb_from_angle(a) * 255).astype(int))
        for a in np.linspace(0, np.pi * 2, 9)[:8]
    ]
    colors = np.random.choice(len(colors_set), replace=False, size=len(thetas))
    # colors = np.arange(len(thetas))  # Just keep colors fixed for simplicity
    canvas = np.ones((size, size)) * 255 // 2
    img = Image.fromarray(np.uint8(canvas)).convert("RGB")
    for c, theta, x0, y0 in zip(colors, thetas, xcoords, ycoords):
        w = line_length * np.cos(theta)
        h = line_length * np.sin(theta)
        img1 = ImageDraw.Draw(img)
        img1.line(
            [(x0 - w // 2, y0 - h // 2), (x0 + w // 2, y0 + h // 2)],
            fill=colors_set[c],
            width=line_thickness,
        )
    return img, xjit, yjit, angle_jit, xcoords, ycoords


def make_gabors_ring(
    thetas,
    locs,
    size=64,
    eccentricity=0.25,
    spatial_jitter=((0, 0), (0, 0)),
    angle_jitter=(0, 0),
    sigx=1.,
    sigy=1.5,
):

    kernels0 = [
        np.real(gabor_kernel(0.25, theta=theta, sigma_x=sigx, sigma_y=sigy)).T
        for theta in thetas
    ]
    kernels = []
    for kernel in kernels0:
        kernel = kernel.copy()
        kmin, kmax = np.min(kernel), np.max(kernel)
        extremum = max([np.abs(kmin), np.abs(kmax)])
        kernel = (kernel / extremum / 2 + 0.5) * 255
        kernels.append(kernel)

    rad = size * eccentricity
    # angles = np.linspace(0, 2 * np.pi, len(thetas), endpoint=False)
    if locs is None:
        # By default, space items evenly
        angles = np.linspace(0, 2 * np.pi, len(thetas), endpoint=False)
    else:
        # Locations get passed in as indexes. Convert to angles.
        angles_set = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angles = angles_set[np.array(locs)]
    angle_jit = np.random.uniform(
        angle_jitter[0] - angle_jitter[1],
        angle_jitter[0] + angle_jitter[1],
    )
    angles += angle_jit
    xjit = np.random.choice(
        np.arange(
            spatial_jitter[0][0] - spatial_jitter[1][0],
            spatial_jitter[0][0] + spatial_jitter[1][0] + 1
        )
    )
    yjit = np.random.choice(
        np.arange(
            spatial_jitter[0][1] - spatial_jitter[1][1],
            spatial_jitter[0][1] + spatial_jitter[1][1] + 1
        )
    )
    xcoords = np.array([rad * np.cos(a) for a in angles]) + size / 2 + xjit
    xcoords = xcoords.astype(int)
    ycoords = np.array([rad * np.sin(a) for a in angles]) + size / 2 + yjit
    ycoords = ycoords.astype(int)
    canvas = np.ones((size, size)) * 255 // 2
    for kernel, x0, y0 in zip(kernels, xcoords, ycoords):
        shp = kernel.shape
        x1 = x0 - shp[0] // 2
        y1 = y0 - shp[1] // 2
        if x1 < 0 or x1 > size - 1 or y1 < 0 or y1 > size - 1:
            raise Exception()
        sqgrid = np.meshgrid(
            np.arange(0, shp[0]).astype(int), np.arange(0, shp[1]).astype(int)
        )
        canvas[sqgrid[0] + x1, sqgrid[1] + y1] = kernel[sqgrid[0], sqgrid[1]]
    img = Image.fromarray(np.uint8(canvas)).convert("RGB")
    return img, xjit, yjit, angle_jit, xcoords, ycoords
