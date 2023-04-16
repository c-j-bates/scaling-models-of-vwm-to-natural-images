import clip
import numpy as np
import os
from pathlib import Path
from stimuli import make_colors_ring, rgb_from_angle
import torch
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CLIP_CACHE = os.environ.get('CLIP_CACHE')
CHECKPOINT_DIR = os.environ.get(
    "TORCH_CKPT_DIR",
    "/n/gershman_lab/lab/cjbates/pytorch_checkpoints/"
)
CHECKPOINT_DIR = Path(CHECKPOINT_DIR).joinpath(
    "psychophysical_scaling"
)


def convert_models_to_fp32(model):
    for p in model.parameters(): 
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


class clip_colors_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_preprocesser,
        img_size=224,
        setsizes=[1, 2, 3, 4],
        lab_center=[70., 20., 38.],
        lab_radius=60.,
        color_names_path='color_names.txt',
        examples_per_epoch=10000,
    ):
        self.examples_per_epoch = examples_per_epoch  # Set arbitrarily
        self.img_size = img_size
        self.setsizes = setsizes
        self.sqsizes = np.arange(39, 40)
        self.lab_center = lab_center
        self.lab_radius = lab_radius
        self.color_names_database = {}
        self.rgb_focal_set = []
        with Path(color_names_path).open('r') as fid:
            lines = fid.readlines()
            for line in lines:
                if not line.startswith('!') and 'grey' not in line:  # Filter duplicates
                    x = line.split(' ')
                    x = [xi for xi in x if xi]
                    name = x[-1].strip()
                    rgb = [int(xi) for xi in x[:-1]]
                    if len(rgb) != 3:
                        raise Exception()
                    self.rgb_focal_set.append(rgb)
                    self.color_names_database[tuple(rgb)] = name
        self.rgb_focal_set = np.array(self.rgb_focal_set)
        self.img_preprocesser = img_preprocesser

    def __len__(self):
        return self.examples_per_epoch

    def get_nearest_color_name(self, rgb):
        dists = np.linalg.norm(
            self.rgb_focal_set - np.array(rgb).reshape(1, -1), axis=1
        )
        nearest = tuple(self.rgb_focal_set[np.argmin(dists)])
        name = self.color_names_database[nearest]
        return name

    def make_caption(self, rgbs):
        color_names = [self.get_nearest_color_name(rgb) for rgb in rgbs]
        # caption = f'A blank background with '
        # for name in color_names[:-1]:
        #     caption += f'a {name} patch, '
        # if len(rgbs) > 1:
        #     caption += f'and a {color_names[-1]} patch'
        # else:
        #     caption += f'a {color_names[-1]} patch'
        caption = ' '.join(color_names)
        return caption

    def __getitem__(self, idx):
        # Randomly sample
        values = np.random.choice(360, size=np.random.choice(self.setsizes))
        rgbs = [rgb_from_angle(v) * 255 for v in values]
        sqsize = np.random.choice(self.sqsizes)
        img = make_colors_ring(
            values * np.pi / 180,
            size=self.img_size,
            sqsize=sqsize,
            lab_center=self.lab_center,
            radius=self.lab_radius,
        )[0]
        image = self.img_preprocesser(img)
        caption = clip.tokenize(self.make_caption(rgbs)).squeeze()
        return image, caption


def train_clip(
    model_name, batch_size=50, start_epoch=-1, end_epoch=100
):
    # Borrowing from:
    # https://github.com/openai/CLIP/issues/83

    save_path = Path(f'{CHECKPOINT_DIR}').joinpath(
        f"clip_{model_name.replace('/', '')}_finetune.pth"
    )

    model, preprocess = clip.load(
        # Must set jit=False for training
        model_name, device=device, jit=False, download_root=CLIP_CACHE
    )
    model.to(device)
    dataset = clip_colors_dataset(preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.98),
        eps=1e-6,
        # weight_decay=0.2
        weight_decay=0.001
    )

    for epoch in (pbar := tqdm(range(start_epoch + 1, end_epoch), dynamic_ncols=True)):
        running_loss = 0
        running_n_correct = 0
        running_count = 0
        for batch, (images, captions) in enumerate(loader):
            
            images = images.to(device)
            captions = captions.to(device)

            # x1 = model.encode_text(captions)
            # x2 = model.encode_text(captions)
            # print(torch.all(x1 == x2))

            logits_per_image, logits_per_caption = model(images, captions)
            ground_truth = torch.arange(
                len(images),
                dtype=torch.long,
                device=device
            )

            loss1 = torch.nn.CrossEntropyLoss()(logits_per_image, ground_truth)
            loss2 = torch.nn.CrossEntropyLoss()(logits_per_caption, ground_truth)
            loss = (loss1 + loss2) / 2

            n_correct = torch.sum(
                torch.argmax(logits_per_image, axis=1) == ground_truth
            ) / len(images)
            running_n_correct += n_correct

            optimizer.zero_grad()
            loss.backward()
            if device == "cpu":
                optimizer.step()
            else: 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            running_loss += loss

            running_count += 1

            if batch % 10 == 0:
                pbar.set_description(
                    f'Running loss: {running_loss / running_count}, correct: ({running_n_correct / running_count})'
                )
            if batch % 100 == 0:
                running_loss = 0
                running_n_correct = 0
                running_count = 0

        torch.save(
            {
                'epoch': epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            save_path
        )

    return model

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'RN50'
    train_clip(model_name)
