import tarfile
from pathlib import Path
from torchvision.datasets import Places365


if __name__ == '__main__':
   import sys
   if len(sys.argv) > 1:
      i_start = int(sys.argv[1])
   else:
      i_start = 0

   places365_pth = "/n/gershman_lab/lab/cjbates/pytorch_datasets/Places365"
   # split_pth = Path(places365_pth).joinpath("data_256")
   split_pth = Path(places365_pth).joinpath("data_large_standard")
   shard_pth = split_pth.joinpath("tar")
   shard_pth.mkdir(exist_ok=True)

   ds = Places365(places365_pth)
   X, _ = ds.load_file_list()
   img_pths = [x[0] for x in X]

   num_imgs = len(img_pths)
   shard_size = 10000
   num_shards = num_imgs // shard_size
   if num_imgs % shard_size:
      num_shards += 1

   for shard in range(i_start, num_shards):
      idx0 = shard * shard_size
      idx1 = idx0 + shard_size
      idx1 = min(num_imgs, idx1)
      savepth = shard_pth.joinpath(f"dataset_shard{shard}.tar")
      print(savepth)
      print(f"Shard {shard} of {num_shards - 1}")
      with tarfile.open(savepth, mode="w", dereference=True) as fid:
         for i in range(idx0, idx1):
            fid.add(img_pths[i])
