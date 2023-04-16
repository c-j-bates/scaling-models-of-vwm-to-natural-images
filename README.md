# scaling-models-of-vwm-to-natural-images
Code used in analyses for https://www.biorxiv.org/content/10.1101/2023.03.17.533050v1.full.pdf
 
This repo contains analysis scripts and Pytorch training scripts, which are meant to be modified on an ad hoc basis, depending on user needs.
I recommend modifying code as you see fit (e.g. to change file paths, comment out parts of analyses you want to omit, etc.).
 
INSTALLATION
 
The most critical package is PyTorch. I followed official PyTorch instructions to use conda environments to install: 
Python (3.9)  
Pytorch (1.13)  
Torchvision (0.14)  
 
On top of these, I installed compatible versions of:  
openai-clip (https://pypi.org/project/openai-clip/)  
timm (https://pypi.org/project/timm/)  
huggingface_hub (https://huggingface.co/docs/huggingface_hub/installation)  
datasets (by Huggingface: https://pypi.org/project/datasets/)  
transformers (by Huggingface: https://huggingface.co/docs/transformers/installation)  
harmonization (by Serre et al.: https://pypi.org/project/Harmonization/)  
visualpriors (https://pypi.org/project/visualpriors/)  
scikit-image  
scikit-learn  
tqdm  
pymatreader (https://pypi.org/project/pymatreader/)  
adjusttext (for one particular plotting analysis; https://pypi.org/project/adjustText/)  
pandas  
matplotlib  
 
For one analysis, I used CounTR (https://github.com/Verg-Avesta/CounTR). I copied their repo locally and created an environment variable to specify its filepath (see tcc_plotting.py).
 
Not all of these packages may be necessary for all analyses, so if you are experiencing compatibility issues, I recommend commenting out code you don't need.
 
USAGE: 
 
To fit TCC models for different DNN architectures on human responses in Scene Wheels dataset and save results to file: 
 
python TCC_modeling.py --model-classes vgg19 clip_RN50 clip_ViT-B16 --scene-wheels-summary  
 
To produce the DNN comparison scatter plot:  
 
python TCC_modeling.py --model-classes vgg19 clip_RN50 clip_ViT-B16 --arch-comparison  
 
To do the same but on VWM datasets: 
  
`python TCC_modeling.py --model-classes vgg19 clip_RN50 clip_ViT-B16 --taylor-bays`  
`python TCC_modeling.py --model-classes vgg19 clip_RN50 clip_ViT-B16 --brady-alvarez`  
 
(Note that these analyses are very resource-intensive and can take a long time to run. I used a high-performance computing cluster with GPUs, many CPU cores, and large storage. I also ran models in parallel for more efficient resource use. My scripts save various intermediate results to file to enable restarts. I have not included these intermediate files in this repo due to their large sizes. Also note that the variable `attention_method` and related variables refer to functionality I ended up abandoning. Same for attention-related variables related to training the VAE.) 
 
To run analyses:  
 
`python tcc_plotting.py`  
 
(with desired set of analyses uncommented at bottom of script) 

