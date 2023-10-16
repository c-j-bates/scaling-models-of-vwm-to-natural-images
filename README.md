# scaling-models-of-vwm-to-natural-images
Code used in analyses for https://www.biorxiv.org/content/10.1101/2023.03.17.533050v1.full.pdf
 
This repo contains analysis scripts and Pytorch training scripts, which are meant to be modified on an ad hoc basis, depending on user needs.
I recommend modifying code as you see fit (e.g. to change file paths, comment out parts of analyses you want to omit, etc.).

Scripts tested on both CentOS Linux 7 (Core) and macOS Monterrey.
 
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
p_tqdm  
pymatreader (https://pypi.org/project/pymatreader/)  
adjusttext (for one particular plotting analysis; https://pypi.org/project/adjustText/)  
pandas  
matplotlib  
webdataset (https://pypi.org/project/webdataset/0.1.25/)
 
For one analysis, I used CounTR (https://github.com/Verg-Avesta/CounTR). I copied their repo locally and created an environment variable to specify its filepath (see tcc_plotting.py).
 
Not all of these packages may be necessary for all analyses, so if you are experiencing compatibility issues, I recommend commenting out code you don't need. Installation time for Python packages on typical laptop hardware should a few minutes.

DATA  

Human data for the scene wheels experiments can be found at scene_wheels_mack_lab_osf/Data/sceneWheel_main_data_n20.csv. For the color working memory experiments, it can be found at brady_alvarez/dataTestAllDegreesOrd.mat. For the orientation WM experiments it can be found at taylor_bays/Exp1_bays2014.mat.

 
USAGE 
 
To fit TCC models for different DNN architectures on human responses in Scene Wheels dataset and save results to file: 
 
`python TCC_modeling.py --model-classes rgb pixels vgg19 clip_RN50 clip_ViT-B16 --scene-wheels-summary`  

Code can be tested on the 'rgb' baseline model, which is less resource-intensive:

`python TCC_modeling.py --model-classes rgb --scene-wheels-summary`  
 
To produce the DNN comparison scatter plot:  
 
`python TCC_modeling.py --arch-comparison`  

(The set of models to be compared is hard-coded in the dnn_arch_comparison function. These need to be run beforehand using the commands above using the --scene-wheels-summary flag.)
 
To fit models on VWM datasets, do: 
  
`python TCC_modeling.py --model-classes vgg19 clip_RN50 clip_ViT-B16 --taylor-bays`  
`python TCC_modeling.py --model-classes vgg19 clip_RN50 clip_ViT-B16 --brady-alvarez`  
 
(Note that these analyses are very resource-intensive and can take a long time to run. I used a high-performance computing cluster with GPUs, many CPU cores, and large storage. I also ran models in parallel for more efficient resource use. My scripts save various intermediate results to file to enable restarts. I have not included these intermediate files in this repo due to their large sizes. Also note that the variable `attention_method` and related variables refer to functionality I ended up abandoning. Same for attention-related variables related to training the VAE.) 
 
To run analyses:  
 
`python tcc_plotting.py`  
 
(with desired set of analyses uncommented at bottom of script) 

