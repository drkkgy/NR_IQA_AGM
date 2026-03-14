'''
Author: Ankit Yadav
Date: 2025-03-26
'''
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import os
from diffusers import StableDiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel,List,Tuple,UNet2DConditionOutput
from diffusers.models.unets.unet_2d_blocks import *
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, pearsonr
import scipy
import h5py
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import ImageFilter
import random
from mvlearn.embed import GCCA      # works for 2+ views
from sklearn.decomposition import PCA
import numpy as np
from seed import *
import cv2
from PIL import Image
import torch.fft
import warnings
import math

# Defining Some Custom Noise to be addedd to images

from torch.utils.data import Subset

def root_dataset(ds):
    """Follow .dataset links until we hit the real dataset
    (i.e. the first object that is *not* a Subset)."""
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds

def add_gaussian_blur(image_tensor, sigma=1.0):
    # Convert tensor to PIL, apply blur, convert back
    pil_img = TF.to_pil_image(image_tensor)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return TF.to_tensor(blurred)

def add_salt_and_pepper_noise(image_tensor, amount=0.02):
    c, h, w = image_tensor.shape
    noisy = image_tensor.clone()
    num_salt = int(amount * h * w)
    num_pepper = int(amount * h * w)

    # Salt (white pixels)
    for _ in range(num_salt):
        i, j = random.randint(0, h-1), random.randint(0, w-1)
        noisy[:, i, j] = 1.0

    # Pepper (black pixels)
    for _ in range(num_pepper):
        i, j = random.randint(0, h-1), random.randint(0, w-1)
        noisy[:, i, j] = 0.0

    return noisy

def add_speckle_noise(image_tensor, scale=0.2):
    noise = torch.randn_like(image_tensor) * scale
    noisy = image_tensor + image_tensor * noise
    return torch.clamp(noisy, 0.0, 1.0)

# Defining the text template for DP-IQA

scenes = 'animal, cityscape, human, indoor, landscape,night, plant, still life, food, water, other'
distortion_type = 'jpeg2000 compression, jpeg compression, motion, white noise, gaussian blur, fastfading, fnoise, lens, diffusion, shifting, color quantization, desaturation, oversaturation, underexposure, overexposure, contrast, white noise with color, impulse, multiplicative, jitter, white noise with denoise, brighten, darken, pixelate, shifting the mean, noneccentricity patch, quantization, color blocking, sharpness, realistic blur, realistic noise, realistic contrast change, other realistic, other, fnoise low‑freq'
quaility_level = 'bad, poor, fair, good, perfect'

#BAD_QUALITY_PROMPT = "Summer is the best time to go to the beach"
BAD_QUALITY_PROMPT="A photograph with severe JPEG2000 compression artifacts, heavy JPEG blocking, intense motion blur, overwhelming white noise, pronounced pixelation and color quantization, very low contrast, severe under- and over-exposure, and overall poor image quality and other poor image quality properties."
#BAD_QUALITY_PROMPT = "Blur,jpeg2000 compression, jpeg compression, motion, white noise, gaussian blur, fastfading, fnoise, lens, diffusion, shifting, color quantization, desaturation, oversaturation, underexposure, overexposure, contrast, white noise with color, impulse, multiplicative, jitter,"# white noise with denoise, brighten, darken, pixelate, shifting the mean, noneccentricity patch, quantization, color blocking, sharpness, realistic blur, realistic noise, realistic contrast change, other realistic, other, fnoise low‑freq" #"Image of with severe JPEG2000 and JPEG compression artifacts, motion blur, white noise, pixelation, color quantization, desaturation, oversaturation, underexposure, overexposure, low contrast, and other distortions."
#BAD_QUALITY_PROMPT ="{} exhibiting severe JPEG2000 compression artifacts, heavy JPEG blocking, intense motion blur, overwhelming white noise, pronounced pixelation, significant color quantization, very low contrast, severe underexposure and overexposure, sensor noise, and other poor image quality artifacts."


# Quality only Prompts

Quality_prompts = ["a photo with {} artifacts, perceived as {} quality".format(distortion, quality) for distortion in distortion_type.split(',') for quality in quaility_level.split(',')]


Text_Template = []

for scene in scenes.split(','):
    for distortion in distortion_type.split(','):
        for quality in quaility_level.split(','):
            Text_Template.append(f'a photo of a {scene} with {distortion} distortion, which is of {quality} quality')


scene_prompts = ["a photo of a " + scene.strip() for scene in scenes.split(',')]

Text_Template_baseline = "A high-resolution photo with visible distortions such as {distortion_type} focus on its visual quality."

def custom_prompt_and_noise(batch_size,tokenizer, text_encoder,image_tensor ,device='cpu'):
    noise_options = {
        "Gaussian Blur": add_gaussian_blur,
        "Salt and Pepper Noise": add_salt_and_pepper_noise,
        "Speckle Noise": add_speckle_noise
    }

    # Randomly choose one noise type
    selected_name, selected_fn = random.choice(list(noise_options.items()))

    # Apply noise per image in batch
    noisy_batch = torch.stack([selected_fn(img) for img in image_tensor])

    # Generate corresponding prompt
    Text_Template = "A high-resolution photo with visible distortions such as {} with additional {} distortion. Focus on its visual quality."
    prompt = Text_Template.format(distortion_type,selected_name)

    # Replicating the prompts to match the batch size
    scale_prompt_by_batch = [prompt] * batch_size
    inputs = tokenizer(scale_prompt_by_batch, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    # Convert token IDs to embeddings using the text encoder
    text_embeddings = text_encoder(inputs)[0]

    return text_embeddings.to(device), noisy_batch.to(torch.bfloat16).to(device)

# Defining the Prompt Processor for DP-IQA

def prompt_processor(prompt_list,batch_size,tokenizer, text_encoder, device='cpu'):
    old_device = device
    # Check for exisitng embeddings
    if os.path.exists(f'/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/universal_embedding_{batch_size}.pt'):
        #print('Loading the embeddings from disk')
        embedding = torch.load(f'/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/universal_embedding_{batch_size}.pt')
        embed_batch_size = embedding.shape[0]
        if embed_batch_size != batch_size:
            raise ValueError(f'Batch size {embed_batch_size} mismatch current config {batch_size}. Please delete the existing embeddings file and recreate it as its misconfigured')
        return embedding.to(old_device)
    else:
        print(f'Creating the embeddings from scratch for batch size: {batch_size}')
        # To save memory, we will use CPU for tokenization and GPU for the rest of the operations
        device = "cpu"
        # Move the text encoder to the device
        text_encoder = text_encoder.to(device)


        # Replicating the prompts to match the batch size
        if len(prompt_list) != 1700:
            raise ValueError('Prompt list must have 1700 elements')
        

        K = len(prompt_list)
        embed_list = []
        # Tokenize prompts
        for i in tqdm(range(K), desc='Tokenizing prompts'):
            
            scale_prompt_by_batch = [prompt_list[i]] * batch_size 
            inputs = tokenizer(scale_prompt_by_batch, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            # Convert token IDs to embeddings using the text encoder
            text_embeddings = text_encoder(inputs)[0]

            # Now performing average pooling to get a single embedding for each prompt as discussed in the paper DP-IQA: https://arxiv.org/abs/2405.19996
            avg_pooled_text_embeddings = torch.mean(text_embeddings, dim=1)
            embed_list.append(avg_pooled_text_embeddings)
            del text_embeddings
            del avg_pooled_text_embeddings
            torch.cuda.empty_cache()

        universal_embedding = torch.stack(embed_list, dim=0)
        universal_embedding = universal_embedding.permute(1,0,2)

        del embed_list
        torch.cuda.empty_cache()

        # Save the embeddings to disk
        torch.save(universal_embedding, f'universal_embedding_{batch_size}.pt')

        return universal_embedding.to(old_device)

def process_prompts_base(prompt,batch_size,tokenizer, text_encoder, device='cpu'):
    scale_prompt_by_batch = [prompt] * batch_size 
    inputs = tokenizer(scale_prompt_by_batch, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    # Convert token IDs to embeddings using the text encoder
    text_embeddings = text_encoder(inputs)[0]

    return text_embeddings



# Defining custom trainable adapters for DP-IQA as shown in paper DP-IQA: https://arxiv.org/abs/2405.19996

# Setting of this adapter are taken from the paper https://arxiv.org/pdf/2110.04544 as discussed in the paper DP-IQA: https://arxiv.org/abs/2405.19996

# Setting up Text Adapters
class TextAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden=192):
        super(TextAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),    # We will test with GELU next to see if that works better
            nn.Linear(hidden, input_dim)
            )
    def forward(self, x):
        return x + self.adapter(x) # Residual connection
    
# Setting up Image Adapters

# Bulding Blocks for the Image Adapter

#Adapted from the paper https://arxiv.org/abs/2302.08453 as discussed in the paper DP-IQA: https://arxiv.org/abs/2405.19996

class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            #nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU()
    
    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x

# Adapted from the paper https://arxiv.org/abs/2302.08453 as discussed in the paper DP-IQA: https://arxiv.org/abs/2405.19996
class DownsampleBlock(nn.Module):
    def __init__(self,in_channel,stride=2):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1), 
            #nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        x = self.downsample(x)
        return x


class ImageAdapter(nn.Module):
    def __init__(self, input_dim=2048, hidden=256):
        super(ImageAdapter, self).__init__()

        self.residual_block11 = ResidualBlock(320, 320)
        self.residual_block12 = ResidualBlock(320, 320)

        self.residual_block21 = ResidualBlock(640, 640)
        self.residual_block22 = ResidualBlock(640, 640)

        self.residual_block31 = ResidualBlock(1280, 1280)
        self.residual_block32 = ResidualBlock(1280, 1280)

        self.residual_block41 = ResidualBlock(1280, 1280)
        self.residual_block42 = ResidualBlock(1280, 1280)


        self.downsample_block2 = DownsampleBlock(320)
        self.downsample_block3 = DownsampleBlock(640)
        self.downsample_block4 = DownsampleBlock(1280)
        self.conv1 = nn.Conv2d(192, 320, kernel_size=3, stride=1, padding=3//2)
        self.conv2 = nn.Conv2d(320, 640, kernel_size=3, stride=1, padding=3//2)
        self.conv3 = nn.Conv2d(640, 1280, kernel_size=3, stride=1, padding=3//2)
        self.conv4 = nn.Conv2d(1280, 1280, kernel_size=3, stride=1, padding=3//2)
        
    def forward(self, x):
        
        F_i = []
        # Performing pixel unshuffle to reduce the spatial resolution of the image as discussed in the paper https://arxiv.org/pdf/2302.08453 
        # mentioned in DP-IQA: https://arxiv.org/abs/2405.19996 
        x = torch.nn.functional.pixel_unshuffle(x, 8)  # 512x512x3 -> 64x64x192
        x = self.conv1(x)
        x = self.residual_block11(x)
        x = self.residual_block12(x)
        F_i.append(x)

        x = self.downsample_block2(x)
        x = self.conv2(x)
        x = self.residual_block21(x)
        x = self.residual_block22(x)
        F_i.append(x)
        x = self.downsample_block3(x)
        x = self.conv3(x)
        x = self.residual_block31(x)
        x = self.residual_block32(x)
        F_i.append(x)
        x = self.downsample_block4(x)
        x = self.conv4(x)
        x = self.residual_block41(x)
        x = self.residual_block42(x)
        F_i.append(x)

        return F_i 
    

# Implementing the Quality Feature Decoder Adapter.

# Creating eh function to extract the features from the unet upsample layers

class FeatureExtractor:
    def __init__(self):
        self.features = defaultdict(list)

    def save_features(self, module, input, output):
        module_name = str(module.__class__).split('.')[-1].split("'")[0]
        self.features[module_name].append(output)

    def get_features(self):
        x = self.features
        F1 = x['UpBlock2D'][0]
        F2,F3,F4 = x['CrossAttnUpBlock2D'][0], x['CrossAttnUpBlock2D'][1], x['CrossAttnUpBlock2D'][2]
        Features = [F1,F2,F3,F4]
        return Features

def register_hooks(model, feature_extractor,modules_to_hook=(CrossAttnUpBlock2D, UpBlock2D)):
    for name, module in model.named_modules():
        if isinstance(module,modules_to_hook ):
            module.register_forward_hook(feature_extractor.save_features)


# Squeeze-and-Excite block
class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Global average pooling
        y = F.silu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)
    


class QFD_adapter(nn.Module):
    def __init__(self, in_channels_list=[1280,1280,640,320],upsample_size=(64,64), mid_channels=512):
        """
        in_channels_list: list of ints, channels of input features [1280,1280,640,320]
        """
        super(QFD_adapter, self).__init__()

        self.adapters = nn.ModuleList()
        for in_ch in in_channels_list:
            self.adapters.append(nn.Sequential(
                nn.Upsample(size=upsample_size, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1),
                SELayer(mid_channels)
            ))

        # Conv1x1 projection head
        self.proj = nn.Sequential(
                nn.Conv2d(mid_channels * len(in_channels_list), 512, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(512, 128, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(128, 32, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(32, 8, kernel_size=1)  # Final output: [B, 8, 64, 64]
            )
        


    def forward(self, x):
        Features = x
        # F1 = x['UpBlock2D'][0]
        # F2,F3,F4 = x['CrossAttnUpBlock2D'][0], x['CrossAttnUpBlock2D'][1], x['CrossAttnUpBlock2D'][2]
        # Features = [F1,F2,F3,F4]
        processed = [adapter(feat) for adapter, feat in zip(self.adapters, Features)]
    
        fused = torch.cat(processed, dim=1)
        out = self.proj(fused)
        # Flatten the output
        out = out.view(out.size(0), -1)
        return out

def process_cleandift_features(feature_dict):
    """
    Process features from the feature_dict and return a list of dictionaries with processed features.
    """
    processed_features = [feature_dict['us3'],
                          feature_dict['us6'],
                          feature_dict['us9'],
                          feature_dict['us10']]
    return processed_features

class QFD_adapter_max_pool(nn.Module):
    def __init__(self, in_channels_list=[1280,1280,640,320],upsample_size=(64,64), mid_channels=512):
        """
        in_channels_list: list of ints, channels of input features [1280,1280,640,320]
        """
        super(QFD_adapter_max_pool, self).__init__()

        self.adapters = nn.ModuleList()
        for in_ch in in_channels_list:
            self.adapters.append(nn.Sequential(
                nn.Upsample(size=upsample_size, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1),
                SELayer(mid_channels)
            ))

        # Conv1x1 projection head
        
        self.proj_max_pool = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(128, 32, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(32, 8, kernel_size=1)  # Final output: [B, 8, 64, 64]
            )


    def forward(self, x):
        Features = x
        # F1 = x['UpBlock2D'][0]
        # F2,F3,F4 = x['CrossAttnUpBlock2D'][0], x['CrossAttnUpBlock2D'][1], x['CrossAttnUpBlock2D'][2]
        # Features = [F1,F2,F3,F4]
        processed = [adapter(feat) for adapter, feat in zip(self.adapters, Features)]
        
    
        fused = torch.stack(processed, dim=1)
        fused = torch.max(fused, dim=1)[0]
        out = self.proj_max_pool(fused)
        # Flatten the output
        out = out.view(out.size(0), -1)
        return out

class QFD_adapter_avg_pool(nn.Module):
    def __init__(self, in_channels_list=[1280,1280,640,320],upsample_size=(64,64), mid_channels=512):
        """
        in_channels_list: list of ints, channels of input features [1280,1280,640,320]
        """
        super(QFD_adapter_avg_pool, self).__init__()

        self.adapters = nn.ModuleList()
        for in_ch in in_channels_list:
            self.adapters.append(nn.Sequential(
                nn.Upsample(size=upsample_size, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1),
                SELayer(mid_channels)
            ))

        # Conv1x1 projection head
        
        self.proj_avg_pool = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(128, 32, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(32, 8, kernel_size=1)  # Final output: [B, 8, 64, 64]
            )


    def forward(self, x):
        Features = x
        # F1 = x['UpBlock2D'][0]
        # F2,F3,F4 = x['CrossAttnUpBlock2D'][0], x['CrossAttnUpBlock2D'][1], x['CrossAttnUpBlock2D'][2]
        # Features = [F1,F2,F3,F4]
        processed = [adapter(feat) for adapter, feat in zip(self.adapters, Features)]
        
    
        fused = torch.stack(processed, dim=1)
        fused = torch.mean(fused, dim=1)
        out = self.proj_avg_pool(fused)
        # Flatten the output
        out = out.view(out.size(0), -1)
        return out




def feature_extract_and_save(features,scores):
    Features = features
    Processed_Features = []
    for i in range(scores.shape[0]):
        feature_dict = {
            'F1': Features[0][i].cpu(),
            'F2': Features[1][i].cpu(),
            'F3': Features[2][i].cpu(),
            'F4': Features[3][i].cpu(),
            'score': scores[i].cpu()

        }
        Processed_Features.append(feature_dict)
    
    return Processed_Features


def feature_extract_and_save_siglip(features,scores):
    Features = features
    Processed_Features = []
    for i in range(scores.shape[0]):
        feature_dict = {
            'F1': Features[i].cpu(),
            'score': scores[i].cpu()

        }
        Processed_Features.append(feature_dict)
    
    return Processed_Features


def process_and_save_embeddings(embedding_dict,h5_file="embeddings.h5"):
    embedding_dict = embedding_dict
    F1 = []
    F2 = []
    F3 = []
    F4 = []
    scores = []

    for i in range(len(embedding_dict)):
        feature_dict = embedding_dict[i]
        F1.append(feature_dict['F1'].to(torch.float16))
        F2.append(feature_dict['F2'].to(torch.float16))
        F3.append(feature_dict['F3'].to(torch.float16))
        F4.append(feature_dict['F4'].to(torch.float16))
        scores.append(feature_dict['score'])
    del embedding_dict
    F1 = torch.stack(F1, dim=0).to('cpu').numpy()
    F2 = torch.stack(F2, dim=0).to('cpu').numpy()
    F3 = torch.stack(F3, dim=0).to('cpu').numpy()
    F4 = torch.stack(F4, dim=0).to('cpu').numpy()
    scores = torch.stack(scores, dim=0).to('cpu').numpy()

    np.savez("embeddings.npz",
    F1=F1,
    F2=F2,
    F3=F3,
    F4=F4,
    scores=scores
    )

    
    
def process_and_save_embeddings_hdf5(embedding_dict, h5_file="embeddings.h5",no_of_thread=12):
    num_items = len(embedding_dict)
    
    # Example shape, replace with your actual shape
    C0, H0, W0 = embedding_dict[0]['F1'].shape
    C1, H1, W1 = embedding_dict[0]['F2'].shape
    C2, H2, W2 = embedding_dict[0]['F3'].shape
    C3, H3, W3 = embedding_dict[0]['F4'].shape

    with h5py.File(h5_file, "w") as hf:

        F1_ds = hf.create_dataset("F1", shape=(num_items, C0, H0, W0), dtype='float16', chunks=True)
        F2_ds = hf.create_dataset("F2", shape=(num_items, C1, H1, W1), dtype='float16', chunks=True)
        F3_ds = hf.create_dataset("F3", shape=(num_items, C2, H2, W2), dtype='float16', chunks=True)
        F4_ds = hf.create_dataset("F4", shape=(num_items, C3, H3, W3), dtype='float16', chunks=True)
        scores_ds = hf.create_dataset("scores", shape=(num_items,), dtype='float32', chunks=True)

        for i in tqdm(range(num_items), desc='Saving embeddings'):
            feature_dict = embedding_dict[i]
            F1_ds[i] = feature_dict['F1'].to(torch.float16).cpu().numpy()
            F2_ds[i] = feature_dict['F2'].to(torch.float16).cpu().numpy()
            F3_ds[i] = feature_dict['F3'].to(torch.float16).cpu().numpy()
            F4_ds[i] = feature_dict['F4'].to(torch.float16).cpu().numpy()
            scores_ds[i] = feature_dict['score'].cpu().numpy()

    print("Embeddings saved efficiently using HDF5.")    

def process_and_save_embeddings_siglip_hdf5(embedding_dict, h5_file="embeddings_siglip.h5",no_of_thread=12):
    num_items = len(embedding_dict)
    
    c = embedding_dict[0]['F1'].shape[0]
    with h5py.File(h5_file, "w") as hf:

        F1_ds = hf.create_dataset("F1", shape=(num_items,c), dtype='float16', chunks=True)
        scores_ds = hf.create_dataset("scores", shape=(num_items,), dtype='float32', chunks=True)

        for i in tqdm(range(num_items), desc='Saving embeddings'):
            feature_dict = embedding_dict[i]
            F1_ds[i] = feature_dict['F1'].to(torch.float16).cpu().numpy()
            scores_ds[i] = feature_dict['score'].cpu().numpy()

    print("Embeddings saved efficiently using HDF5 for SIGLIP.")    

def load_embeddings_hdf5(h5_file="embeddings.h5"):
    with h5py.File(h5_file, "r") as hf:
        F1 = torch.from_numpy(hf["F1"][:]).to(torch.float16)
        F2 = torch.from_numpy(hf["F2"][:]).to(torch.float16)
        F3 = torch.from_numpy(hf["F3"][:]).to(torch.float16)
        F4 = torch.from_numpy(hf["F4"][:]).to(torch.float16)
        scores = torch.from_numpy(hf["scores"][:]).to(torch.float32)

    return {'F1':F1,'F2':F2,'F3':F3,'F4':F4,'scores':scores}

def load_embeddings_hdf5_siglip(h5_file="embeddings_siglip.h5"):
    with h5py.File(h5_file, "r") as hf:
        F1 = torch.from_numpy(hf["F1"][:]).to(torch.float16)
        scores = torch.from_numpy(hf["scores"][:]).to(torch.float32)

    return {'F1':F1,'scores':scores}

class ClassifyThenRegress(BaseEstimator, RegressorMixin):
    def __init__(self, classifier, regressors, bins):
        self.classifier = classifier
        self.regressors = regressors
        self.bins       = bins

    def fit(self, X, y):   # already fitted; return self for API compliance
        return self

    def predict(self, X):
        c_hat = self.classifier.predict(X)
        preds = np.empty(len(X))
        for c in np.unique(c_hat):
            mask = c_hat == c
            preds[mask] = self.regressors[c].predict(X[mask])
        return preds


# Defining the 3 layer MLP for the final prediction as shown in the paper DP-IQA: https://arxiv.org/abs/2405.19996
class mlp_3_layer(nn.Module):
    def __init__(self, input_dim=32768, hidden=256):
        super(mlp_3_layer, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
            )
    def forward(self, x):
        return self.adapter(x)
    
class mlp_3_layer_sigmoid(nn.Module):
    def __init__(self, input_dim=32768, hidden=256):
        super(mlp_3_layer_sigmoid, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1)
            )
    def forward(self, x):
        return self.adapter(x)
    
class mlp_3_layer_sigmoid_fusion(nn.Module):
    def __init__(self, input_dim=32768, hidden=256,siglip_input_dim=1024):
        super(mlp_3_layer_sigmoid_fusion, self).__init__()
        #self.a_params = nn.Parameter(torch.ones(1))  # learnable parameter for scaling
        self.projection = nn.Linear(input_dim, siglip_input_dim)
        self.adapter = nn.Sequential(
            nn.Linear(siglip_input_dim, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1)
            )
    def forward(self, unet_x,siglip_x):

        x = (1-0.5)*self.projection(unet_x) + 0.5*siglip_x

        return self.adapter(x)
    
class mlp_3_layer_siglip(nn.Module):
    def __init__(self, input_dim=1024, hidden=512):
        super(mlp_3_layer_siglip, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
            )
    def forward(self, x):
        return self.adapter(x)
    

# Trying out different variants of MLP layers

class DynamicWeightMLP_3_Layer(nn.Module):
    def __init__(self, input_dim=1024, hidden=512):
        super(DynamicWeightMLP_3_Layer, self).__init__()
        self.layer_aux1 = nn.Linear(input_dim, hidden)

        self.layer1 = nn.Linear(input_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        #self.layer3 = nn.Linear(hidden, 1)
        self.b3 = nn.Parameter(torch.zeros(1))  # learnable bias

    def forward(self, x):
        W_aux = self.layer_aux1(x)
        h = self.relu(self.layer1(x))
        h = self.relu(self.layer2(h))
        # Combine the outputs
        W_aux = W_aux.squeeze(1).expand_as(h)  # Expand W_aux to match h's shape
        out = (W_aux * h).sum(dim=1, keepdim=True) + self.b3
        return out

# MLP with sigmoid activation

# Defining the 3 layer MLP for the final prediction as shown in the paper DP-IQA: https://arxiv.org/abs/2405.19996
class mlp_3_layer(nn.Module):
    def __init__(self, input_dim=32768, hidden=512):
        super(mlp_3_layer, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
            )
    def forward(self, x):
        return self.adapter(x)
    

# Approach 1: Using Sigmoid activation function in early layers    
# class mlp_3_layer_sigmoid_siglip(nn.Module):
#     def __init__(self, input_dim=1024, hidden=512):
#         super(mlp_3_layer_sigmoid_siglip, self).__init__()
#         self.adapter = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, 1)
#             )
#     def forward(self, x):
#         return self.adapter(x)

# Approach 2: Using Sigmoid activation function in last layers
class mlp_3_layer_sigmoid_siglip(nn.Module):
    def __init__(self, input_dim=1024, hidden=512):
        super(mlp_3_layer_sigmoid_siglip, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Sigmoid(), #  It was ReLU before
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(), #  It was ReLU before
            nn.Linear(hidden, 1),
            # #  No activation before
            )
    def forward(self, x):
        return self.adapter(x)
    

class mlp_3_layer_sigmoid_siglip_scaled_activation(nn.Module):

    def __init__(self, input_dim=1024, hidden=512):
        super().__init__()
        # first two layers → produce the embedding
        self.layer1 = nn.Linear(input_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, 1)

        # the three activations
        self.act_sig  = nn.Sigmoid()
        self.act_leak = nn.LeakyReLU()
        self.act_gelu = nn.GELU()
        self.act_tanh  = nn.Tanh()

        # learnable mixing parameters
        eps  = 1e-2
        init = (torch.ones(4) / 4 + torch.randn(4) * eps).float()  # ~0.25±0.01
        self.mix1 = nn.Parameter(init.clone())  # keep fp32 for precision
        self.mix2 = nn.Parameter(init.clone())

    def _mixed_activation(self, x, mix):
        """
        x: Tensor of shape [B, D]
        mix: nn.Parameter of shape [4]
        returns: weighted sum of [sigmoid(x), leaky(x), gelu(x), tanh(x)]
        """
        acts = torch.stack([
            self.act_sig(x),
            self.act_leak(x),
            self.act_gelu(x),
            self.act_tanh(x),
        ], dim=0)                      # shape [4, B, D]
        
        weights = F.softmax(mix, dim=0) # shape [4], sums to 1
        # broadcast to [4, 1, 1], multiply & sum over first dim → [B, D]
        return (weights[:, None, None] * acts).sum(dim=0)

    def forward(self, x):
        # pass through first two layers
        h = self.layer1(x)
        h = self._mixed_activation(h,self.mix1)
        h = self.layer2(h)
        h = self._mixed_activation(h,self.mix2)

        # now get the score
        score = self.layer3(h)

        # return both
        return score
    
class mlp_3_layer_sigmoid_siglip_scaled_activation_norm(nn.Module):

    def __init__(self, input_dim=1024, hidden=512):
        super().__init__()
        # first two layers → produce the embedding
        self.layer1 = nn.Linear(input_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, 1)

        # the three activations
        self.act_sig  = nn.Sigmoid()
        self.act_leak = nn.LeakyReLU()
        self.act_gelu = nn.GELU()
        self.act_tanh  = nn.Tanh()

        # learnable mixing parameters
        eps  = 1e-2
        init = (torch.ones(4) / 4 + torch.randn(4) * eps).float()  # ~0.25±0.01
        self.mix1 = nn.Parameter(init.clone())  # keep fp32 for precision
        self.mix2 = nn.Parameter(init.clone())

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def _mixed_activation(self, x, mix):
        """
        x: Tensor of shape [B, D]
        mix: nn.Parameter of shape [4]
        returns: weighted sum of [sigmoid(x), leaky(x), gelu(x), tanh(x)]
        """
        acts = torch.stack([
            self.act_sig(x),
            self.act_leak(x),
            self.act_gelu(x),
            self.act_tanh(x),
        ], dim=0)                      # shape [4, B, D]
        
        weights = F.softmax(mix, dim=0) # shape [4], sums to 1
        # broadcast to [4, 1, 1], multiply & sum over first dim → [B, D]
        return (weights[:, None, None] * acts).sum(dim=0)

    def forward(self, x):
        # pass through first two layers
        h = self.layer1(x)
        h = self.norm1(h)  # Normalize the output of the first layer
        h = self._mixed_activation(h,self.mix1)
        h = self.layer2(h)
        h = self.norm2(h)  # Normalize the output of the second layer
        h = self._mixed_activation(h,self.mix2)


        # now get the score
        score = self.layer3(h)

        # return both
        return score

    def get_mixing_weights(self):
        """
        Returns a dict with the soft-maxed weights so we can log/inspect them.
        Keys: mix1_sig, mix1_leaky, …, mix2_tanh
        """
        with torch.no_grad():
            w1 = F.softmax(self.mix1, dim=0).cpu().tolist()
            w2 = F.softmax(self.mix2, dim=0).cpu().tolist()
        keys = ["sig", "leaky", "gelu", "tanh"]
        return {f"mix1_{k}": w1[i] for i, k in enumerate(keys)} | {
               f"mix2_{k}": w2[i] for i, k in enumerate(keys)}
    
# Lernable Sigmoid

class ParamSigmoid(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # one α and β per hidden‐dim (or use dim=1 for a single scalar)
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x is [B, dim]
        return torch.sigmoid(self.alpha * x + self.beta)

    
class mlp_3_layer_sigmoid_siglip_lernable(nn.Module):

    def __init__(self, input_dim=1024, hidden=512):
        super().__init__()
        # first two layers → produce the embedding
        self.layer1 = nn.Linear(input_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, 1)

        # activations
        self.act_sig  = ParamSigmoid(hidden)
        self.act_leak = nn.LeakyReLU()



    def forward(self, x):
        # pass through first two layers
        h = self.layer1(x)
        h = self.act_sig(h)
        h = self.layer2(h)
        h = self.act_leak(h)

        # now get the score
        score = self.layer3(h)

        # return both
        return score
    

class mlp_3_layer_sigmoid_siglip_embed(nn.Module):
    def __init__(self, input_dim=1024, hidden=512):
        super().__init__()
        # first two layers → produce the embedding
        self.layer1 = nn.Linear(input_dim, hidden)
        self.act1   = nn.Sigmoid()          # was ReLU
        self.layer2 = nn.Linear(hidden, hidden)
        self.act2   = nn.LeakyReLU()        # was ReLU

        # final layer → produces a single score
        self.layer3 = nn.Linear(hidden, 1)

    def forward(self, x):
        # pass through first two layers
        h = self.layer1(x)
        h = self.act1(h)
        embed = self.layer2(h)
        embed = self.act2(embed)

        # now get the score
        score = self.layer3(embed)

        # return both
        return score, embed
# Testing with Fourier Neural Operator (FNO) for the MLP layer 

class mlp_3_layer_sigmoid_cross_attention_patchwise(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Ebeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_sigmoid_cross_attention_patchwise, self).__init__()

        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                                num_heads=n_heads,
                                                batch_first=True)
        # self.cross_attn2 = nn.MultiheadAttention(embed_dim=input_dim,
        #                                         num_heads=n_heads,
        #                                         batch_first=True)
        # self.cross_attn3 = nn.MultiheadAttention(embed_dim=input_dim,
        #                                         num_heads=n_heads,
        #                                         batch_first=True)

        # self.self_attn = nn.MultiheadAttention(embed_dim=input_dim,
        #                                         num_heads=n_heads,
        #                                         batch_first=True)

        self.text_projection = nn.Linear(input_dim*2, input_dim)  # Project text embeddings to match image embeddings
        #self.alpha = nn.Parameter(torch.tensor(0.0))  # Learnable parameter for scaling text embeddings Best case has no alpha


        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.Sigmoid() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)


        #self.layernorm = nn.LayerNorm(input_dim)
        #self.attn_pool = nn.Linear(input_dim, 1)


        # #  No activation before
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        # cls_x = F.normalize(cls_x, p=2, dim=-1)  # Normalize the class token embeddings
        x_seq = x # [B, N, 1024]
        # x_seq = F.normalize(x_seq, p=2, dim=-1)  # Normalize the image embeddings
        # # import pdb
        # # pdb.set_trace()
        rd_x = tx_seq * cls_x
        tx_combined = torch.cat((tx_seq, rd_x), dim=-1)  # [B, M, 2048]
        tx_seq = self.text_projection(tx_combined)


        # attn_out, _ = self.cross_attn(query=x_seq, key=tx_seq,value=tx_seq)
        # attn_out, _ = self.cross_attn(query=attn_out, key=tx_seq,value=tx_seq)
        # attn_out, _ = self.cross_attn(query=attn_out, key=tx_seq,value=tx_seq)

        attn_out, _ = self.cross_attn(query=x_seq, key=tx_seq,value=tx_seq)
        #attn_out,_ = self.self_attn(query=attn_out, key=attn_out,value=attn_out)  # Self attention on the image embeddings  
        #attn_out = (1-self.alpha)*attn_out + self.alpha*x_seq  # Residual connection
        attn_out, _ = self.cross_attn(query=attn_out, key=tx_seq,value=tx_seq)
        #attn_out,_ = self.self_attn(query=attn_out, key=attn_out,value=attn_out)
        #attn_out = (1-self.alpha)*attn_out + self.alpha*x_seq  # Residual connection
        attn_out, _ = self.cross_attn(query=attn_out, key=tx_seq,value=tx_seq)
        #attn_out,_ = self.self_attn(query=attn_out, key=attn_out,value=attn_out)
        #attn_out = (1-self.alpha)*attn_out + self.alpha*x_seq  # Residual connection
        #attn_out, _ = self.cross_attn(query=attn_out, key=tx_seq,value=tx_seq)

        # MixFuse Style  https://www.sciencedirect.com/science/article/pii/S0888327025002328?utm_source=chatgpt.com
        #attn_out,_ = self.self_attn(query=attn_out, key=attn_out,value=attn_out)
        # attn_out,_ = self.self_attn(query=attn_out, key=attn_out,value=attn_out)
        # attn_out,_ = self.self_attn(query=attn_out, key=attn_out,value=attn_out) 
        
        # print(attn_out.shape)
        # attn_out is [B, 1, D]; remove seq dim
        
        #attn_out,_ = self.cross_attn2(query=attn_out, key=tx_seq,value=tx_seq)

        # attn_out,_ = self.cross_attn3(query=attn_out, key=tx_seq,value=tx_seq)

        #Adaptive Pooling
        #weights = F.softmax(self.attn_pool(attn_out), dim=1)
        #h = (attn_out * weights).sum(dim=1)
        h = attn_out.max(dim=1).values
        
        #diff = x - h  # [B, D]
        # Concatenating the image and cross attention output
        #h = torch.cat((cls_x.squeeze(1),h), dim=-1)  # [B, N + D]
        # Apply LayerNorm to the attention output
        #h = self.layernorm(h)  # [B, D]

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h) # Bilinear layer
        h = self.acvt2(h)
        out = self.fc3(h)
        return out
    
class mlp_3_layer_sigmoid_cross_attention_patchwise_simple_cross_attn(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Ebeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_sigmoid_cross_attention_patchwise_simple_cross_attn, self).__init__()

        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                                num_heads=n_heads,
                                                batch_first=True)
       
        self.text_projection = nn.Linear(input_dim, input_dim)  # Project text embeddings to match image embeddings

        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.Sigmoid() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        
        x_seq = x # [B, N, 1024]
        tx_seq = self.text_projection(tx_seq)


        attn_out1, _ = self.cross_attn(query=x_seq, key=tx_seq,value=tx_seq)
        attn_out1  = attn_out1 + x_seq  # Residual connection
        attn_out2, _ = self.cross_attn(query=attn_out1, key=tx_seq,value=tx_seq)
        attn_out2  = attn_out2 + attn_out1  # Residual connection
        attn_out3, _ = self.cross_attn(query=attn_out2, key=tx_seq,value=tx_seq)
        attn_out3  = attn_out3 + attn_out2  # Residual connection
        
        h = attn_out3.max(dim=1).values
        

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h)
        h = self.acvt2(h)
        out = self.fc3(h)
        return out
    
class mlp_3_layer_sigmoid_cross_attention_patchwise_simple_cross_attn_layer_norm(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Ebeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_sigmoid_cross_attention_patchwise_simple_cross_attn_layer_norm, self).__init__()

        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                                num_heads=n_heads,
                                                batch_first=True)
       
        self.text_projection = nn.Linear(input_dim, input_dim)  # Project text embeddings to match image embeddings

        self.norm_attn1   = nn.LayerNorm(input_dim)
        self.norm_attn2   = nn.LayerNorm(input_dim)
        self.norm_attn3   = nn.LayerNorm(input_dim)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.Sigmoid() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        
        x_seq = x # [B, N, 1024]
        tx_seq = self.text_projection(tx_seq)


        attn_out1, _ = self.cross_attn(query=x_seq, key=tx_seq,value=tx_seq)
        attn_out1  = self.norm_attn1(attn_out1 + x_seq)  # Residual connection
        attn_out2, _ = self.cross_attn(query=attn_out1, key=tx_seq,value=tx_seq)
        attn_out2  = self.norm_attn2(attn_out2 + attn_out1)  # Residual connection
        attn_out3, _ = self.cross_attn(query=attn_out2, key=tx_seq,value=tx_seq)
        attn_out3  = self.norm_attn3(attn_out3 + attn_out2)  # Residual connection
        
        h = attn_out3.max(dim=1).values
        

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h)
        h = self.acvt2(h)
        out = self.fc3(h)
        return out
    
class mlp_3_layer_cross_attention_patchwise_simple_cross_attn(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Ebeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_cross_attention_patchwise_simple_cross_attn, self).__init__()

        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                                num_heads=n_heads,
                                                batch_first=True)
       
        self.text_projection = nn.Linear(input_dim, input_dim)  # Project text embeddings to match image embeddings

        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.LeakyReLU() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        
        x_seq = x # [B, N, 1024]
        tx_seq = self.text_projection(tx_seq)


        attn_out1, _ = self.cross_attn(query=x_seq, key=tx_seq,value=tx_seq)
        attn_out1  = attn_out1 + x_seq  # Residual connection
        attn_out2, _ = self.cross_attn(query=attn_out1, key=tx_seq,value=tx_seq)
        attn_out2  = attn_out2 + attn_out1  # Residual connection
        attn_out3, _ = self.cross_attn(query=attn_out2, key=tx_seq,value=tx_seq)
        attn_out3  = attn_out3 + attn_out2  # Residual connection
        
        h = attn_out3.max(dim=1).values
        

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h)
        h = self.acvt2(h)
        out = self.fc3(h)
        return out
    
class mlp_3_layer_self_attention_patchwise_simple_self_attn(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Ebeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_self_attention_patchwise_simple_self_attn, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                                num_heads=n_heads,
                                                batch_first=True)
       
        #self.text_projection = nn.Linear(input_dim, input_dim)  # Project text embeddings to match image embeddings

        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.LeakyReLU() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        
        x_seq = x # [B, N, 1024]
        #tx_seq = self.text_projection(tx_seq)


        attn_out1, _ = self.self_attn(query=x_seq, key=x_seq,value=x_seq)
        attn_out1  = attn_out1 + x_seq  # Residual connection
        attn_out2, _ = self.self_attn(query=attn_out1, key=attn_out1,value=attn_out1)
        attn_out2  = attn_out2 + attn_out1  # Residual connection
        attn_out3, _ = self.self_attn(query=attn_out2, key=attn_out2,value=attn_out2)
        attn_out3  = attn_out3 + attn_out2  # Residual connection
        
        h = attn_out3.max(dim=1).values
        

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h)
        h = self.acvt2(h)
        out = self.fc3(h)
        return out

class mlp_3_layer_self_attention_patchwise_simple_self_attn_sigmoid(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Ebeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_self_attention_patchwise_simple_self_attn_sigmoid, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                                num_heads=n_heads,
                                                batch_first=True)
       
        #self.text_projection = nn.Linear(input_dim, input_dim)  # Project text embeddings to match image embeddings

        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.Sigmoid() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        
        x_seq = x # [B, N, 1024]
        #tx_seq = self.text_projection(tx_seq)


        attn_out1, _ = self.self_attn(query=x_seq, key=x_seq,value=x_seq)
        attn_out1  = attn_out1 + x_seq  # Residual connection
        attn_out2, _ = self.self_attn(query=attn_out1, key=attn_out1,value=attn_out1)
        attn_out2  = attn_out2 + attn_out1  # Residual connection
        attn_out3, _ = self.self_attn(query=attn_out2, key=attn_out2,value=attn_out2)
        attn_out3  = attn_out3 + attn_out2  # Residual connection
        
        h = attn_out3.max(dim=1).values
        

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h)
        h = self.acvt2(h)
        out = self.fc3(h)
        return out
    

class mlp_3_layer_sigmoid_cross_attention_patchwise_with_embed_out(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Ebeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_sigmoid_cross_attention_patchwise_with_embed_out, self).__init__()

        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                                num_heads=n_heads,
                                                batch_first=True)

        self.text_projection = nn.Linear(input_dim*2, input_dim)  # Project text embeddings to match image embeddings

        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.Sigmoid() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)

        # #  No activation before
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        x_seq = x # [B, N, 1024]
        rd_x = tx_seq * cls_x
        tx_combined = torch.cat((tx_seq, rd_x), dim=-1)  # [B, M, 2048]
        tx_seq = self.text_projection(tx_combined)


        attn_out, _ = self.cross_attn(query=x_seq, key=tx_seq,value=tx_seq)
       
        attn_out, _ = self.cross_attn(query=attn_out, key=tx_seq,value=tx_seq)
        
        attn_out, _ = self.cross_attn(query=attn_out, key=tx_seq,value=tx_seq)
        
        h = attn_out.max(dim=1).values
        

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h) # Bilinear layer
        h = self.acvt2(h)
        out = self.fc3(h)
        return (out,h)
    

class mlp_3_layer_sigmoid_cross_attention_patchwise_with_embed_out_rationale(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Embeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_sigmoid_cross_attention_patchwise_with_embed_out_rationale, self).__init__()

        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                                num_heads=n_heads,
                                                batch_first=True)

        self.text_projection = nn.Linear(input_dim*2, input_dim)  # Project text embeddings to match image embeddings

        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.Sigmoid() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)

        # #  No activation before
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        x_seq = x # [B, N, 1024]
        rd_x = tx_seq * cls_x
        tx_combined = torch.cat((tx_seq, rd_x), dim=-1)  # [B, M, 2048]
        tx_seq = self.text_projection(tx_combined)


        attn_out, _ = self.cross_attn(query=x_seq, key=tx_seq,value=tx_seq)
       
        attn_out, _ = self.cross_attn(query=attn_out, key=tx_seq,value=tx_seq)
        
        attn_out, _ = self.cross_attn(query=attn_out, key=tx_seq,value=tx_seq)
        
        h = attn_out.max(dim=1).values
        

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h) # Bilinear layer
        h = self.acvt2(h)
        out = self.fc3(h)
        w = self.fc3.weight.squeeze(0)
        r = h*w
        return (out,r)
    
class WeightedHeadCrossAttn(nn.Module):
    def __init__(self, embed_dim: int = 1024, n_heads: int = 16):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.d = embed_dim // n_heads
        self.h = n_heads

        # separate linear projections (q, k, v) so we can keep them differentiable
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # learnable head logits -> softmax'ed every forward
        self.head_logits = nn.Parameter(torch.zeros(n_heads))

    def _reshape(self, x):
        # (B, L, D) -> (B, H, L, d)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d).transpose(1, 2)

    def forward(self, q, k, v):      # q,k,v: (B,L,D)

        q = self._reshape(self.q_proj(q))
        k = self._reshape(self.k_proj(k))
        v = self._reshape(self.v_proj(v))

        # SDP-attention → (B,H,L_q,d)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        

        # learned weights (H,)
        head_w = F.softmax(self.head_logits, dim=0).view(1, -1, 1, 1)
        attn_out = attn_out * head_w           # weight each head, **no sum**

        # concat heads back: (B,L_q,H*d) == (B,L_q,embed_dim)
        B, H, L_q, d = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_q, H * d)

        return self.out_proj(attn_out)         # now shapes match
    
class mlp_3_layer_sigmoid_cross_attention_patchwise_with_embed_out_rationale_weighted_heads(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Embeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_sigmoid_cross_attention_patchwise_with_embed_out_rationale_weighted_heads, self).__init__()

        # self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim,
        #                                         num_heads=n_heads,
        #                                         batch_first=True)

        self.cross1 = WeightedHeadCrossAttn(input_dim, n_heads)
        self.cross2 = WeightedHeadCrossAttn(input_dim, n_heads)
        self.cross3 = WeightedHeadCrossAttn(input_dim, n_heads)
        
        self.text_projection = nn.Linear(input_dim*2, input_dim)  # Project text embeddings to match image embeddings

        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.Sigmoid() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)

        # #  No activation before
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        x_seq = x # [B, N, 1024]
        rd_x = tx_seq * cls_x
        tx_combined = torch.cat((tx_seq, rd_x), dim=-1)  # [B, M, 2048]
        tx_seq = self.text_projection(tx_combined)

        # import pdb
        # pdb.set_trace()
        # Learn head weights
        attn1 = self.cross1(x, tx_seq, tx_seq)
        attn2 = self.cross2(attn1, tx_seq, tx_seq)
        attn3 = self.cross3(attn2, tx_seq, tx_seq)
        
        
        h = attn3.max(dim=1).values
        

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h) # Bilinear layer
        h = self.acvt2(h)
        out = self.fc3(h)
        w = self.fc3.weight.squeeze(0)
        r = h*w
        return (out,r)

class mlp_3_layer_sigmoid_cross_attention_patchwise_with_weighted_heads(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, n_heads=16):
        """
        Input: Image Embeddings of shape [B, input_dim], tx: Text Embeddings of shape [B, input_dim]
        Output: Single scalar Score for each image
        """
        super(mlp_3_layer_sigmoid_cross_attention_patchwise_with_weighted_heads, self).__init__()

        # self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim,
        #                                         num_heads=n_heads,
        #                                         batch_first=True)

        self.cross1 = WeightedHeadCrossAttn(input_dim, n_heads)
        # self.cross2 = WeightedHeadCrossAttn(input_dim, n_heads)
        # self.cross3 = WeightedHeadCrossAttn(input_dim, n_heads)
        
        self.text_projection = nn.Linear(input_dim, input_dim)  # Project text embeddings to match image embeddings

        self.fc1 = nn.Linear(input_dim, hidden)
        self.acvt1 = nn.Sigmoid() #  It was ReLU before
        self.fc2 = nn.Linear(hidden, hidden)
        self.acvt2 = nn.LeakyReLU() #  It was ReLU before
        self.fc3 = nn.Linear(hidden, 1)

        # #  No activation before
            
    def forward(self, x,cls_x, tx,tx_cls): 
        # tx = F.normalize(tx, p=2, dim=1)
        if tx.dim() == 2:
            tx_seq = tx.unsqueeze(1)
        else:
            tx_seq = tx
        
        cls_x = cls_x.unsqueeze(1)  # [B, 1, 1024]
        tx_cls = tx_cls.unsqueeze(1)  # [B, 1, 1024]
        x_seq = x # [B, N, 1024]
        # rd_x = tx_seq * cls_x
        # tx_combined = torch.cat((tx_seq, rd_x), dim=-1)  # [B, M, 2048]
        tx_seq = self.text_projection(tx_seq)

        # import pdb
        # pdb.set_trace()
        # Learn head weights
        attn1 = self.cross1(x, tx_seq, tx_seq)
        attn1 = attn1 + x_seq
        attn2 = self.cross1(attn1, tx_seq, tx_seq)
        attn2 = attn2 + attn1
        attn3 = self.cross1(attn2, tx_seq, tx_seq)
        attn3 = attn3 + attn2
        
        h = attn3.max(dim=1).values
        

        h = self.fc1(h)  # linear layer
        h = self.acvt1(h)
        h = self.fc2(h) # Bilinear layer
        h = self.acvt2(h)
        out = self.fc3(h)
        # w = self.fc3.weight.squeeze(0)
        # r = h*w
        return (out)



class MeanEmbedRegularization:
    def __init__(self,
                 embed_dim:   int = 1024,
                 num_buckets: int = 10,
                 device:      str = 'cuda'):
        self.num_buckets = num_buckets
        self.embedding_dim = embed_dim
        self.device = device
        self.Beta = 0.9999

        # Initialize each bucket mean to zero‐vector, and count to 0
        self.bucket_means  = [
            torch.zeros(self.embedding_dim, device=self.device)
            for _ in range(self.num_buckets)
        ]
        self.bucket_counts = [0 for _ in range(self.num_buckets)]

    def update_buckets(self,
                       embeddings: torch.Tensor,
                       scores:     torch.Tensor):
        """
        embeddings: FloatTensor [B, D], not necessarily normalized.
        scores:     FloatTensor [B], each in [0,1].

        This L2‐normalizes each row of `embeddings` once, picks a bucket
        based on `scores[b]`, and then does a running‐mean update of that bucket.
        """
        B, D = embeddings.shape
        # 1) L2‐normalize each embedding row
        #normalized_embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)  # [B, D]
        normalized_embeddings = embeddings  # Not normalizing here, as we will normalize later in the loss function

        for b_idx in range(B):
            s = scores[b_idx].item()
            bucket_idx = min(int(s / (1/self.num_buckets)), self.num_buckets - 1) # 0.25

            e = normalized_embeddings[b_idx]  # [D]

            count = self.bucket_counts[bucket_idx]
            if count == 0:
                # first vector → mean = normalized e
                self.bucket_means[bucket_idx] = e.clone().detach()
                self.bucket_counts[bucket_idx] = 1
            else:
                old_mean = self.bucket_means[bucket_idx]  # [D]
                new_count = count + 1
                # running average:
                #self.bucket_means[bucket_idx] = F.normalize((self.Beta*(old_mean * count) + (1-self.Beta)*e) / new_count,dim=-1, eps=1e-8).detach()
                self.bucket_means[bucket_idx] = self.Beta * old_mean + (1 - self.Beta) * e
                self.bucket_means[bucket_idx] = F.normalize(self.bucket_means[bucket_idx], dim=-1, eps=1e-8).detach()
                self.bucket_counts[bucket_idx] = new_count

    def bucket_contrastive_loss(self,
                                z:            torch.Tensor,
                                pred_scores:  torch.Tensor,
                                epoch: int,
                                temperature:  float = 0.2, ):

        if epoch < 1:
            # We do not compute loss in the first epoch so that the means can be poulated in one pass
            return torch.tensor(0.0, device=self.device)
        """
        z:            FloatTensor [B, D] (raw embeddings)—we will normalize inside.
        pred_scores:  FloatTensor [B], each in [0, 1].
        temperature:  float (e.g. 0.07)

        Returns: scalar InfoNCE‐style loss, treating each bucket‐mean as one "class."
        """
        B, D = z.shape
        # import pdb
        # pdb.set_trace()
        # 1) Compute true bucket index for each sample [We have 4 buckets, so we scale scores to [0, 3]]
        true_buckets = (pred_scores / (1/self.num_buckets)).floor().long() #0.25        # [B]
        true_buckets = torch.clamp(true_buckets, min=0, max=self.num_buckets - 1)
        true_buckets = true_buckets.squeeze()  # [B]

        # 2) Normalize input embeddings z
        z_norm = F.normalize(z, p=2, dim=-1, eps=1e-8)           # [B, D]

        # 3) Build a [C, D] tensor of bucket means
        bucket_means_tensor = torch.stack(self.bucket_means, dim=0)  # [C, D]
        # (Note: each self.bucket_means[i] may not be unit‐norm,
        #  so we L2‐normalize now)
        bucket_means_norm = F.normalize(bucket_means_tensor, p=2, dim=-1, eps=1e-8)  # [C, D]

        # 4) Cosine‐score logits = z_norm · (bucket_means_norm.T), scaled by 1/temperature
        logits = torch.matmul(z_norm, bucket_means_norm.T) / temperature  # [B, C]

        # 5) Cross‐entropy over these logits
        loss = F.cross_entropy(logits, true_buckets)
        return loss
    
class MeanEmbedRegularization_center_loss:
    def __init__(self,
                 embed_dim:   int = 1024,
                 num_buckets: int = 10,
                 device:      str = 'cuda'):
        self.num_buckets = num_buckets
        self.embedding_dim = embed_dim
        self.device = device
        self.Beta = 0.9999

        # Initialize each bucket mean to zero‐vector, and count to 0
        self.bucket_means  = [
            torch.zeros(self.embedding_dim, device=self.device)
            for _ in range(self.num_buckets)
        ]
        self.bucket_counts = [0 for _ in range(self.num_buckets)]

    def update_buckets(self,
                       embeddings: torch.Tensor,
                       scores:     torch.Tensor):
        """
        embeddings: FloatTensor [B, D], not necessarily normalized.
        scores:     FloatTensor [B], each in [0,1].

        This L2‐normalizes each row of `embeddings` once, picks a bucket
        based on `scores[b]`, and then does a running‐mean update of that bucket.
        """
        B, D = embeddings.shape
        # 1) L2‐normalize each embedding row
        #normalized_embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)  # [B, D]
        normalized_embeddings = embeddings  # Not normalizing here, as we will normalize later in the loss function
        
        for b_idx in range(B):
            s = scores[b_idx].item()
            bucket_idx = min(int(s / (1/self.num_buckets)), self.num_buckets - 1) # 0.25

            e = normalized_embeddings[b_idx]  # [D]

            count = self.bucket_counts[bucket_idx]
            if count == 0:
                # first vector → mean = normalized e
                self.bucket_means[bucket_idx] = e.clone().detach()
                self.bucket_counts[bucket_idx] = 1
            else:
                old_mean = self.bucket_means[bucket_idx]  # [D]
                new_count = count + 1
                # running average:
                #self.bucket_means[bucket_idx] = F.normalize((self.Beta*(old_mean * count) + (1-self.Beta)*e) / new_count,dim=-1, eps=1e-8).detach()
                self.bucket_means[bucket_idx] = self.Beta * old_mean + (1 - self.Beta) * e
                self.bucket_means[bucket_idx] = F.normalize(self.bucket_means[bucket_idx], dim=-1, eps=1e-8).detach()
                self.bucket_counts[bucket_idx] = new_count

    def update_buckets_without_beta(self,
                       embeddings: torch.Tensor,
                       scores:     torch.Tensor):
        """
        embeddings: FloatTensor [B, D], not necessarily normalized.
        scores:     FloatTensor [B], each in [0,1].

        This L2‐normalizes each row of `embeddings` once, picks a bucket
        based on `scores[b]`, and then does a running‐mean update of that bucket.
        """
        B, D = embeddings.shape
        # 1) L2‐normalize each embedding row
        #normalized_embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)  # [B, D]
        normalized_embeddings = embeddings  # Not normalizing here, as we will normalize later in the loss function
        
        for b_idx in range(B):
            s = scores[b_idx].item()
            bucket_idx = min(int(s / (1/self.num_buckets)), self.num_buckets - 1) # 0.25

            e = normalized_embeddings[b_idx]  # [D]

            count = self.bucket_counts[bucket_idx]
            if count == 0:
                # first vector → mean = normalized e
                self.bucket_means[bucket_idx] = e.clone().detach()
                self.bucket_counts[bucket_idx] = 1
            else:
                old_mean = self.bucket_means[bucket_idx]  # [D]
                new_count = count + 1
                # running average:
                #self.bucket_means[bucket_idx] = F.normalize((self.Beta*(old_mean * count) + (1-self.Beta)*e) / new_count,dim=-1, eps=1e-8).detach()
                self.bucket_means[bucket_idx] = (old_mean*count + e)/ new_count
                self.bucket_means[bucket_idx] = F.normalize(self.bucket_means[bucket_idx], dim=-1, eps=1e-8).detach()
                self.bucket_counts[bucket_idx] = new_count

    def bucket_contrastive_loss(self,
                                z:            torch.Tensor,
                                pred_scores:  torch.Tensor,
                                epoch: int,
                                temperature:  float = 0.2, ):

        if epoch < 1:
            # We do not compute loss in the first epoch so that the means can be poulated in one pass
            return torch.tensor(0.0, device=self.device)
        """
        z:            FloatTensor [B, D] (raw embeddings)—we will normalize inside.
        pred_scores:  FloatTensor [B], each in [0, 1].
        temperature:  float (e.g. 0.07)

        Returns: scalar InfoNCE‐style loss, treating each bucket‐mean as one "class."
        """
        B, D = z.shape

        # 1) Compute true bucket index for each sample [We have 4 buckets, so we scale scores to [0, 3]]
        true_buckets = (pred_scores / (1/self.num_buckets)).floor().long() #0.25        # [B]
        true_buckets = torch.clamp(true_buckets, min=0, max=self.num_buckets - 1)
        true_buckets = true_buckets.squeeze()  # [B]

        # 2) Normalize input embeddings z
        z_norm = F.normalize(z, p=2, dim=-1, eps=1e-8)           # [B, D]

        # 3) Get the mean tensor for the corresonding bucket based on GT

        target_mean_tensor = torch.stack([self.bucket_means[cat_index] for cat_index in true_buckets],dim=0)
        # We will normalise the target mean tensors now
        target_mean_tensor = F.normalize(target_mean_tensor, p=2, dim=-1, eps=1e-8)  # [B, D]

        # 4) Now we will calculate MSE for the target mean tensor and the z_norm
        # MSE loss between the normalized embeddings and the target mean tensor
        loss = F.mse_loss(z_norm, target_mean_tensor, reduction='mean')

        return loss
    
    def bucket_contrastive_loss_mae(self,
                                z:            torch.Tensor,
                                pred_scores:  torch.Tensor,
                                epoch: int,
                                temperature:  float = 0.2, ):

        if epoch < 1:
            # We do not compute loss in the first epoch so that the means can be poulated in one pass
            return torch.tensor(0.0, device=self.device)
        """
        z:            FloatTensor [B, D] (raw embeddings)—we will normalize inside.
        pred_scores:  FloatTensor [B], each in [0, 1].
        temperature:  float (e.g. 0.07)

        Returns: scalar InfoNCE‐style loss, treating each bucket‐mean as one "class."
        """
        B, D = z.shape

        # 1) Compute true bucket index for each sample [We have 4 buckets, so we scale scores to [0, 3]]
        true_buckets = (pred_scores / (1/self.num_buckets)).floor().long() #0.25        # [B]
        true_buckets = torch.clamp(true_buckets, min=0, max=self.num_buckets - 1)
        true_buckets = true_buckets.squeeze()  # [B]

        # 2) Normalize input embeddings z
        z_norm = F.normalize(z, p=2, dim=-1, eps=1e-8)           # [B, D]

        # 3) Get the mean tensor for the corresonding bucket based on GT

        target_mean_tensor = torch.stack([self.bucket_means[cat_index] for cat_index in true_buckets],dim=0)
        # We will normalise the target mean tensors now
        target_mean_tensor = F.normalize(target_mean_tensor, p=2, dim=-1, eps=1e-8)  # [B, D]

        # 4) Now we will calculate MSE for the target mean tensor and the z_norm
        # MAE loss between the normalized embeddings and the target mean tensor
        loss = F.l1_loss(z_norm, target_mean_tensor, reduction='mean')

        return loss

class MeanEmbedRegularisationLambda(nn.Module):
    def __init__(self,init_lambda_value = 0.1):
        super(MeanEmbedRegularisationLambda,self).__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(init_lambda_value)))

    def forward(self,reg_loss):
        lambda_ = torch.exp(self.log_lambda)
        return lambda_ * reg_loss,lambda_

# Defining two layer MLP for CLP Loss as discussed in the paper https://arxiv.org/pdf/2104.14746

class mlp_2_layer(nn.Module):
    def __init__(self, input_dim=32768, hidden=128):
        super(mlp_2_layer, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
            )
    def forward(self, x):
        return self.adapter(x)
    


class FNO1DLayer(nn.Module):
    def __init__(self, in_dim, modes):
        super().__init__()
        self.in_dim = in_dim
        self.modes = modes  # number of Fourier modes to keep
        self.weight = nn.Parameter(torch.randn(self.modes, dtype=torch.cfloat))  # complex weights

    def forward(self, x):
        # x: [B, in_dim]
        if x.dtype != torch.float32:
            oringal_dtype = x.dtype
            x = x.to(torch.float32)
        else:
            oringal_dtype = x.dtype

        
        x_ft = torch.fft.rfft(x, dim=-1)  # [B, in_dim//2 + 1]

        # Apply Fourier filter (only to first `modes`)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :self.modes] = x_ft[:, :self.modes] * self.weight[:self.modes]

        # Inverse FFT to go back to spatial domain
        x = torch.fft.irfft(out_ft, n=self.in_dim, dim=-1)

        return x.to(oringal_dtype)  # return to original dtype

class FNO_MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, fno_modes=256): # 256
        super().__init__()
        self.fno = FNO1DLayer(input_dim, fno_modes)
        self.fno2 = FNO1DLayer(input_dim, fno_modes)
        self.fno3 = FNO1DLayer(input_dim, fno_modes)

        # self.adapter1 = nn.Sequential(
        #     nn.Linear(input_dim, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, input_dim),
        # )

        # self.adapter2 = nn.Sequential(
        #     nn.Linear(input_dim, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, input_dim),
        # )

        self.attention = nn.MultiheadAttention(input_dim, num_heads=8,batch_first=True)

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        fno_out = self.fno(x)
        #fno_out = self.adapter1(fno_out) # Small adapters to modulate the input
        fno_out = self.fno2(fno_out)
        #fno_out = self.adapter2(fno_out) # Small adapters to modulate the input
        fno_out = self.fno3(fno_out)
        # using attention to combine the resiudal with the fno output

        attend,_ = self.attention(x, fno_out, fno_out)  # Apply attention
        merged = x + attend
        norm = merged.norm(p=2,dim=-1,keepdim=True)  # L2 normalization
        merged = merged / (norm + 1e-6)  # Avoid division by zero
        fno_out = merged # Concatenate along the last dimension
        return self.adapter(fno_out)  # Residual-style sending only the output of the last FNO layer to the adapter


# FiLM (Feature Wise Linear Modulation Layer )

class FiLM_mlp(nn.Module):
    def __init__(self, input_dim=1024, hidden=256):
        super(FiLM_mlp, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        # self.relu = nn.ReLU(True) # Inplace operation for RELU
        # self.ln1 = nn.LayerNorm(normalized_shape=hidden, eps=1e-5, elementwise_affine=True)
        # self.ln2 = nn.LayerNorm(normalized_shape=hidden, eps=1e-5, elementwise_affine=True)
        # self.ln3 = nn.LayerNorm(normalized_shape=hidden, eps=1e-5, elementwise_affine=True)

        self.out = nn.Linear(hidden, 1)

        # FiLM parameters
#        small MLP to produce [γ, β]
        self.film = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2, 2*hidden)
        )

        # FiLM parameters
# #        small MLP to produce [γ, β]
        self.film2 = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.Sigmoid(),
            nn.Linear(input_dim//2, 2*hidden)
        )

#         self.film3 = nn.Sequential(
#             nn.Linear(input_dim, input_dim//2),
#             nn.ReLU(True),
#             nn.Linear(input_dim//2, 2*hidden)
#         )

        # Initialisingf the FiLM parameters with γ≈1, β≈0 at start
        for film in (self.film,self.film2):#, self.film2, self.film3):
            nn.init.zeros_(film[-1].weight)
            nn.init.ones_( film[-1].bias[:hidden] )
            nn.init.zeros_(film[-1].bias[hidden:] )

        
        self.act1 = nn.Sigmoid()
        self.act2 = nn.LeakyReLU(True)
        self.act3 = nn.LeakyReLU(True)

    def forward(self, x):
        # first block + FiLM
        h1 = self.layer1(x)                # (B, hidden)
        γ, β = self.film2(x).chunk(2, dim=1)
        h = γ * h1 + β   # apply film
        #h = self.ln1(h)                   # apply Layernom
        h1 = self.act1(h) + 0.5*h1

        # second block
        h2 = self.layer2(h1)
        γ2, β2 = self.film(x).chunk(2, dim=1)
        h = γ2 * h2 + β2                    # apply film
        #h = self.ln2(h)                   # apply Layernom
        h2 = self.act2(h) + 0.5*h2

        # third block
        h3 = self.layer3(h2)
        #γ3, β3 = self.film(x).chunk(2, dim=1)
        h = γ2 * h3 + β2                     # apply film
        #h = self.ln3(h)                   # apply Layernom
        h3 = self.act3(h) + 0.5*h3
        
        # final output
        return self.out(h3)

# Defining the MLP layer to extract quality features from siglip embeddings

class mlp_3_layer_siglip_quality_embedding(nn.Module):
    def __init__(self, input_dim=1024, hidden=512,output_dim=128):
        super(mlp_3_layer_siglip_quality_embedding, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
            )
    def forward(self, x):
        return self.adapter(x)
    

# Embedding common feature extraction method using GCCA

def extract_common_features(X, Y, PCA_components=256 ,n_components=128):
    """
    Extract common features from two sets of features using GCCA.

    Args:
        X (numpy.ndarray)/Tensors: First set of features.
        Y (numpy.ndarray)/Tensors: Second set of features.
        n_components (int): Number of components to extract.

    Returns:
        numpy.ndarray: Common features.
    """

    # Ensure X and Y are numpy arrays
    if isinstance(X, torch.Tensor):
        X = X.to(torch.float16).cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.to(torch.float16).cpu().numpy()

    # Do PCA to reduce dimensionality
    pca_X = PCA(n_components=PCA_components).fit_transform(X)
    pca_Y = PCA(n_components=PCA_components).fit_transform(Y)
    
    # Perform GCCA
    gcca = GCCA(n_components=n_components)
    common_features = gcca.fit([X, Y])
    common_X,common_Y = gcca.transform([pca_X, pca_Y]) 
    common_features = 0.5*(common_X + common_Y)   
    return common_features,common_X,common_Y
    
# Defining the Unet model to modifiy the image features as shown in the paper DP-IQA: https://arxiv.org/abs/2405.19996

# Defining the margin loss function

def margin_loss(y, yp, lambda_=0.25):
    """
    Computes the margin loss for image quality ranking.

    Args:
        y (Tensor): Ground truth scores, shape (n,)
        yp (Tensor): Predicted scores, shape (n,)
        lambda_ (float): Scaling factor for margin, usually in [0, 1]

    Returns:
        Tensor: Scalar margin loss
    """
    n = y.size(0)
    sigma_y = torch.std(y, unbiased=False)  # Standard deviation of ground truth
    m = lambda_ * sigma_y

    loss = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            sign = torch.sign(y[i] - y[j])
            if sign != 0:
                diff = -(sign * (yp[i] - yp[j])) + m
                loss += torch.clamp(diff, min=0.0)
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=y.device)
    
    return (2.0 / (n * (n - 1))) * loss



class custome_UNet2dConditionalModel(UNet2DConditionModel):

    def __init__(self, *args, **kwargs):
        # Initialize the parent class with all arguments
        super().__init__(*args, **kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        custom_image_embeddings: Optional[List] = None,
        return_dict: bool = True,
        return_features: bool = False,
        ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""

            The [`UNet2DConditionModel`] forward method.

            Args:
                sample (`torch.Tensor`):
                    The noisy input tensor with the following shape `(batch, channel, height, width)`.
                timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
                encoder_hidden_states (`torch.Tensor`):
                    The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
                class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                    Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
                timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                    Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                    through the `self.time_embedding` layer to obtain the timestep embeddings.
                attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                    An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                    is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                    negative values to the attention scores corresponding to "discard" tokens.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                    `self.processor` in
                    [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                added_cond_kwargs: (`dict`, *optional*):
                    A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                    are passed along to the UNet blocks.
                down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                    A tuple of tensors that if specified are added to the residuals of down unet blocks.
                mid_block_additional_residual: (`torch.Tensor`, *optional*):
                    A tensor that if specified is added to the residual of the middle unet block.
                down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                    additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
                encoder_attention_mask (`torch.Tensor`):
                    A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                    `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                    which adds large negative values to the attention scores corresponding to "discard" tokens.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                    tuple.

            Returns:
                [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                    If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                    otherwise a `tuple` is returned where the first element is the sample tensor.
        """

        # Check if the custromer embedding is provided
        #########################################################Ankit Yadav#####################################################
        USE_PEFT_BACKEND= False # Fixing this due to mismatch of Unet versions fix this patching later
        if custom_image_embeddings is not None and not isinstance(custom_image_embeddings, list):
            raise ValueError('custom_image_embeddings must be a list of tensors')
        #########################################################Ankit Yadav#####################################################
        #########################################################Ankit Yadav#####################################################
        if return_features:
            feature_list = []
        #########################################################Ankit Yadav#####################################################
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                        and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                        for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                #########################################################Ankit Yadav#####################################################
                if return_features:
                    feature_list.append(sample)
                #########################################################Ankit Yadav################################################
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
                #########################################################Ankit Yadav#####################################################
                if return_features:
                    feature_list.append(sample)
                #########################################################Ankit Yadav#####################################################
        #########################################################Ankit Yadav#####################################################
        if return_features:
            #print("Note: Returning features no need to use .sample method")
            return feature_list
        #########################################################Ankit Yadav#####################################################
        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)



# Code to calculate the SRCC and PLCC

class metric:
    def __init__(self):
        self.result = {}

    # def calcuate_srcc(self,gt,pt):
    #     srcc,_ = spearmanr(gt,pt)
    #     self.result["SRCC"] = float(srcc)

    #     return srcc
    
    # def calculate_plcc(self,gt,pt):
    #     plcc,_ = pearsonr(gt,pt)
    #     self.result["PLCC"] = float(plcc)

    #     return plcc
    
    def calcuate_srcc(self,tensor1, tensor2):
        tensor1_np = tensor1
        tensor2_np = tensor2

        rank1 = scipy.stats.rankdata(tensor1_np)
        rank2 = scipy.stats.rankdata(tensor2_np)

        srcc, _ = spearmanr(rank1, rank2)
        self.result["SRCC"] = float(srcc)
        return srcc


    def calculate_plcc(self,tensor1, tensor2):
        tensor1 = torch.from_numpy(tensor1)
        tensor2 = torch.from_numpy(tensor2)
        x_mean = tensor1.mean()
        y_mean = tensor2.mean()

        numerator = ((tensor1 - x_mean) * (tensor2 - y_mean)).sum()

        x_var = ((tensor1 - x_mean) ** 2).sum()
        y_var = ((tensor2 - y_mean) ** 2).sum()

        plcc = numerator / torch.sqrt(x_var * y_var)

        self.result["PLCC"] = float(plcc)
        return plcc

        
# KNN-SVM Regression



class HybridKNN_SVR:
    """
    Hybrid KNN‑SVR for regression – adaptation of
    Fuentes‑Pineda & Meza‑Ruiz, 2020 (arXiv:2007.00045).

    Parameters
    ----------
    k : int
        Number of neighbours for the global KNN component.
    radius : float
        Maximum distance θ allowed for a 'confident' KNN prediction.
    sigma : float
        Maximum standard deviation of the neighbour targets that
        is still considered 'consistent'.
    m : int
        Number of nearest samples used to train the local SVR
        (taken from the whole training set, regardless of their target value).
    svr_kernel, C, gamma, epsilon :
        Standard ε‑SVR hyper‑parameters (see sklearn.svm.SVR).
    """

    def __init__(
        self,
        *,
        k: int = 5,
        radius: float = 0.5,
        sigma: float = 0.05,
        m: int = 50,
        svr_kernel: str = "rbf",
        C: float = 10.0,
        gamma: str | float = "scale",
        epsilon: float = 0.1,
    ):
        self.k = k
        self.radius = radius
        self.sigma = sigma
        self.m = m
        self._knn = KNeighborsRegressor(n_neighbors=self.k, metric='euclidean')
        self._svr_params = dict(kernel=svr_kernel, C=C, gamma=gamma, epsilon=epsilon)
        # cache
        self._X = None
        self._y = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._X, self._y = np.asarray(X), np.asarray(y)
        self._knn.fit(self._X, self._y)
        return self

    # ------------------------------------------------------------------
    def _local_svr(self, x_q: np.ndarray) -> float:
        """Train a tiny SVR on the `m` closest points to `x_q`."""
        dist = np.linalg.norm(self._X - x_q, axis=1)
        idx = np.argsort(dist)[: self.m]
        svr = SVR(**self._svr_params)
        svr.fit(self._X[idx], self._y[idx])
        return svr.predict(x_q.reshape(1, -1))[0]

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        dists, idx = self._knn.kneighbors(X, return_distance=True)

        preds = []
        for i, (d, id_row) in enumerate(zip(dists, idx)):
            neigh_targets = self._y[id_row]
            # --- neighbourhood consistency test ----
            close_enough = d[-1] <= self.radius          # max dist ≤ θ
            consistent = neigh_targets.std() <= self.sigma
            if close_enough and consistent:
                # confident KNN: weighted mean already computed by KNN
                preds.append(self._knn.predict(X[i].reshape(1, -1))[0])
            else:
                preds.append(self._local_svr(X[i]))
        return np.asarray(preds)

# Visualisation of the features using Grad-CAM

class SIGLIPWithMLP(nn.Module):
    def __init__(self, base_model, mlp_head,device,layer=18,resnet=False):
        super(SIGLIPWithMLP, self).__init__()
        self.siglip    = base_model
        self.mlp_head  = mlp_head
        self.device    = device
        self.layer     = layer
        self.resnet    = resnet

    def forward(self, inputs):
        """
        images: a FloatTensor of shape (B,3,H,W), already on self.device
        returns: tensor of shape (B,) with your predicted scores
        """

        # 1) Extract features  # model.module(**inputs).last_hidden_state[:,0,:]
        if self.resnet:
            warnings.warn("Resent152 being used falling to get the image features for Resnet152 if not used please check code !!")
            features = self.siglip(**inputs).pooler_output
            features = features.squeeze(-1).squeeze(-1)
        else:
            try:
                features = self.siglip.get_image_features(inputs)
            except Exception as e:
                warnings.warn("DINO being used falling to get the image features for DINO if not used please check code !!")
                
                try:
                    #features = self.siglip(inputs).last_hidden_state[:,0,:]
                    # Average pooling
                    #features = self.siglip(inputs).pooler_output
                    features = self.siglip(inputs).last_hidden_state.mean(dim=1)
                except Exception as e:

                    
                    warnings.warn("Perception being used falling to get the image features for perception if not used please check code !!")
                    features = self.siglip.encode_image_layers(inputs,layer_idx=self.layer)
                
        # features has shape (B, hidden)

        # 2) Score head
        scores = self.mlp_head(features)      # (B,1)
        return scores.squeeze(1)   
    
           # (B,)

class SIGLIPWithMLP_embed(nn.Module):
    def __init__(self, base_model, mlp_head,device,layer=18):
        super(SIGLIPWithMLP_embed, self).__init__()
        self.siglip    = base_model
        self.mlp_head  = mlp_head
        self.device    = device
        self.layer     = layer

    def forward(self, inputs):
        """
        images: a FloatTensor of shape (B,3,H,W), already on self.device
        returns: tensor of shape (B,) with your predicted scores
        """

        # 1) Extract features  # model.module(**inputs).last_hidden_state[:,0,:]
        try:
            features = self.siglip.get_image_features(inputs)
        except Exception as e:
            warnings.warn("DINO being used falling to get the image features for DINO if not used please check code !!")
            
            try:
                #features = self.siglip(inputs).last_hidden_state[:,0,:]
                # Average pooling
                #features = self.siglip(inputs).pooler_output
                features = self.siglip(inputs).last_hidden_state.mean(dim=1)
            except Exception as e:
                warnings.warn("Perception being used falling to get the image features for perception if not used please check code !!")
                #features = self.siglip.encode_image(inputs)
                features = self.siglip.encode_image_layers(inputs,layer_idx=self.layer)
        # features has shape (B, hidden)

        # 2) Score head
        scores,embed = self.mlp_head(features)      # (B,1)
        return scores.squeeze(1)   
    
           # (B,)

class SIGLIPWithMLP_cross_attention_patch_wise_no_embed(nn.Module):
    def __init__(self, base_model, mlp_head,device,processor,BAD_QUALITY_PROMPT,layer=18):
        super(SIGLIPWithMLP_cross_attention_patch_wise_no_embed, self).__init__()
        self.siglip    = base_model
        self.mlp_head  = mlp_head
        self.device    = device
        self.layer     = layer
        self.processor = processor
        self.BAD_QUALITY_PROMPT = BAD_QUALITY_PROMPT

    def forward(self, images):
        """
        images: a FloatTensor of shape (B,3,H,W), already on self.device
        returns: tensor of shape (B,) with your predicted scores
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs = inputs['pixel_values']
        # 1) Extract features  # model.module(**inputs).last_hidden_state[:,0,:]
        try:
            features = self.siglip.vision_model(pixel_values=inputs,return_dict=True)

            image_patch_tokens = features.last_hidden_state 
            image_cls_token = features.pooler_output 



            # candidate_prompts = [f"A photograph of {scene}"  for scene in scenes.split(",")]
            # best_prompt = None
            # match_inputs = self.processor(text=candidate_prompts, images=images, return_tensors="pt",padding='max_length', truncation=True, max_length=64).to(self.device )
            # with torch.no_grad():
            #     outputs = self.siglip(**match_inputs)
            #     logits_per_image = outputs.logits_per_image  # Image-text similarity scores
            #     probs = torch.sigmoid(logits_per_image)  # Convert logits to probabilities

            #     best_match_index = [probs[i].argmax().item() for i in range(len(images))] # Get the index of the best match for each image
            #     best_prompt = [candidate_prompts[best_match_index[i]] for i in range(len(images))] # Get the best prompt for each image
            # quality_prompts = [self.BAD_QUALITY_PROMPT.format(prompt) for prompt in best_prompt]
                    
            # fixed_prompt  = [[prompt for prompt in  self.BAD_QUALITY_PROMPT.split(",")]]*len(images)
            # text_inputs = [self.processor(text=p, return_tensors="pt",padding=True).to(self.device) for p in fixed_prompt]

            # text_features = [self.siglip.get_text_features(**t_input) for t_input in text_inputs] #module
            # text_features = torch.stack(text_features, dim=0) #module
            
            fixed_prompt = [self.BAD_QUALITY_PROMPT] * len(images) 
            text_inputs = self.processor(text=fixed_prompt, return_tensors="pt").to(self.device)
            #text_features = self.siglip.get_text_features(**text_inputs) #module
            text_features = self.siglip.text_model(**text_inputs,return_dict=True) #module
            text_patch_tokens = text_features.last_hidden_state #module
            text_cls_token = text_features.pooler_output #module

            del text_inputs

        except Exception as e:
            warnings.warn("DINO being used falling to get the image features for DINO if not used please check code !!")
            
            try:
                #features = self.siglip(inputs).last_hidden_state[:,0,:]
                # Average pooling
                #features = self.siglip(inputs).pooler_output
                features = self.siglip(inputs).last_hidden_state.mean(dim=1)
            except Exception as e:
                warnings.warn("Perception being used falling to get the image features for perception if not used please check code !!")
                features = self.siglip.encode_image_layers(inputs,layer_idx=self.layer)
        # features has shape (B, hidden)

        # 2) Score head
        scores = self.mlp_head(image_patch_tokens,image_cls_token,text_patch_tokens,text_cls_token)      # (B,1)
        return scores.squeeze(1)              # (B,)
                    # features = model.module.vision_model(**inputs,return_dict=True) #module
                    # image_patch_tokens = features.last_hidden_state #module
                    # image_cls_token = features.pooler_output #module
                    # text_features = model.module.get_text_features(**text_inputs) #module

                   
                    # # extractor.features.clear()
                    # score = mlp(image_patch_tokens,image_cls_token,text_features)

class SIGLIPWithMLP_cross_attention_patch_wise(nn.Module):
    def __init__(self, base_model, mlp_head,device,processor,BAD_QUALITY_PROMPT,layer=18):
        super(SIGLIPWithMLP_cross_attention_patch_wise, self).__init__()
        self.siglip    = base_model
        self.mlp_head  = mlp_head
        self.device    = device
        self.layer     = layer
        self.processor = processor
        self.BAD_QUALITY_PROMPT = BAD_QUALITY_PROMPT

    def forward(self, images):
        """
        images: a FloatTensor of shape (B,3,H,W), already on self.device
        returns: tensor of shape (B,) with your predicted scores
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs = inputs['pixel_values']
        # 1) Extract features  # model.module(**inputs).last_hidden_state[:,0,:]
        try:
            features = self.siglip.vision_model(pixel_values=inputs,return_dict=True)

            image_patch_tokens = features.last_hidden_state 
            image_cls_token = features.pooler_output 



            # candidate_prompts = [f"A photograph of {scene}"  for scene in scenes.split(",")]
            # best_prompt = None
            # match_inputs = self.processor(text=candidate_prompts, images=images, return_tensors="pt",padding='max_length', truncation=True, max_length=64).to(self.device )
            # with torch.no_grad():
            #     outputs = self.siglip(**match_inputs)
            #     logits_per_image = outputs.logits_per_image  # Image-text similarity scores
            #     probs = torch.sigmoid(logits_per_image)  # Convert logits to probabilities

            #     best_match_index = [probs[i].argmax().item() for i in range(len(images))] # Get the index of the best match for each image
            #     best_prompt = [candidate_prompts[best_match_index[i]] for i in range(len(images))] # Get the best prompt for each image
            # quality_prompts = [self.BAD_QUALITY_PROMPT.format(prompt) for prompt in best_prompt]
                    
            # fixed_prompt  = [[prompt for prompt in  self.BAD_QUALITY_PROMPT.split(",")]]*len(images)
            # text_inputs = [self.processor(text=p, return_tensors="pt",padding=True).to(self.device) for p in fixed_prompt]

            # text_features = [self.siglip.get_text_features(**t_input) for t_input in text_inputs] #module
            # text_features = torch.stack(text_features, dim=0) #module
            
            fixed_prompt = [self.BAD_QUALITY_PROMPT] * len(images) 
            text_inputs = self.processor(text=fixed_prompt, return_tensors="pt").to(self.device)
            #text_features = self.siglip.get_text_features(**text_inputs) #module
            text_features = self.siglip.text_model(**text_inputs,return_dict=True) #module
            text_patch_tokens = text_features.last_hidden_state #module
            text_cls_token = text_features.pooler_output #module

            del text_inputs

        except Exception as e:
            warnings.warn("DINO being used falling to get the image features for DINO if not used please check code !!")
            
            try:
                #features = self.siglip(inputs).last_hidden_state[:,0,:]
                # Average pooling
                #features = self.siglip(inputs).pooler_output
                features = self.siglip(inputs).last_hidden_state.mean(dim=1)
            except Exception as e:
                warnings.warn("Perception being used falling to get the image features for perception if not used please check code !!")
                features = self.siglip.encode_image_layers(inputs,layer_idx=self.layer)
        # features has shape (B, hidden)

        # 2) Score head
        scores,embed = self.mlp_head(image_patch_tokens,image_cls_token,text_patch_tokens,text_cls_token)      # (B,1)
        return scores.squeeze(1)              # (B,)
                    # features = model.module.vision_model(**inputs,return_dict=True) #module
                    # image_patch_tokens = features.last_hidden_state #module
                    # image_cls_token = features.pooler_output #module
                    # text_features = model.module.get_text_features(**text_inputs) #module

                   
                    # # extractor.features.clear()
                    # score = mlp(image_patch_tokens,image_cls_token,text_features)

class SIGLIPWithMLP_cross_attention_patch_wise_grouped_token(nn.Module):
    def __init__(self, base_model, mlp_head,device,processor,BAD_QUALITY_PROMPT,layer=18):
        super(SIGLIPWithMLP_cross_attention_patch_wise_grouped_token, self).__init__()
        """
        Note: Here this is not designed to handel embedding out if you want to do it modify the mlp head output in this class to fix it currently treating it 
        as normal case where you just get the scores and not the embeddings.
        """
        self.siglip    = base_model
        self.mlp_head  = mlp_head
        self.device    = device
        self.layer     = layer
        self.processor = processor
        self.BAD_QUALITY_PROMPT = BAD_QUALITY_PROMPT

    def forward(self, images):
        """
        images: a FloatTensor of shape (B,3,H,W), already on self.device
        returns: tensor of shape (B,) with your predicted scores
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs = inputs['pixel_values']
        # 1) Extract features  # model.module(**inputs).last_hidden_state[:,0,:]
        try:
            features = self.siglip.vision_model(pixel_values=inputs,return_dict=True)

            image_patch_tokens = features.last_hidden_state 
            image_cls_token = features.pooler_output 



            # candidate_prompts = [f"A photograph of {scene}"  for scene in scenes.split(",")]
            # best_prompt = None
            # match_inputs = self.processor(text=candidate_prompts, images=images, return_tensors="pt",padding='max_length', truncation=True, max_length=64).to(self.device )
            # with torch.no_grad():
            #     outputs = self.siglip(**match_inputs)
            #     logits_per_image = outputs.logits_per_image  # Image-text similarity scores
            #     probs = torch.sigmoid(logits_per_image)  # Convert logits to probabilities

            #     best_match_index = [probs[i].argmax().item() for i in range(len(images))] # Get the index of the best match for each image
            #     best_prompt = [candidate_prompts[best_match_index[i]] for i in range(len(images))] # Get the best prompt for each image
            # quality_prompts = [self.BAD_QUALITY_PROMPT.format(prompt) for prompt in best_prompt]
                    
            # fixed_prompt  = [[prompt for prompt in  self.BAD_QUALITY_PROMPT.split(",")]]*len(images)
            # text_inputs = [self.processor(text=p, return_tensors="pt",padding=True).to(self.device) for p in fixed_prompt]

            # text_features = [self.siglip.get_text_features(**t_input) for t_input in text_inputs] #module
            # text_features = torch.stack(text_features, dim=0) #module
            
            # fixed_prompt = [self.BAD_QUALITY_PROMPT] * len(images) 
            # text_inputs = self.processor(text=fixed_prompt, return_tensors="pt").to(self.device)
            # #text_features = self.siglip.get_text_features(**text_inputs) #module
            # text_features = self.siglip.text_model(**text_inputs,return_dict=True) #module
            # text_patch_tokens = text_features.last_hidden_state #module
            # text_cls_token = text_features.pooler_output #module

            text_patch_tokens_list = []
            text_cls_token_list = []

            # Grouping the BAD_QUALITY_PROMPT for individual quality part. 
            for sub_prompt in BAD_QUALITY_PROMPT.split(","):
                sub_prompt = [sub_prompt]* len(images)
                text_inputs = self.processor(text=sub_prompt, return_tensors="pt").to(self.device)
                text_features = self.siglip.text_model(**text_inputs,return_dict=True) 
                text_patch_tokens = text_features.last_hidden_state 
                text_cls_token = text_features.pooler_output 
                text_patch_tokens_list.append(text_patch_tokens.mean(dim=1)) # Average pooling over the patch tokens
                text_cls_token_list.append(text_cls_token)
                del text_inputs,text_features
                # clear the Cuda cache
                torch.cuda.empty_cache()


            text_patch_tokens = torch.stack(text_patch_tokens_list, dim=1)
            text_cls_token = torch.stack(text_cls_token_list, dim=1).mean(dim=1) 

            

        except Exception as e:
            warnings.warn("DINO being used falling to get the image features for DINO if not used please check code !!")
            
            try:
                #features = self.siglip(inputs).last_hidden_state[:,0,:]
                # Average pooling
                #features = self.siglip(inputs).pooler_output
                features = self.siglip(inputs).last_hidden_state.mean(dim=1)
            except Exception as e:
                warnings.warn("Perception being used falling to get the image features for perception if not used please check code !!")
                features = self.siglip.encode_image_layers(inputs,layer_idx=self.layer)
        # features has shape (B, hidden)

        # 2) Score head

        scores = self.mlp_head(image_patch_tokens,image_cls_token,text_patch_tokens,text_cls_token)      # (B,1)
        return scores.squeeze(1)              # (B,)
                    # features = model.module.vision_model(**inputs,return_dict=True) #module
                    # image_patch_tokens = features.last_hidden_state #module
                    # image_cls_token = features.pooler_output #module
                    # text_features = model.module.get_text_features(**text_inputs) #module

                   
                    # # extractor.features.clear()
                    # score = mlp(image_patch_tokens,image_cls_token,text_features)                    

class UNETWithMLP(nn.Module):
    def __init__(self, base_model, mlp_head,qfd_adapter,device):
        super(UNETWithMLP, self).__init__()
        self.siglip    = base_model
        self.mlp_head  = mlp_head
        self.qfd_adapter = qfd_adapter
        self.device    = device

    def forward(self, inputs):
        """
        images: a FloatTensor of shape (B,3,H,W), already on self.device
        returns: tensor of shape (B,) with your predicted scores
        """

        # 1) Extract features  # model.module(**inputs).last_hidden_state[:,0,:]
        try:
            features = self.siglip.get_features(inputs, None, t=None, feat_key=None)

            quality_feature = features
            quality_feature = process_cleandift_features(quality_feature)  
            quality_feature = self.qfd_adapter(quality_feature)
            
            #features = features.mean(dim=1).flatten(start_dim=1)  
        except Exception as e:
            raise e

        # 2) Score head
        scores = self.mlp_head(quality_feature)      # (B,1)
        return scores.squeeze(1)              # (B,)
    

        # inputs = self.processor(images=images, return_tensors="pt").to(model.device)
        #             siglip_features = siglip_model.get_image_features(**inputs) #module

class UNET_SIGLIP_WithMLP(nn.Module):
    def __init__(self, base_model, mlp_head,qfd_adapter,siglip,processor,device):
        super(UNET_SIGLIP_WithMLP, self).__init__()
        self.unet    = base_model
        self.mlp_head  = mlp_head
        self.qfd_adapter = qfd_adapter
        self.siglip    = siglip
        self.processor = processor
        self.device    = device

    def forward(self, inputs):
        """
        images: a FloatTensor of shape (B,3,H,W), already on self.device
        returns: tensor of shape (B,) with your predicted scores
        """

        # 1) Extract features  # model.module(**inputs).last_hidden_state[:,0,:]
        try:
            siglip_inputs = self.processor(images=inputs, return_tensors="pt").to(self.device)
            siglip_inputs = siglip_inputs['pixel_values']

            features = self.unet.get_features(siglip_inputs, None, t=None, feat_key=None)

            quality_feature = features
            quality_feature = process_cleandift_features(quality_feature)  
            quality_feature = self.qfd_adapter(quality_feature)

            
            siglip_features = self.siglip.get_image_features(siglip_inputs)
            
            #features = features.mean(dim=1).flatten(start_dim=1)  
        except Exception as e:
            raise e

        # 2) Score head
        scores = self.mlp_head(quality_feature,siglip_features)      # (B,1)
        return scores.squeeze(1)              # (B,)
    

def Overlay(img, heatmap, alpha=(0.6,0.4)):
    """
    Overlay a heatmap on an image.
    Args:
        img (numpy.ndarray): Original image.
        heatmap (numpy.ndarray): Heatmap to overlay.
        alpha (float): Transparency factor for the overlay.
    Returns:
        numpy.ndarray: Image with heatmap overlay.
    """

    # Unormalize the image
    mean = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
    unnormed = img * std + mean
    unormed = torch.clamp(unnormed, 0.0, 1.0)
    orignal_image_rgb = unormed.cpu().permute(1,2,0).numpy()
    heat_map_uint_8 = (heatmap * 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heat_map_uint_8, cv2.COLORMAP_JET) 
    orignal_image_rgb_uint8 = (orignal_image_rgb * 255).astype(np.uint8)
    orignal_image_rgb_uint8 = cv2.cvtColor(orignal_image_rgb_uint8, cv2.COLOR_RGB2BGR)
    super_imposed = cv2.addWeighted(orignal_image_rgb_uint8, alpha[0], color_heatmap, alpha[1], 0)
    return super_imposed,orignal_image_rgb_uint8
# class Cam:
#     def __init__(self, model,mlp_head, target_layers):
#         self.model = model
#         self.target_layers = target_layers
#         self.gradients = []
#         self.activations = []

#         for layer in target_layers:
#             layer.register_forward_hook(self.save_activation)
#             layer.register_backward_hook(self.save_gradient)

#     def save_gradient(self, module, grad_in, grad_out):
#         self.gradients.append(grad_out[0])


def apply_blur(image: Image.Image, blur_ratio=0.1) -> Image.Image:

    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Check for the dimension of the image

    if image_np.ndim == 2:  # grayscale
        img_cv = image_np
    else:
        img_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


     # Calculate kernel size as 10% of width and height
    h, w = img_cv.shape[:2]
    ksize = int(min(h, w) * blur_ratio)
    ksize = ksize + 1 if ksize % 2 == 0 else ksize  # Ensure kernel is odd

    blurred = cv2.GaussianBlur(img_cv, (ksize, ksize), 0)


    # Convert back to PIL Image

    if blurred.ndim == 2:
        return Image.fromarray(blurred)
    else:
        blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blurred_rgb)
    


def apply_gaussian_noise_tensor(x: torch.Tensor, noise_level: float = 0.01, mean: float = 0.0) -> torch.Tensor:
    """
    Additive Gaussian noise on a tensor.
    noise_level: variance level in [0,1], e.g. 0.01 for 1% noise strength.
    mean: mean of the Gaussian noise.
    """
    var = noise_level
    std = var ** 0.5
    noise = torch.randn_like(x) * std + mean
    return (x + noise).clamp(0.0, 1.0)


def apply_salt_pepper_noise_tensor(x: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """
    Salt-and-pepper noise on a tensor.
    noise_level: fraction of pixels to corrupt, e.g. 0.01 for 1%.
    """
    x_noisy = x.clone()
    # supports both batched (N,C,H,W) and single (C,H,W)
    if x.ndim == 4:
        N, C, H, W = x.shape
        num_pixels = int(noise_level * H * W)
        for n in range(N):
            for _ in range(num_pixels):
                i = torch.randint(0, H, (1,)).item()
                j = torch.randint(0, W, (1,)).item()
                x_noisy[n, :, i, j] = torch.bernoulli(torch.tensor(0.5))
    else:
        C, H, W = x.shape
        num_pixels = int(noise_level * H * W)
        for _ in range(num_pixels):
            i = torch.randint(0, H, (1,)).item()
            j = torch.randint(0, W, (1,)).item()
            x_noisy[:, i, j] = torch.bernoulli(torch.tensor(0.5))
    return x_noisy


def apply_speckle_noise_tensor(x: torch.Tensor, noise_level: float = 0.01, mean: float = 0.0) -> torch.Tensor:
    """
    Multiplicative speckle noise on a tensor.
    noise_level: variance level in [0,1], e.g. 0.01 for 1%.
    mean: mean of the speckle distribution.
    """
    var = noise_level
    std = var ** 0.5
    noise = torch.randn_like(x) * std + mean
    return (x + x * noise).clamp(0.0, 1.0)


def apply_poisson_noise_tensor(x: torch.Tensor, noise_level: float = 1.0) -> torch.Tensor:
    """
    Poisson (shot) noise on a tensor.
    noise_level: scales the photon count; lower values simulate stronger noise.
    For noise_level in (0,1], values closer to 0 produce more "shot" noise.
    """
    # Compute unique count then nearest power-of-two
    unique_vals = torch.unique(x).numel()
    power = int(torch.ceil(torch.log2(torch.tensor(unique_vals, dtype=torch.float32))))
    base = 2 ** power
    vals = noise_level * base
    # Scale, sample, and rescale
    x_scaled = x * vals
    noisy = torch.poisson(x_scaled)
    return (noisy / vals).clamp(0.0, 1.0)


def apply_uniform_noise_tensor(x: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """
    Uniform noise on a tensor.
    noise_level: maximum amplitude of uniform noise, e.g. 0.01 for ±1%.
    """
    low, high = -noise_level, noise_level
    noise = (high - low) * torch.rand_like(x) + low
    return (x + noise).clamp(0.0, 1.0)


def apply_gaussian_blur_tensor(x: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """
    Gaussian blur on a tensor via depthwise convolution.
    noise_level: controls kernel size relative to image min dim.
    Automatically scales input to [0,1] if values exceed 1.0, and returns float32 [0,1].
    """
    # Convert to float and normalize to [0,1]
    x_f = x.clone().float()
    if x_f.max() > 1.0:
        x_f = x_f / 255.0

    # Determine dynamic kernel size
    _, _, H, W = x_f.shape if x_f.ndim == 4 else (1, *x_f.shape)
    k = max(3, int(noise_level * min(H, W) * 2 + 1))
    kernel_size = k if k % 2 == 1 else k + 1
    sigma = noise_level * min(H, W) / 8.0 + 0.5

    # Build Gaussian kernel
    coords = torch.arange(kernel_size, device=x_f.device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    grid = coords.unsqueeze(0).repeat(kernel_size, 1)
    gauss = torch.exp(-(grid**2 + grid.t()**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    gauss_kernel = gauss.view(1, 1, kernel_size, kernel_size)

    # Prepare batch
    batched = True
    if x_f.ndim == 3:
        x_f = x_f.unsqueeze(0)
        batched = False
    N, C, H, W = x_f.shape
    weight = gauss_kernel.repeat(C, 1, 1, 1).to(x_f.device)

    # Depthwise convolution
    blurred = F.conv2d(x_f, weight, padding=kernel_size//2, groups=C)
    
    # Return float32 in [0,1]
    return blurred.squeeze(0) if not batched else blurred

def apply_random_noise(x: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """
    Apply a random noise/blur function from the noise_functions list.
    noise_level: passed to the chosen function.
    """

    noise_functions = [
    apply_gaussian_noise_tensor,
    apply_salt_pepper_noise_tensor,
    apply_speckle_noise_tensor,
    apply_poisson_noise_tensor,
    apply_uniform_noise_tensor,
    #apply_gaussian_blur_tensor Skiiping for now
    ]

    fn = random.choice(noise_functions)
    return fn(x, noise_level)

def apply_random_noise_only_blur(x: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """
    Apply just Blur Function.
    noise_level: passed to the chosen function.
    """

    noise_functions = [
    apply_gaussian_blur_tensor
    ]

    fn = random.choice(noise_functions)
    return fn(x, noise_level)
### Defining functions for scaling the loss into final loss


class ScaleLossSoftmax(nn.Module):
    def __init__(self,num_losses=4):
        super(ScaleLossSoftmax, self).__init__()
        self.raw_weights = nn.Parameter(torch.randn(num_losses))
        self.softmax = nn.Softmax(dim=0)
        self.return_params_weights = False
    def forward(self, loss_list):
        """
        Scale the loss using softmax to ensure all losses are positive and sum to 1.
        Args:
            loss_list (list): List of losses to be scaled.
        Returns:
            list: Scaled losses.
        """
        weights = self.softmax(self.raw_weights)
        loss_tensor = torch.stack(loss_list)
        total_loss = torch.sum(weights * loss_tensor)
        
        if self.return_params_weights:
            return total_loss, weights
        else:
            return total_loss
        

# Trying out different variants of MLP layers
class mlp_3_layer_leakyrelu_siglip2(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden: int = 512):
        super(mlp_3_layer_leakyrelu_siglip2, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.adapter(x)

class mlp_3_layer_tanh_siglip2(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden: int = 512):
        super(mlp_3_layer_tanh_siglip2, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.adapter(x)

class mlp_3_layer_gelu_siglip2(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden: int = 512):
        super(mlp_3_layer_gelu_siglip2, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.adapter(x)

class mlp_3_layer_sigmoid_siglip2(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden: int = 512):
        super(mlp_3_layer_sigmoid_siglip2, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.adapter(x)
    
class ParamLeakyReLU2(nn.Module):
    """
    Learnable Leaky-ReLU/PReLU with either a single scalar or per-channel slopes.
    If per_channel=True, 'dim' must be the hidden size of the preceding Linear.
    """
    def __init__(self, dim: int | None = None, init_a: float = 0.25, per_channel: bool = True):
        super().__init__()
        if per_channel:
            assert dim is not None, "dim (hidden size) required for per-channel slopes"
            self.a = nn.Parameter(torch.full((dim,), init_a, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))

    def forward(self, x):
        # works for [B, dim] and broadcasts 'a' if scalar
        return torch.where(x >= 0, x, self.a * x)


class ParamSigmoid2(nn.Module):
    """
    σ(α x + β) with learnable α (slope) and β (bias).
    Per-channel parameters are recommended (dim = hidden size).
    """
    def __init__(self, dim: int | None = None, init_alpha: float = 1.0, init_beta: float = 0.0, per_channel: bool = True, clamp: float = 20.0):
        super().__init__()
        if per_channel:
            assert dim is not None, "dim (hidden size) required for per-channel parameters"
            self.alpha = nn.Parameter(torch.full((dim,), init_alpha, dtype=torch.float32))
            self.beta  = nn.Parameter(torch.full((dim,), init_beta,  dtype=torch.float32))
        else:
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
            self.beta  = nn.Parameter(torch.tensor(init_beta,  dtype=torch.float32))
        self.clamp = clamp  # prevent extreme saturation if desired

    def forward(self, x):
        z = self.alpha * x + self.beta
        if self.clamp is not None:
            z = z.clamp(-self.clamp, self.clamp)
        return torch.sigmoid(z)


# Gating for activations
# ───────────────────────────── gated blend module ───────────────────────────── #

class GatedBlend(nn.Module):
    """
    y = w * ParamSigmoid2(x) + (1 - w) * ParamLeakyReLU2(x)
    where w = sigmoid(g).  Initial g = 0 → w = 0.5  (balanced start).
    """
    def __init__(self, dim: int, per_channel: bool = True,
                 init_alpha: float = 1.0, init_beta: float = 0.0,
                 init_a: float = 0.25):
        super().__init__()
        self.sig_act = ParamSigmoid2(dim, init_alpha, init_beta, per_channel)
        self.lrelu_act = ParamLeakyReLU2(dim, init_a, per_channel)

        # gate g; initialise to 0 so sigmoid(g)=0.5
        if per_channel:
            self.g = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.g = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        w = torch.sigmoid(self.g)                # [dim] or scalar
        return w * self.sig_act(x) + (1.0 - w) * self.lrelu_act(x)

###########################################################################################################################
# ─────────────────────────────── new MLP variants ───────────────────────────── #

class MLP3_Gated(nn.Module):
    """
    Linear -> GatedBlend(ParamSig+ParamLReLU) -> Linear -> ParamLeakyReLU -> Linear -> output
    """
    def __init__(self, input_dim=1024, hidden=512, per_channel=True):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden)
        self.act1 = GatedBlend(hidden, per_channel)          # <-- blended first layer
        self.fc2  = nn.Linear(hidden, hidden)
        self.act2 = ParamLeakyReLU2(hidden, init_a=0.25, per_channel=per_channel)
        self.fc3  = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class MLP3_Gated_embed(nn.Module):
    """
    Same as above but also returns the final hidden representation.
    """
    def __init__(self, input_dim=1024, hidden=512, per_channel=True):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden)
        self.act1 = GatedBlend(hidden, per_channel)
        self.fc2  = nn.Linear(hidden, hidden)
        self.act2 = ParamLeakyReLU2(hidden, init_a=0.25, per_channel=per_channel)
        self.fc3  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        pred = self.fc3(h)
        return pred, h

########################################################################################################################

class MLP3_ParamActs(nn.Module):
    """
    Linear -> ParamSigmoid -> Linear -> ParamLeakyReLU -> Linear -> output
    Suggested inits:
      - ParamSigmoid: alpha=1.0, beta=0.0  (analogous to Swish β≈1 start)
      - ParamLeakyReLU: a=0.25 (PReLU paper init)
    """
    def __init__(self, input_dim=1024, hidden=512, per_channel=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.act1 = ParamSigmoid2(dim=hidden if per_channel else None,
                                 init_alpha=1.0, init_beta=0.0, per_channel=per_channel)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = ParamLeakyReLU2(dim=hidden if per_channel else None,
                                   init_a=0.25, per_channel=per_channel)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return self.fc3(x)
    
class MLP3_ParamActs_embed(nn.Module):
    """
    Linear -> ParamSigmoid -> Linear -> ParamLeakyReLU -> Linear -> output
    Suggested inits:
      - ParamSigmoid: alpha=1.0, beta=0.0  (analogous to Swish β≈1 start)
      - ParamLeakyReLU: a=0.25 (PReLU paper init)
    """
    def __init__(self, input_dim=1024, hidden=512, per_channel=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.act1 = ParamSigmoid2(dim=hidden if per_channel else None,
                                 init_alpha=1.0, init_beta=0.0, per_channel=per_channel)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = ParamLeakyReLU2(dim=hidden if per_channel else None,
                                   init_a=0.25, per_channel=per_channel)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        pred = self.fc3(h)
        return pred,h
#################################################################################################################################
# Mean Embedding Regularization
class MeanEmbedRegularizationV2:
    """
    Mean‑Embedding Regularisation with quantile buckets, adaptive λ, EMA centres,
    temperature scheduling, and extra collapse‑guards.

    Args
    ----
    embed_dim       : dimension D of the embeddings
    train_size      : #images in the training split  (needed for auto buckets)
    device          : 'cuda' or 'cpu'
    small_set_thr   : threshold separating 'small' vs 'large' sets (default 3_000)

    Call sequence
    -------------
        >>> mer = MeanEmbedRegularizationV2(...)
        >>> for epoch in range(E):
        ...     lam, tau = mer.sched(epoch, E)   # 1) λ, τ for this epoch
        ...     mer.update_buckets(h.detach(), mos)    # 2) running means
        ...     ce = mer.bucket_ce_loss(h, mos, tau)   # 3) CE part
        ...     loss = mse + lam * ce + mer.reg_terms()# 4) full loss
    """
    # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ initialisation ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ #
    def __init__(self,
                 embed_dim      : int,
                 train_size     : int,
                 device         : str = "cuda",
                 small_set_thr  : int = 3_000,
                 num_buckets    : str | int = "auto",
                 centre_m_small : float = 0.90,
                 centre_m_large : float = 0.95,
                 beta_entropy   : float = 1e-2):
        self.D         = embed_dim
        self.device    = device
        self.beta      = beta_entropy

        # 1) choose #buckets
        if num_buckets == "auto":
            self.C = 10 if train_size <= small_set_thr else 20
        else:
            self.C = num_buckets

        # 2) running quantile edges, updated each epoch
        self.edges   = torch.linspace(0, 1, self.C + 1, device=device)
        # 3) bucket centres & counts
        self.means   = torch.zeros(self.C, self.D, device=device)  # unit‑norm later
        self.counts  = torch.zeros(self.C,  dtype=torch.long, device=device)
        self.mom     = centre_m_small if train_size <= small_set_thr else centre_m_large

    # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ schedulers: λ and τ ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ #
    def sched(self, epoch: int, total_epochs: int,
              lam_max_small=(.25, .05), tau_range_small=(.07, .10),
              tau_range_large=(.10, .20)):
        """Return λ(t) and τ(t) for this epoch."""
        pct   = epoch / max(1, total_epochs)                # 0‑‑>1
        warm  = min(pct / 0.10, 1.)                         # warm‑up 10 %
        decay = 0.5 * (1 + math.cos(math.pi * pct))         # cosine

        # λ schedule
        lam_max = lam_max_small[0] if self.C <= 10 else lam_max_small[1]
        lam_t   = warm * lam_max * decay                    # warm‑up then decay

        # τ schedule
        tau_lo, tau_hi = tau_range_small if self.C <= 10 else tau_range_large
        tau_t  = tau_lo + (tau_hi - tau_lo) * pct           # linear ↑

        return lam_t, tau_t

    # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ bucket maintenance ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ #
    @torch.no_grad()
    def update_buckets(self, h: torch.Tensor, mos: torch.Tensor):
        """EMA update of bucket means using *quantile* edges per epoch."""
        # 1) recompute edges as quantiles of the *current* batch MOS
        self.edges = torch.quantile(
            mos.float(), torch.linspace(0, 1, self.C + 1, device=mos.device)
        )
        # 2) unit‑norm embeddings
        mos = mos.view(-1)
        h_norm = F.normalize(h, dim=-1, eps=1e-8)
        # 3) bucket assignment (searchsorted faster than loop)
        idx = torch.bucketize(mos, self.edges[1:-1])        # [B]
        for c in range(self.C):
            mask = (idx == c)
            if not torch.any(mask): continue
            new_mu = h_norm[mask].mean(dim=0)
            if self.counts[c] == 0:
                self.means[c] = new_mu
            else:
                self.means[c] = F.normalize(
                    self.mom * self.means[c] + (1-self.mom) * new_mu,
                    dim=-1, eps=1e-8)
            self.counts[c] += mask.sum()

    # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ CE + regularisers ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ #
    def bucket_ce_loss(self,
                       h          : torch.Tensor,
                       mos        : torch.Tensor,
                       tau        : float):
        """Cosine logits + CE (entropy‑augmented)."""
        if (self.counts == 0).any():          # cold start guard
            return torch.zeros([], device=h.device)

        # 1) unit‑norm
        h = F.normalize(h, dim=-1, eps=1e-8)
        C_norm = F.normalize(self.means, dim=-1, eps=1e-8)

        # 2) logits & targets
        idx = torch.bucketize(mos, self.edges[1:-1])
        logits = torch.matmul(h, C_norm.T) / tau            # cosine / τ

        # 3) CE
        ce = F.cross_entropy(logits, idx)

        # 4) entropy bonus
        probs = F.softmax(logits, dim=-1)
        ent   = -(probs * probs.log()).sum(dim=-1).mean()
        return ce - self.beta * ent

    # orthogonality reg (optional, call once per step if you want)
    def reg_terms(self, alpha_ortho=1e-3):
        if (self.counts == 0).any(): return 0.
        M = F.normalize(self.means, dim=-1, eps=1e-8)
        I = torch.eye(self.C, device=self.device)
        ortho = (M @ M.T - I).abs().sum() / self.C**2
        return alpha_ortho * ortho

########################################################
class ScaledSigmoid(nn.Module):
    """
    f_alpha(z) = alpha * sigmoid(z / alpha)
    - alpha: default 2.0 per Ven & Lederer (2021)
    - learnable: set True if you want alpha to be trained; it’s kept positive via softplus.
    """
    def __init__(self, alpha: float = 2.0, learnable: bool = False):
        super().__init__()
        if learnable:
            # param kept positive: alpha = softplus(raw) + eps
            self.raw = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha)) - 1.0))
            self.eps = 1e-6
        else:
            self.register_buffer("alpha_const", torch.tensor(float(alpha)))
            self.raw = None
            self.eps = 0.0

    @property
    def alpha(self):
        return (F.softplus(self.raw) + self.eps) if self.raw is not None else self.alpha_const

    def forward(self, x):
        a = self.alpha
        return a * torch.sigmoid(x / a)


class mlp_3_layer_sigmoid_scaled_2(nn.Module):
    def __init__(self, input_dim=32768, hidden=512, alpha=2.0, learnable_alpha=False):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            ScaledSigmoid(alpha=alpha, learnable=learnable_alpha),  # <— scaled sigmoid here
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.adapter(x)

class mlp_3_layer_sigmoid_scaled_2_embed(nn.Module):
    def __init__(self, input_dim=32768, hidden=512, alpha=2.0, learnable_alpha=False):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            ScaledSigmoid(alpha=alpha, learnable=learnable_alpha),  # <— scaled sigmoid here
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """Same as :class:`mlp_3_layer_sigmoid_scaled_2` but for embed variant."""
        h = self.adapter[0](x)
        h = self.adapter[1](h)
        h = self.adapter[2](h)
        h = self.adapter[3](h)
        embed = h
        score = self.adapter[4](embed)
        return score, embed
########################################################

################################################################################################################
# The version of the Gated Network with Lernable Gaama for the Sigmoid scaling 
# and fixed lr for the parameters
# And both the layers become Gateblend

class ParamLeakyReLU2_2(nn.Module):
    """
    Learnable Leaky-ReLU/PReLU with either a single scalar or per-channel slopes.
    If per_channel=True, 'dim' must be the hidden size of the preceding Linear.
    """
    def __init__(self, dim: int | None = None, init_a: float = 0.25, per_channel: bool = True):
        super().__init__()
        if per_channel:
            assert dim is not None, "dim (hidden size) required for per-channel slopes"
            self.a = nn.Parameter(torch.full((dim,), init_a, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))

    def forward(self, x):
        # works for [B, dim] and broadcasts 'a' if scalar
        return torch.where(x >= 0, x, self.a * x)


class ParamSigmoid2_2(nn.Module):
    """
    σ(α x + β) with learnable α (slope) and β (bias).
    Per-channel parameters are recommended (dim = hidden size).
    """
    def __init__(self, dim: int | None = None, init_alpha: float = 1.0, init_beta: float = 0.0, per_channel: bool = True, 
                clamp: float = 20.0,init_gamma: float = 2.0):
        super().__init__()
        if per_channel:
            assert dim is not None, "dim (hidden size) required for per-channel parameters"
            self.alpha = nn.Parameter(torch.full((dim,), init_alpha, dtype=torch.float32))
            self.beta  = nn.Parameter(torch.full((dim,), init_beta,  dtype=torch.float32))
            self.gamma = nn.Parameter(torch.full((dim,), init_gamma, dtype=torch.float32))
        else:
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
            self.beta  = nn.Parameter(torch.tensor(init_beta,  dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

        self.clamp = clamp  # prevent extreme saturation if desired

    def forward(self, x):
        z = self.alpha * x + self.beta
        if self.clamp is not None:
            z = z.clamp(-self.clamp, self.clamp)
        return self.gamma * torch.sigmoid(z)


# Gating for activations
# ───────────────────────────── gated blend module ───────────────────────────── #

class GatedBlend_2(nn.Module):
    """
    y = w * ParamSigmoid2(x) + (1 - w) * ParamLeakyReLU2(x)
    where w = sigmoid(g).  Initial g = 0 → w = 0.5  (balanced start).
    """
    def __init__(self, dim: int, per_channel: bool = True,
                 init_alpha: float = 1.0, init_beta: float = 0.0,
                 init_a: float = 0.25):
        super().__init__()
        self.sig_act = ParamSigmoid2_2(dim, init_alpha, init_beta, per_channel)
        self.lrelu_act = ParamLeakyReLU2_2(dim, init_a, per_channel)

        # gate g; initialise to 0 so sigmoid(g)=0.5
        if per_channel:
            self.g = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.g = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        w = torch.sigmoid(self.g)                # [dim] or scalar
        return w * self.sig_act(x) + (1.0 - w) * self.lrelu_act(x)

###########################################################################################################################
# ─────────────────────────────── new MLP variants ───────────────────────────── #

class MLP3_Gated_2(nn.Module):
    """
    Linear -> GatedBlend(ParamSig+ParamLReLU) -> Linear -> ParamLeakyReLU -> Linear -> output
    """
    def __init__(self, input_dim=1024, hidden=512, per_channel=True):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden)
        self.act1 = GatedBlend_2(hidden, per_channel)          # <-- blended first layer
        self.fc2  = nn.Linear(hidden, hidden)
        self.act2 = ParamLeakyReLU2(hidden, init_a=0.25, per_channel=per_channel)
        #self.act2 = GatedBlend_2(hidden, per_channel)          # <-- blended first layer
        self.fc3  = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class MLP3_Gated_embed_2(nn.Module):
    """
    Same as above but also returns the final hidden representation.
    """
    def __init__(self, input_dim=1024, hidden=512, per_channel=True):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden)
        self.act1 = GatedBlend_2(hidden, per_channel)
        self.fc2  = nn.Linear(hidden, hidden)
        self.act2 = ParamLeakyReLU2(hidden, init_a=0.25, per_channel=per_channel)
        #self.act2 = GatedBlend_2(hidden, per_channel)
        self.fc3  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        pred = self.fc3(h)
        return pred, h

########################################################################################################################
################################################################################################################
# Mean Embedd Regulairsation with some improvements 
import torch
import torch.nn.functional as F

# Mean Embedding Regularization with optional quantile bins (computed from a dataloader)
class MeanEmbedRegularization3:
    def __init__(self,
                 embed_dim=1024,
                 num_buckets=20,
                 device='cuda',
                 use_ema=True,
                 beta=0.997,
                 use_quantile_bins=False,
                 bin_edges=None,
                 dataloader=None,          # <- pass your train dataloader here to auto-compute edges
                 score_key="score",        # <- key in each batch holding scores/MOS
                 sample_limit=None,        # <- optional cap on how many scores to sample for speed
                 refresh_each_epoch=False  # <- call begin_epoch(...) to refresh if True
                 ):
        self.C = num_buckets
        self.D = embed_dim
        self.device = device
        self.use_ema = use_ema
        self.beta = beta
        self.use_quantile_bins = use_quantile_bins
        self.score_key = score_key
        self.sample_limit = sample_limit
        self.refresh_each_epoch = refresh_each_epoch

        # Running “sufficient statistics” (fp32, no grad)
        self.bucket_sums   = torch.zeros(self.C, self.D, device=device)
        self.bucket_counts = torch.zeros(self.C, 1,       device=device)
        self.bucket_means  = torch.zeros(self.C, self.D, device=device)

        # Bin edges
        if self.use_quantile_bins:
            if bin_edges is not None:
                assert len(bin_edges) == self.C + 1
                self.bin_edges = bin_edges.to(device)
            elif dataloader is not None:
                self.bin_edges = self._compute_bin_edges_from_loader(
                    dataloader, score_key=self.score_key, sample_limit=self.sample_limit
                )
            else:
                raise ValueError(
                    "use_quantile_bins=True requires either bin_edges or a dataloader to compute them."
                )
        else:
            # uniform edges are implicit; no bin_edges needed
            self.bin_edges = None

    # ---- epoch hook (optional) ----
    @torch.no_grad()
    def begin_epoch(self, dataloader=None):
        """
        Call at the start of an epoch if refresh_each_epoch=True and use_quantile_bins=True.
        """
        if self.use_quantile_bins and self.refresh_each_epoch:
            assert dataloader is not None, "dataloader is required to refresh quantile edges."
            self.bin_edges = self._compute_bin_edges_from_loader(
                dataloader, score_key=self.score_key, sample_limit=self.sample_limit
            )

    # ---- quantile edges from loader ----
    @torch.no_grad()
    def _compute_bin_edges_from_loader(self, dataloader, score_key="score", sample_limit=None):
        scores_cpu = []
        seen = 0
        for batch in dataloader:
            s = batch[score_key]
            if not torch.is_tensor(s):
                s = torch.as_tensor(s)
            s = s.view(-1).float().cpu()
            scores_cpu.append(s)
            seen += s.numel()
            if sample_limit is not None and seen >= sample_limit:
                break
        if len(scores_cpu) == 0:
            raise ValueError("No scores collected from dataloader to compute quantile edges.")
        scores = torch.cat(scores_cpu, dim=0).to(self.device)

        qs = torch.linspace(0, 1, self.C + 1, device=self.device)
        #edges = torch.quantile(scores, qs)
        edges = torch.quantile(scores, torch.linspace(0, 1, self.C + 1, device=self.device))
        # ensure strictly non-decreasing and pad range a touch
        eps = 1e-7
        edges = torch.maximum(edges, edges.roll(1, 0) + eps)
        edges[0]  = 0.0 - eps
        edges[-1] = 1.0 + eps

        # DDP sync so all ranks use the same edges
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(edges, src=0)
        return edges

    @torch.no_grad()
    def _assign_buckets(self, scores: torch.Tensor):
        if self.use_quantile_bins:
            idx = torch.bucketize(scores, self.bin_edges) - 1
            return idx.clamp_(0, self.C - 1)
        # uniform equal-width bins over [0,1]
        eps = torch.finfo(scores.dtype).eps
        s = scores.clamp_(0.0, 1.0 - eps)
        idx = torch.floor(s * self.C).long()
        return idx.clamp_(0, self.C - 1)

    @torch.no_grad()
    def update_buckets(self, embeddings: torch.Tensor, scores: torch.Tensor):
        """
        Safe-by-default: detaches, disables autocast, casts to fp32.
        Call this AFTER optimizer.step().
        """
        # keep numerics stable (fp32), ensure no grad
        from torch.cuda.amp import autocast
        with autocast(enabled=False):
            E = embeddings.detach().float()                  # [B, D] fp32
            S = scores.detach().float().view(-1)             # [B]    fp32

            E = F.normalize(E, dim=-1, eps=1e-8)
            idx = self._assign_buckets(S)                    # [B]

            sums   = torch.zeros_like(self.bucket_sums)      # [C, D]
            counts = torch.zeros_like(self.bucket_counts)    # [C, 1]
            sums.index_add_(0, idx, E)
            counts.index_add_(0, idx, torch.ones_like(S).unsqueeze(1))

            # DDP sync so all ranks see the same bank
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(sums)
                torch.distributed.all_reduce(counts)

            if self.use_ema:
                self.bucket_sums   = self.beta * self.bucket_sums   + (1 - self.beta) * sums
                self.bucket_counts = self.beta * self.bucket_counts + (1 - self.beta) * counts
            else:
                self.bucket_sums   += sums
                self.bucket_counts += counts

            means = self.bucket_sums / self.bucket_counts.clamp_min(1.0)
            self.bucket_means = F.normalize(means, dim=-1, eps=1e-8)

    def bucket_contrastive_loss(self, z: torch.Tensor, pred_scores: torch.Tensor,
                                epoch: int, temperature: float = 0.2):
        if epoch < 1:
            return torch.tensor(0.0, device=self.device)

        z_norm = F.normalize(z, dim=-1, eps=1e-8)            # [B, D]

        # labels do NOT carry a graph
        with torch.no_grad():
            true_buckets = self._assign_buckets(pred_scores.detach().view(-1))

        logits = (z_norm @ self.bucket_means.T) / temperature  # [B, C]

        # inverse-frequency weighting (defensive clamp)
        with torch.no_grad():
            freq = self.bucket_counts.squeeze(1).clamp_min(1.0)
            inv  = (freq.sum() / freq)
            w    = (inv / inv.mean()).clamp(max=(inv.mean() * 3))  # cap extreme weights

        return F.cross_entropy(logits, true_buckets, weight=w)

################################################################################################################