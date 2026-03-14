'''
Dataloader for various IQA datasets
Author Ankit Yadav
Date: 2025-03-25
'''

import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from tqdm import tqdm

image_size = 512

# For improving the performance of the Dataloaders

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize, ConvertImageDtype, Compose

class InMemoryImageDataset(Dataset):
    def __init__(self, images):
        """
        Args:
            images (List[PIL.Image or torch.Tensor]): List of images.
            image_size (int): Size to which images will be resized.
        """
        
        self.db_name = 'dummy'  # To match your eval loop's expectation

        # Convert and resize if needed
        self.images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))  # Assuming normalized float tensor
            img = img.convert("RGB").resize((image_size, image_size))
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()
        return {'image': img_tensor}



class KonIQ_10K(Dataset):
    def __init__(self, path_to_db,):
        self.root = path_to_db
        self.db_name = 'KonIQ_10K'
        self.image_size = image_size

        if not os.path.exists(self.root):
            raise ValueError(f"Path {self.root} does not exist")

        self.image_dir = os.path.join(self.root, 'koniq10k_512x384/512x384/') 
        self.data_file = os.path.join(self.root, 'koniq10k_scores_and_distributions/koniq10k_scores_and_distributions.csv')

        if not os.path.exists(self.image_dir):
            raise ValueError(f"Path {self.image_dir} does not exist")

        if not os.path.exists(self.data_file):
            raise ValueError(f"Path {self.data_file} does not exist")

        self.data = pd.read_csv(self.data_file)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, score = self.data.iloc[idx]['image_name'], self.data.iloc[idx]['MOS']/100 # As done in reference paper
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size)) # Set this to 224, 224 for saving gpu memory # Set this to 512x512
        # Convert image to tensor
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
        score = torch.tensor(score).float()
        return {'image': image, 'score': score}

class KonIQ_10K_inmemory(Dataset):
    def __init__(self, path_to_db,):
        self.root = path_to_db
        self.db_name = 'KonIQ_10K'
        self.image_size = image_size

        if not os.path.exists(self.root):
            raise ValueError(f"Path {self.root} does not exist")

        self.image_dir = os.path.join(self.root, 'koniq10k_512x384/512x384/') 
        self.data_file = os.path.join(self.root, 'koniq10k_scores_and_distributions/koniq10k_scores_and_distributions.csv')

        if not os.path.exists(self.image_dir):
            raise ValueError(f"Path {self.image_dir} does not exist")

        if not os.path.exists(self.data_file):
            raise ValueError(f"Path {self.data_file} does not exist")

        self.data = pd.read_csv(self.data_file)
          # Load images into memory
        print("Loading images into memory... (approx. 3GB)")
        self.images = []
        for image_name in tqdm(self.data['image_name']):
            image_path = os.path.join(self.image_dir, image_name)
            with Image.open(image_path).convert('RGB') as img:
                self.images.append(img.resize((self.image_size, self.image_size)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, score = self.data.iloc[idx]['image_name'], self.data.iloc[idx]['MOS']/100 # As done in reference paper
       
        image = self.images[idx]
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
        score = torch.tensor(score).float()
        return {'image': image, 'score': score}
        

class CLIVE(Dataset):
    def __init__(self, path_to_db ):
        self.root = path_to_db
        self.db_name = 'CLIVE'
        self.image_size = image_size

        if not os.path.exists(self.root):
            raise ValueError(f"Path {self.root} does not exist")
        
        self.all_images = loadmat(os.path.join(self.root, 'Data/AllImages_release.mat'))['AllImages_release'] # [n][0][0]
        self.all_mos = loadmat(os.path.join(self.root, 'Data/AllMOS_release.mat'))['AllMOS_release']
        self.all_std = loadmat(os.path.join(self.root, 'Data/AllStdDev_release.mat'))['AllStdDev_release']

        self.image_dir = [os.path.join(self.root, 'Images/', image[0][0]) for image in self.all_images]
        self.mos = [float(mos) for mos in self.all_mos[0]]

        self.std = [float(std) for std in self.all_std[0]]
        self.data = pd.DataFrame({'image_name': self.image_dir, 'MOS': self.mos , 'STD': self.std})

        # Ensure we delte the variables after we are done with them
        del self.all_images, self.all_mos, self.all_std, self.image_dir, self.mos, self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, score = self.data.iloc[idx]['image_name'], self.data.iloc[idx]['MOS']/100

        image = Image.open(image_path).convert('RGB')

        # To fit in the GPU memory remove this when using the full dataset on A100
        image = image.resize((self.image_size, self.image_size))

        # Convert image to tensor
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
        score = torch.tensor(score).float()
        return {'image': image, 'score': score}

class CLIVE_inmemory(Dataset):
    def __init__(self, path_to_db ):
        self.root = path_to_db
        self.db_name = 'CLIVE'
        self.image_size = image_size

        if not os.path.exists(self.root):
            raise ValueError(f"Path {self.root} does not exist")
        
        self.all_images = loadmat(os.path.join(self.root, 'Data/AllImages_release.mat'))['AllImages_release'] # [n][0][0]
        self.all_mos = loadmat(os.path.join(self.root, 'Data/AllMOS_release.mat'))['AllMOS_release']
        self.all_std = loadmat(os.path.join(self.root, 'Data/AllStdDev_release.mat'))['AllStdDev_release']

        self.image_dir = [os.path.join(self.root, 'Images/', image[0][0]) for image in self.all_images]
        self.mos = [float(mos) for mos in self.all_mos[0]]
        self.mos = [mos/100 for mos in self.mos]  # Dividing moss by 100 as done in reference paper

        self.std = [float(std) for std in self.all_std[0]]
        self.data = pd.DataFrame({'image_name': self.image_dir, 'MOS': self.mos , 'STD': self.std})  # Dividing moss by 100 as done in reference paper

         # Load images into memory
        print("Loading images into memory... (approx. 1GB)")
        self.images = []
        for image_path in self.image_dir:
            with Image.open(image_path).convert('RGB') as img:
                self.images.append(img.resize((self.image_size, self.image_size)))
                
        # Ensure we delte the variables after we are done with them
        del self.all_images, self.all_mos, self.all_std, self.image_dir, self.mos, self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, score = self.data.iloc[idx]['image_name'], self.data.iloc[idx]['MOS']

        image = self.images[idx]

        # Convert image to tensor
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
        score = torch.tensor(score).float()
        return {'image': image, 'score': score}
        

class SPAQ(Dataset):
    def __init__(self, path_to_db ):
        self.root = path_to_db
        self.db_path = os.path.join(self.root, 'SPAQ_dataset/Annotations/')
        self.db_name = 'SPAQ'
        self.image_size = image_size

        if not os.path.exists(self.root):
            raise ValueError(f"Path {self.root} does not exist")
        
        self.data = pd.read_excel(os.path.join(self.db_path, 'MOS_and_Image_attribute_scores.xlsx'))
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, score = self.data.iloc[idx]['Image name'], self.data.iloc[idx]['MOS']/100 # As done in reference paper
        image_path = os.path.join(self.root, 'TestImage/', image_name)
        image = Image.open(image_path).convert('RGB')

        # To fit in the GPU memory remove this when using the full dataset on A100
        image = image.resize((self.image_size, self.image_size))

        # Convert image to tensor
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
        score = torch.tensor(score).float()
        return {'image': image, 'score': score}

# class SPAQ(Dataset):
#     def __init__(self, path_to_db):
#         self.root       = path_to_db
#         self.db_path    = os.path.join(self.root, 'SPAQ_dataset', 'Annotations')
#         self.image_dir  = os.path.join(self.root, 'TestImage')
#         self.db_name    = 'SPAQ'
#         self.image_size = image_size

#         if not os.path.exists(self.root):
#             raise ValueError(f"Path {self.root} does not exist: {self.root}")

#         # Read the Excel once, normalize MOS to [0,1]
#         df = pd.read_excel(os.path.join(self.db_path, 'MOS_and_Image_attribute_scores.xlsx'))
#         df['mos_norm'] = df['MOS'] / 100.0
#         self.data = df[['Image name', 'mos_norm']].rename(
#             columns={'Image name': 'name', 'mos_norm': 'score'}
#         )

#         # Common pipe: read_image → Resize → float32 in [0,1]
#         self.pipe = Compose([
#             Resize((self.image_size, self.image_size), antialias=True),
#             ConvertImageDtype(torch.float32),
#         ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row        = self.data.iloc[idx]
#         image_name = row['name']
#         score      = torch.tensor(row['score'], dtype=torch.float32)

#         image_path = os.path.join(self.image_dir, image_name)
#         if not os.path.isfile(image_path):
#             raise FileNotFoundError(f"Image not found: {image_path}")

#         # load as [C,H,W] uint8
#         image = read_image(image_path, mode=ImageReadMode.RGB)

#         # resize + to float32 [0,1]
#         image = self.pipe(image)

#         return {'image': image, 'score': score}
    

class AGIQA3K(Dataset):
    def __init__(self, path_to_db ):
        self.root = path_to_db
        self.db_name = 'AGIQA3K'
        self.image_size = image_size

        if not os.path.exists(self.root):
            raise ValueError(f"Path {self.root} does not exist")
        
        self.data = pd.read_csv(os.path.join(self.root, 'data.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, score_quality,score_align = self.data.iloc[idx]['name'], self.data.iloc[idx]['mos_quality']/5, self.data.iloc[idx]['mos_align']/5
        image_path = os.path.join(self.root, 'images/', image_name)
        image = Image.open(image_path).convert('RGB')

        # To fit in the GPU memory remove this when using the full dataset on A100
        image = image.resize((self.image_size, self.image_size))

        # Convert image to tensor
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
        score_quality = torch.tensor(score_quality).float()
        score_align = torch.tensor(score_align).float()

        # If using both quality and align scores
        #  return {'image': image, 'score_quality': score_quality, 'score_align': score_align} 
        return {'image': image, 'score': score_quality}
    
class AGIQA1K(Dataset):
    def __init__(self, path_to_db ):
        self.root = path_to_db
        self.db_name = 'AGIQA1K'
        self.image_size = image_size

        if not os.path.exists(self.root):
            raise ValueError(f"Path {self.root} does not exist")
        
        self.data = pd.read_excel(os.path.join(self.root, 'AIGC_MOS_Zscore.xlsx'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, score = self.data.iloc[idx]['Image'], self.data.iloc[idx]['MOS']/5
        image_path = os.path.join(self.root, 'images/', image_name)
        image = Image.open(image_path).convert('RGB')

        # To fit in the GPU memory remove this when using the full dataset on A100
        image = image.resize((self.image_size, self.image_size)) # Set this to 224, 224 for saving gpu memory # Set this to 512x512

        # Convert image to tensor
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
        score = torch.tensor(score).float()

        return {'image': image, 'score': score}

# class KADID10K(Dataset):
#     def __init__(self, path_to_db ):

#         self.root = path_to_db
#         self.db_name = 'KADID10K'
#         self.image_size = image_size

#         if not os.path.exists(self.root):
#             raise ValueError(f"Path {self.root} does not exist")
        
#         self.data = pd.read_csv(os.path.join(self.root, 'kadid10k/dmos.csv'))

#         self.pipe  = Compose([
#             Resize((image_size, image_size), antialias=True),
#             ConvertImageDtype(torch.float32)    # scales to [0,1]
#         ])


        

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image_name, score = self.data.iloc[idx]['dist_img'], self.data.iloc[idx]['dmos']
#         score = (score - 1)/4 # Normalize the score to [0, 1]
#         image_path = os.path.join(self.root, 'kadid10k/images/', image_name)
#         #image = Image.open(image_path).convert('RGB')
#         image = read_image(image_path, mode=ImageReadMode.RGB)
#         # # To fit in the GPU memory remove this when using the full dataset on A100
#         # image = image.resize((self.image_size, self.image_size))

#         # # Convert image to tensor
#         # image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
#         image = self.pipe(image)
#         score = torch.tensor(score).float()

#         return {'image': image, 'score': score}

class KADID10K(Dataset):
    def __init__(self, path_to_db ):

        self.root = path_to_db
        self.db_name = 'KADID10K'
        self.image_size = image_size

        if not os.path.exists(self.root):
            raise ValueError(f"Path {self.root} does not exist")
        
        self.data = pd.read_csv(os.path.join(self.root, 'kadid10k/dmos.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, score = self.data.iloc[idx]['dist_img'], self.data.iloc[idx]['dmos']
        score = (score - 1)/4 # Normalize the score to [0, 1]
        image_path = os.path.join(self.root, 'kadid10k/images/', image_name)
        image = Image.open(image_path).convert('RGB')

        # To fit in the GPU memory remove this when using the full dataset on A100
        image = image.resize((self.image_size, self.image_size))

        # Convert image to tensor
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
        score = torch.tensor(score).float()

        return {'image': image, 'score': score}



class FLIVE(Dataset):
    def __init__(self, path_to_db ):
        self.image_size = image_size

        self.root = path_to_db
        self.db_name = 'FLIVE'

        # self.pipe  = Compose([
        #     Resize((image_size, image_size), antialias=True),
        #     ConvertImageDtype(torch.float32)    # scales to [0,1]
        # ])

        if not os.path.exists(self.root):
            raise ValueError(f"Path {self.root} does not exist")
        
        self.data = pd.read_csv(os.path.join(self.root, 'labels_image.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, score = self.data.iloc[idx]['name'], self.data.iloc[idx]['mos']/100
        image_path = os.path.join(self.root, 'database/', image_name)
        # Handel image name issue

        try:  
            image = Image.open(image_path).convert('RGB')
            #image = read_image(image_path, mode=ImageReadMode.RGB)
        except FileNotFoundError:#RuntimeError:
            if image_path.split('.')[-2].split('__')[-2].split('/')[-1] == 'AVA':
                image_path = image_path.replace('AVA__', '')
            image = Image.open(image_path).convert('RGB')
            #image = read_image(image_path, mode=ImageReadMode.RGB)
        # # To fit in the GPU memory remove this when using the full dataset on A100
        image = image.resize((self.image_size, self.image_size))

        # # Convert image to tensor
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
        #image = self.pipe(image)
        score = torch.tensor(score).float()

        return {'image': image, 'score': score}

# class FLIVE(Dataset):
#     def __init__(self, path_to_db ):
#         self.image_size = image_size

#         #raise NotImplementedError("FLIVE dataset is not implemented yet some images are missing")
#         self.root = path_to_db
#         self.db_name = 'FLIVE'

#         if not os.path.exists(self.root):
#             raise ValueError(f"Path {self.root} does not exist")
        
#         self.data = pd.read_csv(os.path.join(self.root, 'labels_image.csv'))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image_name, score = self.data.iloc[idx]['name'], self.data.iloc[idx]['mos']/100
#         image_path = os.path.join(self.root, 'database/', image_name)
#         try:  
#             image = Image.open(image_path).convert('RGB')
#             #image = read_image(image_path, mode=ImageReadMode.RGB)
#         except RuntimeError:
#             if image_path.split('.')[-2].split('__')[-2].split('/')[-1] == 'AVA':
#                 image_path = image_path.replace('AVA__', '')
#             image = Image.open(image_path).convert('RGB')
#             #image = read_image(image_path, mode=ImageReadMode.RGB)
#         # # To fit in the GPU memory remove this when using the full dataset on A100
#         #image = Image.open(image_path).convert('RGB')

#         # To fit in the GPU memory remove this when using the full dataset on A100
#         image = image.resize((self.image_size, self.image_size))

#         # Convert image to tensor
#         image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
#         score = torch.tensor(score).float()

#         return {'image': image, 'score': score}

def main():
    flive = FLIVE('./Dataset/FLIVE/database')
    print(flive.__len__())
    print(flive.__getitem__(0))   


if __name__ == '__main__':
    main()