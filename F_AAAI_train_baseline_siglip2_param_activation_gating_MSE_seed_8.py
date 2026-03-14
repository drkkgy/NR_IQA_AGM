'''
finer tune the Evaluation model
Author Ankit Yadav
Date: 2025-02-06
'''

import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
from peft import PromptEncoderConfig, TaskType
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from glob import glob
import wandb
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import torch.distributed as dist
from tqdm import tqdm
import requests
from io import BytesIO
import time
import pandas as pd
import copy
import numpy as np
import json
import random
from torch.utils.data import random_split
import copy

# DEBUG for REMOTE
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["WANDB_MODE"] = "offline"

##################################### GRAD CAM Imports ########################################
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt
import cv2
###############################################################################################
from pathlib import Path

from seed import *



# Importing the text prompt template
from util import Text_Template_baseline,process_prompts_base,custome_UNet2dConditionalModel,register_hooks,FeatureExtractor,QFD_adapter,mlp_3_layer_siglip,margin_loss,QFD_adapter_max_pool,QFD_adapter_avg_pool,DynamicWeightMLP_3_Layer,FiLM_mlp
from util import Overlay, mlp_3_layer_sigmoid_siglip,FNO_MLP, apply_random_noise,ScaleLossSoftmax,BAD_QUALITY_PROMPT,SIGLIPWithMLP_cross_attention_patch_wise,mlp_3_layer_sigmoid_cross_attention_patchwise_with_embed_out
from util import scenes,MeanEmbedRegularization,mlp_2_layer,MeanEmbedRegularization_center_loss
from util import mlp_3_layer,SIGLIPWithMLP,MLP3_ParamActs,ParamSigmoid2, ParamLeakyReLU2,MLP3_Gated

# Importing different datasets
from sklearn.model_selection import train_test_split
from dataset import KonIQ_10K,CLIVE,SPAQ,AGIQA3K,AGIQA1K,FLIVE,KADID10K

# Importing in memory dataset
from dataset import KonIQ_10K_inmemory, CLIVE_inmemory
# Importing the metrics 
from util import metric

# Suppressing warnings 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class diffusion_trainer:
    def __init__(self,model_config,train_config,Data_config=None):

        self.previous_lora_path = None
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
                           

        if model_config is None:
            self.model_config = {
                "model_id" : "google/siglip2-so400m-patch16-512"
            }
        else:
            self.model_config = model_config
        
        self.Data_config = Data_config

        if Data_config is None:
            assert "Please provide the Data_config"
        
        if train_config is None:
            self.train_config = {
                "epochs" : 5, # Number of epochs
                "batch_size" : 4, # Batch size
                "learning_rate" : 5e-5, # Learning rate
                "weight_decay" : 0.01, # Weight Decay
                "checkpoint_steps" : 1000, # Save checkpoint every X steps
                "max_checkpoints" : 2, # Keep only the last N checkpoints for each stage
                "stage_name" : "stage1", # Stage name
                "lora_config" : {
                    "r" : 8, # Number of attention heads
                    "lora_alpha" : 16, # LoRA alpha
                    "lora_dropout" : 0.1, # LoRA dropout
                    "target_modules" : r".*\.(to_q|to_k|to_v)$", # Target modules
                    #"target_modules" : [r".*\.(to_q|to_k)$"], # Target modules
                },
                "DPT_config" : {
                    "no_lernable_tokens" : 50, # Number of attention heads
                },
                "PEFT_Method": "DPT",
                "logger_mode" : "wandb", # wandb/tensorboard Logger mode
                "wandb_project" : "diffusion_trainer_project-2", # wandb project name
                "gradient_accumulation_steps" : 2, # Gradient accumulation steps
                "log_samples_every" : 500, # Log samples every X steps
                "do_eval" : False, # Perform evaluation
                "eval_epoch_steps" : 10, # Evaluation steps
                "lr_scheduler": True, # Use LR Scheduler
                "lr_scheduler_T_max": [10], # LR Scheduler T_max
                "lr_warmup_ratio": 0.1, # LR Warmup Ratio
                "early_stopping": True,  # Enable Early Stopping
                "patience": 3, # Patience for Early Stopping
                "use_gradient_clip": True, # Use Gradient Clipping
                "gradinet_clip": 1.0, # Gradient clipping # 0.5-1.0 is a good range.
                "Resume": True,
                "dry_run" : False # Dry run for debugging True or False
            }
        else:
            self.train_config = train_config

        # Set the dry run flag
        self.dry_run = self.train_config["dry_run"]
        self.processor = AutoProcessor.from_pretrained(self.model_config["model_id"])

    def clean_old_checkpoints(self,stage_name):
        " Keep the last n number of checkpoints"
        checkpoint_files = sorted(glob(f"checkpoints/{stage_name}_step*.pt"), key=os.path.getctime)
        checkpoint_dir = sorted(glob(f"checkpoints/{self.train_config['stage_name']}_step*/"), key=os.path.getctime)
        if len(checkpoint_files) > self.train_config["max_checkpoints"]:
            for checkpoint_file in checkpoint_files[:-self.train_config["max_checkpoints"]]:
                os.remove(checkpoint_file) # Remove the old checkpoints to save disk space
        if len(checkpoint_dir) > self.train_config["max_checkpoints"]:
            for checkpoint_file in checkpoint_dir[:-self.train_config["max_checkpoints"]]:
                os.system(f"rm -rf {checkpoint_file}") # Remove the old checkpoints to save disk space
    

    def Train_loop(self,dataset_id,accelerator,device):
        
        checkpoint_path = None
        # initialising wandb
        device = accelerator.device



        if accelerator.is_main_process:
            wandb.init(project=self.train_config["wandb_project"],name=f"diffusion_trainer_project-3_{self.train_config['stage_name']}_{dataset_id}")
            # Saving the code files for reproducibility.
            code_artifacts = wandb.Artifact(name="source-code", type="code")
            # How to get the name of current file here A:
            code_artifacts.add_file(__file__)
            code_artifacts.add_file("util.py")
            wandb.log_artifact(code_artifacts)

        # Initialising the model
        try:
            model = AutoModel.from_pretrained(self.model_config["model_id"], torch_dtype=torch.bfloat16).to(device) #float32
        except Exception as e:
            print(f"Error in initialising the model: {e}")

        # Initialise the Components of DP-IQA

        
        #qfd_adapter = QFD_adapter_avg_pool().to(device).to(torch.bfloat16) # Setting QFD to do max pooling instead of concatenating.
        mlp = MLP3_Gated(input_dim=1152).to(device).to(torch.bfloat16) 
        # Ensure that the mlp activation params are float32
        with torch.no_grad():
            for m in mlp.modules():
                if isinstance(m, (ParamSigmoid2, ParamLeakyReLU2)):
                    m.float()  

        model.requires_grad_(True)
        #qfd_adapter.requires_grad_(True)
        mlp.requires_grad_(True)


        ######################################### Modify this section if you want to use DPT #######################################
        
        if self.train_config["PEFT_Method"] == "DPT":

            print("Using DPT Parameter Efficient Fine Tuning !!")
            vision_cfg = model.vision_model.config
            
            # Monkey patching the model to use DPT
            model.config.vocab_size = model.config.text_config.vocab_size
            
            dpt_config = PromptEncoderConfig(
                peft_type="P_TUNING",
                task_type = TaskType.FEATURE_EXTRACTION,
                num_layers=vision_cfg.num_hidden_layers,
                num_virtual_tokens=self.train_config["DPT_config"]["no_lernable_tokens"],
                encoder_reparameterization_type="MLP",
                token_dim = vision_cfg.hidden_size,
                num_transformer_submodules=vision_cfg.num_hidden_layers,
                num_attention_heads=vision_cfg.num_attention_heads,

            )

            # Apply to the model
            try:
                model = get_peft_model(model, dpt_config)
                print(model.print_trainable_parameters())
            except Exception as e:
                print(f"Error in applying the DPT to the model: {e}")

            
        ######################################### Modify this section if you want to use DPT #######################################


        ##################################### Modify this secttion if you want to use LoRA ########################################

        if self.train_config["PEFT_Method"] == "LoRA":
            print("Using LoRA Parameter Efficient Fine Tuning !!")
            # Configuring the LoRA
            Lora_config = LoraConfig(r=self.train_config["lora_config"]["r"], 
                                lora_alpha=self.train_config["lora_config"]["lora_alpha"], 
                                lora_dropout=self.train_config["lora_config"]["lora_dropout"], 
                                target_modules=self.train_config["lora_config"]["target_modules"])
            
            # Apply to the model
            try:
                model = get_peft_model(model, Lora_config)
                print(model.print_trainable_parameters())

            except Exception as e:
                print(f"Error in applying the LoRA to the model: {e}")

        # # Merging the LoRA weights from the previous stage
        # if self.previous_lora_path is not None:

        #     lora_state_dict = torch.load(self.previous_lora_path)
        #     try:
        #         set_peft_model_state_dict(pipe.unet, lora_state_dict)
        #         pipe.unet = pipe.unet.merge_and_unload()
        #     except Exception as e:
        #         print(f"Error in merging the LoRA weights: {e}")

        # # Configuring the LoRA
        # config = LoraConfig(r=self.train_config["lora_config"]["r"], 
        #                     lora_alpha=self.train_config["lora_config"]["lora_alpha"], 
        #                     lora_dropout=self.train_config["lora_config"]["lora_dropout"], 
        #                     target_modules=self.train_config["lora_config"]["target_modules"])
        
        # # Apply to the model
        # try:
        #     pipe.unet = get_peft_model(pipe.unet, config)
        # except Exception as e:
        #     print(f"Error in applying the LoRA to the model: {e}")

        # # Loading the Dataset and Dataloader
        # # Load Dataset

        # # Lading the LIAON-8m dataset from the Hugging face dicrectly 
        # # dataset_train = load_dataset("laion/laion-art", split="train")
        # # split_dataset = dataset_train.train_test_split(test_size=0.2)
        # # dataset_train = split_dataset["train"]
        # # dataset_eval = split_dataset["test"]
        ##################################### Modify this secttion if you want to use LoRA ########################################

        
        if dataset_id == "CLIVE":
            print("Using CLIVE Dataset !!")

            full_data = CLIVE_inmemory(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/CLIVE/ChallengeDB_release")
            total_len = len(full_data)
            train_len = int(0.8 * total_len)
            val_len   = total_len - train_len

            train_dataset, eval_dataset = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(Seed))
        elif dataset_id == "KonIQ_10K_CLIVE":
            print("Training on KonIQ_10K and Testing on CLIVE Dataset !!")

            train_dataset = KonIQ_10K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/KonIQ_10K")
            eval_dataset = CLIVE_inmemory(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/CLIVE/ChallengeDB_release")
            total_len_train = len(train_dataset)
            total_len_eval = len(eval_dataset)
            train_dataset,_ = random_split(train_dataset, [total_len_train, 0], generator=torch.Generator().manual_seed(Seed))
            eval_dataset,_ = random_split(eval_dataset, [total_len_eval, 0], generator=torch.Generator().manual_seed(Seed))
        elif dataset_id == "CLIVE_KonIQ_10K":
            print("Training on CLIVE and Testing on KonIQ_10K Dataset !!")

            eval_dataset = KonIQ_10K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/KonIQ_10K")
            train_dataset = CLIVE_inmemory(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/CLIVE/ChallengeDB_release")
            total_len_train = len(train_dataset)
            total_len_eval = len(eval_dataset)
            train_dataset,_ = random_split(train_dataset, [total_len_train, 0], generator=torch.Generator().manual_seed(Seed))
            eval_dataset,_ = random_split(eval_dataset, [total_len_eval, 0], generator=torch.Generator().manual_seed(Seed))
        
        elif dataset_id == "KonIQ_10K":
            print("Using KonIQ_10K Dataset !!")
            full_data = KonIQ_10K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/KonIQ_10K")
            total_len = len(full_data)
            train_len = int(0.8 * total_len)
            val_len   = total_len - train_len
            train_dataset, eval_dataset = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(Seed))

        elif dataset_id == "SPAQ":
            print("Using SPAQ Dataset !!")

            full_data = SPAQ(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/SPAQ")
            total_len = len(full_data)
            train_len = int(0.8 * total_len)
            val_len   = total_len - train_len
            train_dataset, eval_dataset = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(Seed))

        elif dataset_id == "KADID10K":
            print("Using KADID10K Dataset !!")

            full_data = KADID10K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/KADID-10K")
            total_len = len(full_data)
            train_len = int(0.8 * total_len)
            val_len   = total_len - train_len
            train_dataset, eval_dataset = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(Seed))

        elif dataset_id == "FLIVE":
            print("Using FLIVE Dataset !!")

            full_data = FLIVE(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/FLIVE")
            total_len = len(full_data)
            train_len = int(0.8 * total_len)
            val_len   = total_len - train_len
            train_dataset, eval_dataset = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(Seed))

        elif dataset_id == "AGIQA3K":
            print("Using AGIQA3K Dataset !!")
            full_data = AGIQA3K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/AGIQA-3k")
            total_len = len(full_data)
            train_len = int(0.8 * total_len)
            val_len   = total_len - train_len
            train_dataset, eval_dataset = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(Seed))

        elif dataset_id == "AGIQA1K":
            print("Using AGIQA1K Dataset !!")
            full_data = AGIQA1K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/AGIQA-1k")
            total_len = len(full_data)
            train_len = int(0.8 * total_len)
            val_len   = total_len - train_len
            train_dataset, eval_dataset = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(Seed))

        else:
            raise ValueError("Invalid dataset_id. Please choose from ['CLIVE', 'KonIQ_10K', 'SPAQ', 'FLIVE' ,'AGIQA3K', 'AGIQA1K']")

        
        
        dataloader_train = DataLoader(train_dataset, batch_size=self.train_config["batch_size"], shuffle=True,drop_last=True,worker_init_fn=seed_worker)
        
        dataloader_eval = DataLoader(eval_dataset, batch_size=self.train_config["batch_size"], shuffle=False,drop_last=True,worker_init_fn=seed_worker)
        

        # Optimizer
        optimizer = torch.optim.Adam(list(model.parameters()) +  list(mlp.parameters()),
                                     lr=self.train_config["learning_rate"],weight_decay=self.train_config["weight_decay"])

        # Scheduler
        scheduler = MultiStepLR(optimizer,milestones=self.train_config['lr_scheduler_T_max'],gamma=0.2)
        
        #Early Stopping
        best_eval_loss = float("inf")
        patience_counter = 0
        best_SRCC = float("-inf")

        # -------------------------------------------------------------
        # Resume-training bookkeeping (actual loading happens *after* prepare)
        # -------------------------------------------------------------
        resume_path = f"resume_state/{self.train_config['stage_name']}_latest.pt"
        global_step = 0
        start_epoch = 0
        best_eval_loss = float("inf")
        patience_counter = 0
        checkpoint = None

        if self.train_config["Resume"] and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            best_eval_loss = checkpoint["best_eval_loss"]
            patience_counter = checkpoint["patience_counter"]
            start_epoch = checkpoint["epoch"]  # continue from next epoch via range starting at this index
            global_step = checkpoint["global_step"]
            best_SRCC = checkpoint.get("best_SRCC", best_SRCC)
            print(f"Resuming training from checkpoint: {resume_path}")
        else:
            print("Starting training from scratch.")
        # Setting the model and modules to train mode
        model.train()
        #qfd_adapter.train()
        mlp.train()
        #Scale_loss.train()

        # Adding the components to the accelerator
        model, mlp, optimizer, scheduler, dataloader_train = accelerator.prepare(
            model, mlp, optimizer, scheduler, dataloader_train
        )

        # -------------------------------------------------------------
        #  Load checkpoint state dicts (after prepare so keys match)
        # -------------------------------------------------------------
        if checkpoint is not None:
            accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
            mlp.load_state_dict(checkpoint["mlp_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Add gradient scaler for mixed precision training

        if os.path.exists('universal_embedding.pt'):
                print('Loading the embeddings from disk')
        


        # tqdm: show overall progress from 0..epochs-1 even when resuming
        for epoch in tqdm(
            range(start_epoch, self.train_config["epochs"]),
            desc="Epochs",
            total=self.train_config["epochs"],
            initial=start_epoch,
        ):
            if self.dry_run:
                cnt=0
            with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True): #float32
                total_loss = 0.0
                accumulated_loss = 0.0
                optimizer.zero_grad()
                


                for batch in tqdm(dataloader_train,desc="Training Batches"):
                    if self.dry_run:
                        # DEbugg
                        cnt+=1
                        if cnt >= 100:
                            break
                    
                    # Check if all required keys are present in the batch
                    if not all(key in batch for key in ["image", "score"]):
                        raise ValueError("Missing required batch keys")
                    


                    images = batch["image"].to(device)
                    inputs = self.processor(images=images, return_tensors="pt").to(model.device)

                    try:
                        features = model.module.get_image_features(**inputs) #module
                    except Exception as e:
                        features = model.get_image_features(**inputs)
                    # extractor.features.clear()
                    score = mlp(features)

                    # Compute SDS Loss
                    loss_mse = torch.nn.functional.mse_loss(score.squeeze(1), batch["score"].to(device))
                    #loss_mae = torch.nn.functional.l1_loss(score.squeeze(1), batch["score"].to(device))
                    loss_margin = margin_loss(score.squeeze(1),batch["score"].to(device))

                    loss = loss_mse + loss_margin

                    w_mean = torch.sigmoid(mlp.module.act1.g).mean().item()
                    wandb.log({"gate/w_mean": w_mean}, commit=False)



                    #loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                    loss = loss / self.train_config["gradient_accumulation_steps"]

                    # Optimize
                    accelerator.backward(loss)
                    accumulated_loss += loss.item()
                    total_loss += loss.item() * self.train_config["gradient_accumulation_steps"]
                    global_step += 1


                    # Gradient Accumulation
                    if (global_step)%self.train_config["gradient_accumulation_steps"] == 0:
                        # Here we use MultiStepLR scheduler so we dont need to call scheduler.step() for steps but 
                        # We will call it for every epoch
                        
                        if self.train_config["use_gradient_clip"]:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_config["gradinet_clip"])
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        #print(f"Step {global_step} - Accumulated Loss: {accumulated_loss:.4f}")
                        tqdm.write(f"Step {global_step} - Accumulated Loss: {accumulated_loss:.4f}")
                        if accelerator.is_main_process:
                            wandb.log({f"{self.train_config['stage_name']}_accumulated_loss":accumulated_loss},commit=False)
                        accumulated_loss = 0.0
                

                    # Save checkpoint every X steps
                    os.makedirs("checkpoints", exist_ok=True)
                    os.makedirs("resume_state", exist_ok=True)
                    if global_step % self.train_config['checkpoint_steps'] == 0 and accelerator.is_main_process:
                        checkpoint_path = f"checkpoints/{self.train_config['stage_name']}_step_train_{dataloader_train.dataset.dataset.db_name}_Test{dataloader_eval.dataset.dataset.db_name}_{global_step}.pt"
                        checkpoint_dir = f"checkpoints/{self.train_config['stage_name']}_step_train_{dataloader_train.dataset.dataset.db_name}_Test{dataloader_eval.dataset.dataset.db_name}_{global_step}/"
                        resume_state_path = f"resume_state/{self.train_config['stage_name']}_latest.pt"
                        print(f"Saving intermediate checkpoint: {checkpoint_path}")

                        try:

                            #torch.save(get_peft_model_state_dict(pipe.unet), checkpoint_path) since we are not using LoRA
                            # Save HF weights directory for inspection
                            accelerator.unwrap_model(model).save_pretrained(checkpoint_dir)
                            #torch.save(qfd_adapter.state_dict(), f"{checkpoint_dir}/qfd_adapter.pt")
                            torch.save(mlp.state_dict(), f"{checkpoint_dir}/mlp.pt")

                            # Note: do NOT update resume_state here; it will be saved at epoch end


                        except Exception as e:
                            print(f"Error in saving the checkpoint: {e}")

                        # Clean up old checkpoints, keeping only the last N
                        self.clean_old_checkpoints(self.train_config['stage_name'])

                    # Log loss for every step
                    if accelerator.is_main_process:
                        wandb.log({f"{self.train_config['stage_name']}_loss":loss.item(),
                                f"{self.train_config['stage_name']}_epoch":epoch+1,
                                f"{self.train_config['stage_name']}_step":global_step,
                                f"{self.train_config['stage_name']}_lr":optimizer.param_groups[0]['lr']})
                    

                if accelerator.is_main_process:
                    wandb.log({f"{self.train_config['stage_name']}_total_loss":total_loss / len(dataloader_train),f"{self.train_config['stage_name']}_epoch":epoch+1},commit=False)
                    print(f"{self.train_config['stage_name']} - Epoch {epoch+1}/{self.train_config['epochs']} - Loss: {total_loss / len(dataloader_train):.4f}")
                
                # Here we call the scheduler for every epoch
                if self.train_config["lr_scheduler"]:
                    print(f"Scheduler Step at Epoch {epoch+1}######################################################")
                    scheduler.step() 
            
                # Run Eval if true and for the step size given

                if self.train_config["do_eval"] and epoch % self.train_config["eval_epoch_steps"] == 0:
                    results,avg_eval_loss,images = self.evaluate_model(model,dataloader_train,dataloader_eval,device,mlp,BAD_QUALITY_PROMPT,Text_Template_baseline,epoch,accelerator=accelerator)
                    SRCC,PLCC = results["SRCC"],results["PLCC"]
                    if accelerator.is_main_process:
                        #print(f"SRCC: {SRCC:.4f}, PLCC: {PLCC:.4f}, Avg. Eval Loss: {avg_eval_loss:.4f}")
                        tqdm.write(f"SRCC: {SRCC:.4f}, PLCC: {PLCC:.4f}, Avg. Eval Loss: {avg_eval_loss:.4f}")
                        wandb.log({f"{self.train_config['stage_name']}_SRCC":SRCC,f"{self.train_config['stage_name']}_PLCC":PLCC,f"{self.train_config['stage_name']}_Avg_Eval_Loss":avg_eval_loss})
                        wandb.log({f"{self.train_config['stage_name']}_grad_cam_images": [wandb.Image(img, caption=f"cam_{i}") for i, img in enumerate(images)]})

                # ------------- BEST CHECKPOINT LOGIC -------------
                if accelerator.is_main_process and SRCC > best_SRCC:
                    best_SRCC = SRCC
                    os.makedirs("best_checkpoints", exist_ok=True)
                    best_dir = (
                        f"best_checkpoints/{self.train_config['stage_name']}_"
                        f"train_{dataloader_train.dataset.dataset.db_name}_"
                        f"test_{dataloader_eval.dataset.dataset.db_name}"
                    )
                    accelerator.unwrap_model(model).save_pretrained(best_dir)
                    torch.save(mlp.state_dict(), f"{best_dir}/mlp.pt")
                    tqdm.write(f"[Best] New best SRCC={best_SRCC:.4f} — saved to {best_dir}")
                    # Clean up old best checkpoints
                    best_ckpts = sorted(glob("best_checkpoints/*"), key=os.path.getctime)
                    if len(best_ckpts) > self.train_config["max_checkpoints"]:
                        for p in best_ckpts[:-self.train_config["max_checkpoints"]]:
                            os.system(f"rm -rf {p}")

                # ------------- Save resume-state at END of epoch -------------
                if accelerator.is_main_process:
                    os.makedirs("resume_state", exist_ok=True)
                    resume_state_path = f"resume_state/{self.train_config['stage_name']}_latest.pt"
                    torch.save(
                        {
                            "epoch": epoch + 1,                # last completed epoch
                            "global_step": global_step,
                            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_eval_loss": best_eval_loss,
                            "patience_counter": patience_counter,
                            "mlp_state_dict": mlp.state_dict(),
                            "best_SRCC": best_SRCC
                        },
                        resume_state_path,
                    )
                    tqdm.write(f"[Checkpoint] Saved epoch-end resume state to {resume_state_path}")

        # Fineal Evaluation
        print("Final Evaluation!!!")
        if self.train_config["do_eval"]:
            results,avg_eval_loss,images = self.evaluate_model(model,dataloader_train,dataloader_eval,device,mlp,BAD_QUALITY_PROMPT,Text_Template_baseline,epoch,accelerator=accelerator)
            SRCC,PLCC = results["SRCC"],results["PLCC"]
            if accelerator.is_main_process:
                print(f"SRCC: {SRCC:.4f}, PLCC: {PLCC:.4f}, Avg. Eval Loss: {avg_eval_loss:.4f}")
                wandb.log({f"{self.train_config['stage_name']}_SRCC":SRCC,f"{self.train_config['stage_name']}_PLCC":PLCC,f"{self.train_config['stage_name']}_Avg_Eval_Loss":avg_eval_loss})
                wandb.log({f"{self.train_config['stage_name']}_grad_cam_images": [wandb.Image(img, caption=f"cam_{i}") for i, img in enumerate(images)]})
        # Save the final model
        print("Saving the final model...")
        checkpoint_path = f"checkpoints/{self.train_config['stage_name']}_step_train_{dataloader_train.dataset.dataset.db_name}_Test{dataloader_eval.dataset.dataset.db_name}_{global_step}.pt"
        checkpoint_dir = f"checkpoints/{self.train_config['stage_name']}_step_train_{dataloader_train.dataset.dataset.db_name}_Test{dataloader_eval.dataset.dataset.db_name}_{global_step}/"
        resume_state_path = f"resume_state/{self.train_config['stage_name']}_latest.pt"
        try:

            accelerator.unwrap_model(model).save_pretrained(checkpoint_dir)
            # torch.save(qfd_adapter.state_dict(), f"{checkpoint_dir}/qfd_adapter.pt")
            torch.save(mlp.state_dict(), f"{checkpoint_dir}/mlp.pt")

            torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_eval_loss": best_eval_loss,
                    "patience_counter": patience_counter,
                    # "qfd_adapter_state_dict": qfd_adapter.state_dict(),
                    "mlp_state_dict": mlp.state_dict(),
                    "best_SRCC": best_SRCC
                }, resume_state_path)


        except Exception as e:
            print(f"Error in saving the checkpoint: {e}")

        # Clean up old checkpoints, keeping only the last N
        self.clean_old_checkpoints(self.train_config['stage_name'])
        
        torch.cuda.empty_cache()

        if accelerator.is_main_process:
            wandb.finish() 
        accelerator.free_memory() # Free up memory 
        return checkpoint_path

   
    def evaluate_model(self,model, dataloader_train,dataloader_test, device, mlp, BAD_QUALITY_PROMPT, Text_Template_baseline,epoch,accelerator=None, write_json=True):
        """

        Evaluation loop

        """
        if accelerator is None:
            raise ValueError("Accelerator is None. Please provide a valid accelerator object.")
        
        print(f"Evaluating Model at Epoch {epoch}")
        all_preds = []
        all_labels = []
        
        model_copy = copy.deepcopy(model)
        mlp_copy = copy.deepcopy(mlp)

        # Handle wrapped vs unwrapped model copies
        base_model_for_cam = model_copy.module if hasattr(model_copy, "module") else model_copy
        combined_model = SIGLIPWithMLP(
                                base_model=base_model_for_cam.float(),
                                mlp_head=mlp_copy.float(),
                                device=device,
                            ).to(device).eval()

        # Setting up Grad CAM
        target_layers = [ combined_model.siglip.base_model.vision_model.embeddings.patch_embedding ]

        cam = GradCAM(
                        model=combined_model,
                        target_layers=target_layers )


        grad_cam_images = []
        total_eval_loss = 0.0
        num_batches = 0
        if self.dry_run:
            cnt = 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): #float32
            for batch in tqdm(dataloader_test, desc="Evaluating"):
                
                if self.dry_run:
                    #debugging
                    cnt+=1
                    if cnt >= 32:
                        break


                if not all(key in batch for key in ["image", "score"]):
                    raise ValueError("Missing required batch keys")

                images = batch["image"].to(device)
                gt_scores = batch["score"].to(device)

                inputs = self.processor(images=images, return_tensors="pt").to(model.device)
                # inputs = images
                # features = model.module.get_image_features(**inputs)

                # quality_feature = features
                # predicted_scores = mlp(quality_feature).squeeze(1)

                predicted_scores = combined_model(inputs['pixel_values'])

                loss_mse = torch.nn.functional.mse_loss(predicted_scores, batch["score"].to(device))
                loss_margin = margin_loss(predicted_scores,batch["score"].to(device))
                loss = loss_mse + loss_margin

                total_eval_loss += loss.item()
                num_batches += 1
                all_preds.append(predicted_scores.float().detach().cpu().numpy())
                all_labels.append(gt_scores.float().detach().cpu().numpy())

        # Uncomment this section if you want to visualise the Grad CAM but need to manage the memory usage issue for RTX 4060Tis
        # # Visualisation loop 
        # for i,batch in tqdm(enumerate(dataloader_test), desc="Visualising Grad CAM"):
            
        #     if i >= 10: # Visualisation for top 10 images
        #         break
        #     if not all(key in batch for key in ["image", "score"]):
        #         raise ValueError("Missing required batch keys")

        #     images = batch["image"].to(device)
        #     gt_scores = batch["score"].to(device)

        #     inputs = self.processor(images=images, return_tensors="pt").to(model.device)
        #     # features = model.module.get_image_features(**inputs)

        #     # quality_feature = features
        #     # predicted_scores = mlp(quality_feature).squeeze(1)

        #     predicted_scores = combined_model(inputs['pixel_values'])
        #     # Doing Grad Cam visualization
        #     targets = [ RawScoresOutputTarget() for _ in range(predicted_scores.size(0)) ]
        #     with torch.enable_grad():
        #         # Unlock the target paameters 
        #         l = combined_model.siglip.base_model.vision_model.embeddings.patch_embedding
        #         for param in l.parameters():
        #             param.requires_grad = True

        #         with torch.amp.autocast(device_type="cuda", enabled=False):
        #             cam.model = cam.model.float()
        #             grayscale_cams = cam(input_tensor=inputs['pixel_values'].to(torch.float), targets=targets)
        #         # 4) Overlay & display
        #         batch_counter = 0
        #     for img_tensor, heatmap in zip(inputs['pixel_values'].cpu(), grayscale_cams):
        #         # rgb_img = img_tensor.permute(1,2,0).to(torch.float32).numpy()   # H×W×3 in [0,1] # Q: torch.fp32 A:
        #         # print(img_tensor.shape,img_tensor.type())
        #         #vis     = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        #         vis,rgb_img = Overlay(img_tensor, heatmap, alpha=(0.8,0.4))
        #         # Concat the input image and visualisation
        #         vis = cv2.hconcat([vis, rgb_img])
        #         filename = os.path.join("./grad_cam/", f"img{i:02d}_{batch_counter:02d}_{dataloader_test.dataset.dataset.db_name}.jpg")
        #         batch_counter += 1
        #         # Visualise the Heat MAP
        #         #plt.imshow(vis)
        #         # vis_bgr = vis
        #         # rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        #         # if rgb_img_bgr.shape != vis_bgr.shape:
        #         #     vis_bgr = cv2.resize(vis_bgr, (rgb_img_bgr.shape[1], rgb_img_bgr.shape[0]))
        #         # #rgb_img_bgr = (rgb_img_bgr * 255).round().astype(np.uint8)    
        #         # combined = cv2.hconcat([rgb_img_bgr, vis_bgr])
        #         cv2.imwrite(filename, vis)
        #         grad_cam_images.append(filename)
        #         # Log on wandb
                
        #         # # Generating the heatmap
        #         # heatmap_img = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        #         # plt.imshow(heatmap_img)
        #         # overlay_img = cv2.addWeighted(heatmap_img, 0.5, (rgb_img * 255).astype(np.uint8), 0.5, 0)
        #         # plt.imshow(overlay_img)
        #         # plt.imshow((rgb_img * 255).astype(np.uint8))


        # Concatenate all predictions and labels
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        metricalc =  metric()
        # Calculate the evaluation metrics
        srcc = metricalc.calcuate_srcc(y_true, y_pred)
        plcc = metricalc.calculate_plcc(y_true, y_pred)

        avg_eval_loss = total_eval_loss / num_batches
        # Copy the metrics dictionary produced by `metric()` for this epoch
        results = dict(metricalc.result)  # ensure we have a fresh mutable copy
        results['avg_eval_loss'] = float(avg_eval_loss)

        # Determine dataset names *before* we build file_path
        try:
            Test_set_name = dataloader_test.dataset.dataset.db_name
            Train_set_name = dataloader_train.dataset.dataset.db_name
        except:
            Test_set_name = dataloader_test.dataset.db_name
            Train_set_name = dataloader_train.dataset.db_name

        if write_json:
            # === Accumulate epoch-wise results into a single JSON file ===
            os.makedirs("results", exist_ok=True)
            file_path = (
                f"results/results_{self.train_config['stage_name']}_"
                f"Train_{Train_set_name}_Test_{Test_set_name}.json"
            )

            # Load any existing results so we can append to them
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        all_results = json.load(f)
                except json.JSONDecodeError:
                    # Corrupted or empty file – start fresh
                    all_results = {}
            else:
                all_results = {}

            # Use epoch number as key to keep a chronological record
            all_results[str(epoch)] = results

            # Write updated dictionary back to disk
            with open(file_path, "w") as f:
                json.dump(all_results, f, indent=4)

            print(f"Results updated in {file_path} for epoch {epoch}.")

        # Set the models dtype back to the original

        # Freeze the model layers unfrozen 
        l = combined_model.siglip.base_model.vision_model.embeddings.patch_embedding
        for param in l.parameters():
            param.requires_grad = False

        # Delete to free up memory
        del combined_model
        del cam
        del target_layers
        del model_copy
        del mlp_copy
        torch.cuda.empty_cache()

        # Free up memory
        accelerator.free_memory()

        return results, avg_eval_loss, grad_cam_images

    # ------------------------------------------------------------------
    # Utility: evaluate a saved checkpoint (inference-only)
    # ------------------------------------------------------------------
    def evaluate_checkpoint(self, stage_name: str, dataset_id: str, checkpoint_dir: str | None = None):
        """Load `checkpoint_dir` (produced by best_checkpoints) and run
        evaluation on *dataset_id* test split. Results are printed and saved
        the same way as during training.

        checkpoint_dir should contain the HF-formatted backbone weights and an
        `mlp.pt` file.
        """

        # -------------------------------------------------------------
        # 1) Locate checkpoint directory automatically if not provided
        # -------------------------------------------------------------

        if checkpoint_dir is None:
            pattern = f"best_checkpoints/{stage_name}_train_*_test_{dataset_id}"
            matches = sorted(glob(pattern), key=os.path.getctime)
            if not matches:
                raise FileNotFoundError(
                    f"Could not find any checkpoint matching pattern {pattern}."
                )
            checkpoint_dir = matches[-1]  # newest
            print(f"Auto-selected checkpoint: {checkpoint_dir}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) Load backbone & MLP
        model = AutoModel.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16).to(device)

        mlp = mlp_3_layer(input_dim=1152).to(device).to(torch.bfloat16)
        mlp_path = os.path.join(checkpoint_dir, "mlp.pt")
        if not os.path.exists(mlp_path):
            raise FileNotFoundError(f"Expected MLP weights at {mlp_path}")
        state = torch.load(mlp_path, map_location=device)
        try:
            mlp.load_state_dict(state)
        except RuntimeError as e:
            # possibly wrapped with 'module.' prefix → strip it
            cleaned = {k.replace("module.", ""): v for k, v in state.items()}
            mlp.load_state_dict(cleaned)

        model.eval(); mlp.eval()

        # 3) Build dataset (only eval portion)
        if dataset_id == "CLIVE":
            dataset = CLIVE_inmemory(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/CLIVE/ChallengeDB_release")
        elif dataset_id == "KonIQ_10K":
            dataset = KonIQ_10K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/KonIQ_10K")
        elif dataset_id == "SPAQ":
            dataset = SPAQ(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/SPAQ")
        elif dataset_id == "KADID10K":
            dataset = KADID10K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/KADID-10K")
        elif dataset_id == "FLIVE":
            dataset = FLIVE(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/FLIVE")
        elif dataset_id == "AGIQA3K":
            dataset = AGIQA3K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/AGIQA-3k")
        elif dataset_id == "AGIQA1K":
            dataset = AGIQA1K(path_to_db="/media/ankit/drkkgy_backup/Research-2/DP-IQA/DP-IQA_implementation/Dataset/AGIQA-1k")
        else:
            raise ValueError(f"Unknown dataset_id {dataset_id}")

        dataloader = DataLoader(dataset, batch_size=self.train_config["batch_size"], shuffle=False, drop_last=False)

        # 4) Re-use evaluate_model (need dummy train_loader)
        dummy_train_loader = DataLoader(dataset, batch_size=1)  # not used for metrics

        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision="no")

        results, avg_loss, _ = self.evaluate_model(
            model, dummy_train_loader, dataloader, device, mlp,
            BAD_QUALITY_PROMPT, Text_Template_baseline, epoch=0,
            accelerator=accelerator,
            write_json=False,
        )
 
        # ---------------- Grad-CAM for first batch ----------------
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
            os.makedirs("grad_cam_eval", exist_ok=True)

            first_batch = next(iter(dataloader))
            images_tensor = first_batch["image"].to(device)

            # Re-build combined model to feed into GradCAM
            combined_model = SIGLIPWithMLP(
                base_model=model.float(),
                mlp_head=mlp.float(),
                device=device,
            ).to(device).eval()

            target_layers = [combined_model.siglip.base_model.vision_model.embeddings.patch_embedding]
            cam = GradCAM(model=combined_model, target_layers=target_layers)

            inputs = self.processor(images=images_tensor, return_tensors="pt").to(device)
            preds = combined_model(inputs["pixel_values"])  # forward pass
            targets = [RawScoresOutputTarget() for _ in range(preds.size(0))]

            with torch.no_grad():
                grayscale_cams = cam(input_tensor=inputs["pixel_values"], targets=targets)

            saved_files = []
            for idx, (img_t, heatmap) in enumerate(zip(images_tensor.cpu(), grayscale_cams)):
                vis, rgb_orig = Overlay(img_t, heatmap, alpha=(0.8, 0.4))
                out_path = f"grad_cam_eval/cam_{idx}.jpg"
                cv2.imwrite(out_path, vis)
                saved_files.append(out_path)

            print(f"Saved Grad-CAM visualisations: {saved_files}")
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")

        print("=== Evaluation Results ===")
        print(results)
        return results


    def Train_stage(self,stage,datasets):

        # Initialising the Accelerator
        accelerator = Accelerator(mixed_precision="no")
        device = accelerator.device
        # local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # torch.cuda.set_device(local_rank)

        if dist.is_initialized():
            local_device = torch.cuda.current_device()  # This returns an int, e.g., 0, 1, etc.
            dist.barrier(device_ids=[local_device])


        
        # Ensure stage name is a plain string (caller sometimes passes a list)
        if isinstance(stage, (list, tuple)):
            stage_name_normalised = stage[0] if stage else "stage"
        else:
            stage_name_normalised = stage

        self.train_config["stage_name"] = stage_name_normalised
        self.previous_lora_path = self.Train_loop(datasets,accelerator,device)




    

def main():
    model_config = {
        "model_id" : "google/siglip2-so400m-patch16-512"#"google/siglip2-large-patch16-512" # Using SD1.5 as mnentioned in the paper
    }

    Data_config = {
        "LIAON-ART-8M" : {"data_path":""},
        "AVA" : {"data_path":""},
        "AADB" : {"data_path":""},
        "TIFA" : {"data_path":""},
        "HIVE" : {"data_path":""},
    }

    train_config = {
                "epochs" : 15, # Number of epochs
                "batch_size" : 2, # Batch size
                "learning_rate" : 1e-4, # Learning rate as mentioned in paper DP-IQA https://arxiv.org/abs/2405.19996
                "weight_decay" : 0, # Weight Decay as the paper DP-IQA doesnt mentione about Weight Decay so setting it to 0 DP-IQA https://arxiv.org/abs/2405.19996
                "checkpoint_steps" : 5000, # Save checkpoint every X steps 75000 is final ckpt
                "max_checkpoints" : 5, # Keep only the last N checkpoints for each stage
                "stage_name" : "stage1", # Stage name
                "lora_config" : {
                    "r" : 4, # Number of attention heads
                    "lora_alpha" : 8, # LoRA alpha
                    "lora_dropout" : 0.05, # LoRA dropout
                    "target_modules" : r".*\.(q_proj|k_proj)$",
                },
                "DPT_config" : {
                    "no_lernable_tokens" : 200, # Number of attention heads # 50 toke 200 token
                },
                "PEFT_Method": "LoRA", # LoRA or DPT, or NA for Full FT
                "logger_mode" : "wandb", # wandb/tensorboard Logger mode
                "wandb_project" : "diffusion_trainer_project-3", # wandb project name
                "gradient_accumulation_steps" : 6, # Gradient accumulation as the paper DP-IQA menitos a batch size of 12 https://arxiv.org/abs/2405.19996
                "log_samples_every" : 50, # Log samples every X steps
                "do_eval" : True, # Perform evaluation
                "eval_epoch_steps" : 1, # Do eval every X epochs
                "lr_scheduler": True, # Use LR Scheduler
                "lr_scheduler_T_max": [30,35], # LR Scheduler T_max one epoxh has around 35000 steps for our dataset with batch size 2 for MultistepLR its the epoch number * Number of GPUs
                "lr_warmup_ratio": 0.1, # LR Warmup Ratio
                "early_stopping": False,  # Enable Early Stopping
                "patience": 3, # Patience for Early Stopping
                "use_gradient_clip": False, # Use Gradient Clipping as the paper DP-IQA doesnt mentione aboutgradient cliping so setting it to False DP-IQA https://arxiv.org/abs/2405.19996
                "gradinet_clip": 1.0, # Gradient clipping # 0.5-1.0 is a good range.
                "Resume": False,
                "dry_run" : False # Set to True for dry run
            }
    



    #stage = ["stage1","stage2","stage3"]
    stage = ["Baseline_param_activation_gating_MSE"+f"_seed{Seed}"] # ["stage1-bmw"]  #
    datasets = ["CLIVE","KonIQ_10K"] #["CLIVE","AGIQA1K","AGIQA3K"]#["KonIQ_10K_CLIVE","CLIVE","KADID10K","AGIQA1K","AGIQA3K","SPAQ","KonIQ_10K"] #["CLIVE"]#["KonIQ_10K_CLIVE"]#["CLIVE","KADID10K","AGIQA1K","AGIQA3K","SPAQ","KonIQ_10K"]
    
    #trainer.previous_lora_path = "best_checkpoints/stage1.pt" # Path to the previous LoRA weights Do this only if you are continuing from a previous stage
    
    for dataset in datasets:
        trainer = diffusion_trainer(model_config,train_config,Data_config)
        trainer.transform =  transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        trainer.Train_stage(stage,dataset)
        #trainer.evaluate_checkpoint(stage,dataset,"best_checkpoints/Baseline_MSE_train_CLIVE_test_CLIVE")
        del trainer




if __name__ == "__main__":
    main()




