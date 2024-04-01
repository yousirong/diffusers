from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
import torch
from huggingface_hub import notebook_login
import torch
from PIL import Image
from diffusers import DDPMScheduler
import torch.nn.functional as F
from configs._base_.model.UNet2DModel import model
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
import math
import os
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from accelerate import notebook_launcher
import glob
notebook_login()

@dataclass
class TrainingConfig:
    image_size = 128  # 생성되는 이미지 해상도
    train_batch_size = 1
    eval_batch_size = 1  # 평가 동안에 샘플링할 이미지 수
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no`는 float32, 자동 혼합 정밀도를 위한 `fp16`
    output_dir = "ddpm-butterflies-128"  # 로컬 및 HF Hub에 저장되는 모델명

    push_to_hub = True  # 저장된 모델을 HF Hub에 업로드할지 여부
    hub_private_repo = False
    overwrite_output_dir = True  # 노트북을 다시 실행할 때 이전 모델에 덮어씌울지
    seed = 0
    
config = TrainingConfig()

# datasets 불러오기 (이미지 데이터셋)
config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# 학습 도중에 'preprocess' 함수를 적용
def transform(example):
    images= [preprocess(image.convert("RGB")) for image in 
            example["images"]]
    return {"images": images}

dataset.set_transform(transform)

# Dataloader 설정
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

sample_image = dataset[0]["images"].unsqueeze(0)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image[0].permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

# MSE loss 
noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

# AdamWarmup Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer = optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs)
)



def make_grid(images, rows, cols):
    w, h =images[0].size
    grid = Image.new("RGB", size= (cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def evaluate(configm, epoch, pipeline):
# 랜덤한 노이즈로 부터 이미지를 추출합니다. 역전파 diffusion 과정
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images
    
    image_grid = make_grid(images, rows=4, col=4)
    
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, medel, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_config=os.path.join(config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            if config.push_to_hub:
                repo_id = create_repo(
                    repo_id = config.hub_model_id or Path(config.output_dir).name, exist_ok=True
                ).repo_id
            accelerator.init_trackers("train_example")
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        
        global_step = 0
        
# model training
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_discription(f"Epoch {epoch}")
            
            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                #이미지에 더할 노이즈를 샘플링합니다.
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]
                
                # 각 이미지를 위한 랜덤한 타임스텝(timestep)을 샘플링
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,),
                                        device= clean_images.device,
                                        dtype=torch.int64
                                    )
                # 각 타임스텝의 노이즈 크기에 따라 깨끗한 이미지에 노이즈를 추가
                # forward diffusion
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                with accelerator.accumulate(model):
                    # 노이즈를 반복적으로 예측 
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)
                    
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                progress_bar.update()
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
                
                # 각 epoch가 끝난 후 evaluate()와 몇가지 데모 이미지를 선택적으로 샘플링하고 모델 저장
            if accelerator.is_main_process:
                pipline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs -1:
                    evaluate(config, epoch, pipeline)
                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs -1:
                    if config.push_to_hub:
                        upload_folder(
                            repo_id = repo_id,
                            folder_path = config.output_dir,
                            commit_message =f"Epoch {epoch}",
                            ignore_patterns=["step_*", "epoch_*"],
                        )
                else:
                    pipeline.save_pretrained(config.output_dir)


args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args=args, num_processes=1)


sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])