
# from diffusers import DiffusionPipeline

# generator = DiffusionPipeline.from_pretrained("yousirong/ddpm-butterflies-128").to("cuda")
# image = generator().images[0]
# image

# # 이미지 저장
# image.save("image.jpg")
from PIL import Image
import numpy as np
from diffusers import DiffusionPipeline

# 이미지를 생성하는 함수
def generate_images(num_images):
    generator = DiffusionPipeline.from_pretrained("yousirong/ddpm-butterflies-128").to("cuda")
    images = [generator().images[0] for _ in range(num_images)]
    return images

# 이미지를 격자 형태로 배열하여 저장하는 함수
def save_grid_of_images(images, grid_size, save_path):
    num_images = len(images)
    num_rows, num_cols = grid_size
    assert num_images <= num_rows * num_cols, "Number of images exceeds grid size"
    # 이미지 크기 가져오기
    image_width, image_height = images[0].size
    # 격자 이미지 생성
    grid_image = Image.new('RGB', (image_width * num_cols, image_height * num_rows))
    # 이미지를 격자에 삽입
    for idx, img in enumerate(images):
        row_idx = idx // num_cols
        col_idx = idx % num_cols
        grid_image.paste(img, (col_idx * image_width, row_idx * image_height))
    # 격자 이미지 저장
    grid_image.save(save_path)

# 이미지 생성
num_images = 16  # 16개의 이미지 생성
images = generate_images(num_images)

# 격자로 배열하여 저장
grid_size = (2, 8)  # 2행 8열의 격자
save_path = "grid_images.jpg"  # 저장할 파일 경로
save_grid_of_images(images, grid_size, save_path)



