from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from torchvision.utils import save_image
generator = DiffusionPipeline.from_pretrained("yousirong/ddpm-butterflies-128").to("cuda")
image = generator().images[0]
image

# 이미지 표시
plt.imshow(image.permute(1, 2, 0).cpu().numpy())
plt.show()


# 이미지 저장
save_image(image, 'generated_image.png')
