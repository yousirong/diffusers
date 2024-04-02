from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("yousirong/ddpm-butterflies-128").to("cuda")
image = generator().images[0]
image