from diffusers import StableDiffusionPipeline
import torch
import os

cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

for i in range(len(cifar10_classes)):
    prompt = "Full-body image of a realistic "+cifar10_classes[i]+" standing on all fours, centered on a clean white background, highly detailed and photorealistic"
    mypath = "./generated_imgs/"+cifar10_classes[i]
    if not os.path.exists(mypath):
        os.mkdir(mypath)
    for j in range(250):
        image_path = "./generated_imgs/"+ cifar10_classes[i] + "/image_" + str(j)+ ".jpg"
        image = pipe(prompt).images[0]
        image.save(image_path)

# prompt = "Full-body image of a realistic deer standing on all fours, centered on a clean white background, highly detailed and photorealistic"
# image = pipe(prompt).images[0]
# image.save("./retro_cars.png")