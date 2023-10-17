
api_replicate_token = "36f7668914f75fddec32286c0d481a005196c6a0"

import os
import replicate
from PIL import Image
import urllib.request

os.environ["REPLICATE_API_TOKEN"] = api_replicate_token

model = replicate.models.get("prompthero/openjourney")
version = model.versions.get("9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb")

inputs = {
    # Input prompt
    # highly detailed stripped cat sitting on a piano at home by studio ghibli, makoto shinkai, by artgerm, by wlop, by greg rutkowski, volumetric lighting, octane render, 4 k resolution, trending on artstation, masterpiece",
    'prompt': "mdjrny-v4 style a cat playing a piano highly detailed stripped cat sitting on a piano at home, children book style, by studio ghibli, makoto shinkai, by artgerm, by wlop, by greg rutkowski, volumetric lighting, octane render, 4 k resolution, trending on artstation, masterpiece",

    # Width of output image. Maximum size is 1024x768 or 768x1024 because
    # of memory limits
    'width': 512,

    # Height of output image. Maximum size is 1024x768 or 768x1024 because
    # of memory limits
    'height': 512,

    # Number of images to output
    'num_outputs': 1,

    # Number of denoising steps
    # Range: 1 to 500
    'num_inference_steps': 50,

    # Scale for classifier-free guidance
    # Range: 1 to 20
    'guidance_scale': 6,

    # Random seed. Leave blank to randomize the seed
    # 'seed': ...,
}

# https://replicate.com/prompthero/openjourney/versions/9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb#output-schema
output_link = version.predict(**inputs)
print(output_link)

save_file_path = 'cat2.png'
urllib.request.urlretrieve(output_link[0], save_file_path)

img = Image.open(save_file_path)
img.show()



print('done')

