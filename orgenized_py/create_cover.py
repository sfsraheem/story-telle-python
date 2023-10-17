import os
from os import path

# !pip install --upgrade Pillow
try:
  from PIL import Image, ImageFont, ImageDraw
except:
  os._exit(00)

path_to_fonts = 'fonts'
if not path.exists(path_to_fonts):
  os.mkdir(path_to_fonts)

# !wget -o path_to_fonts "https://www.freefontspro.com/d/14454/arial.zip"
# !unzip -o arial.zip

# !pip install fpdf
# !pip install --upgrade openai
#
# !wget -o path_to_fonts "https://www.freefontspro.com/d/14779/baskerville.zip"
# !unzip -o baskerville.zip

import getpass, os
import io
import warnings

# from google.colab import drive, files, widgets
# import ipywidgets as ipywidgets
# from IPython.display import display
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from torchvision.transforms import GaussianBlur
from fpdf import FPDF
from torchvision.transforms import GaussianBlur
from PIL import Image
import openai
import requests
from io import BytesIO

global styles_dict
styles_dict = {
    'fantasy': 'drawn by disney concept artists, high quality, highly detailed, elegant, sharp focus,'
               ' concept art, digital painting',
    'superhero': 'background hyper detailed, character concept, full body, dynamic pose,'
                 ' intricate, highly detailed, digital painting, artstation, concept art, smooth, sharp focus',
    'elegant': 'wind, sky, clouds, the moon, moonlight, stars, universe, fireflies, butterflies, lights, '
               'lens flares effects, swirly bokeh, brush effect'
    # 'fantasy': 'fantasy art drawn by disney concept artists, golden colour, high quality, highly detailed, elegant, sharp focus, concept art, character concepts, digital painting, mystery, adventure',
    # 'fantasy': 'fantasy art by disney artists, high quality, highly detailed, elegant, sharp focus, concept art, digital painting',
    # 'superhero': 'background hyper detailed, character concept, full body, dynamic pose, intricate, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha',
    # 'elegant': 'wind, sky, clouds, the moon, moonlight, stars, universe, fireflies, butterflies, lights, lens flares effects, swirly bokeh, brush effect, In style of Yoji Shinkawa, Jackson Pollock, wojtek fus, by Makoto Shinkai, concept art, celestial, amazing, astonishing, wonderful, beautiful, highly detailed, centered'
  }


#@title Create of the cover { display-mode: "form" }
def make_space_for_title(s):
    words = s.split()
    curr_len = 0
    for i, word in enumerate(words):
        if curr_len + len(word) > 15:
            words[i] = '\n' + word
            curr_len = 0
        curr_len += len(word) + 1
    return ' '.join(words).replace(' \n', '\n')


def generate_img(prompt, base_image=None, mask=None, samples=1, dalle=False, seed=42, start_schedule=0.1):
  print(prompt)
  if dalle:
    print("Generating - DALLE")
    API_key = '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'
    prompt = prompt
    base_image.save("img.png","PNG")
    mask = mask.rotate(angle=90.0)
    mask.save("mask.png","PNG")
    openai.api_key = API_key
    # response = openai.Image.create_edit(
    #   image=open("img.png", "rb"),
    #   mask=open("mask.png", "rb"),
    #   prompt=prompt,
    #   n=1,
    #   size="512x512"
    # )
    response = openai.Image.create(
      prompt=prompt,
      n=1,
      size="512x512"
    )
    images = []
    # iterating over the generator produces the api response
    for resp in response['data']:
      image_url = resp['url']
      image = requests.get(image_url)
      image = Image.open(BytesIO(image.content))
      images.append(image)
  else:
    print("Generating - Stable Diffusion")
    API_key = 'sk-2uM6WfcdKbfJCr9G0PV1rjBUI1KMC6tAWZB5U1u8nYE4V9F5'

    stability_api = client.StabilityInference(
        key=API_key,
        verbose=False,
        engine="stable-diffusion-512-v2-1"
    )

    # the object returned is a python generator
    answers = stability_api.generate(
        prompt=prompt,
        samples=samples,
        init_image=base_image,
        mask_image=mask,
        seed=seed,
        start_schedule=start_schedule,
    )

    images = []
    # iterating over the generator produces the api response
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                # save the image to disk
                # img.save('{}/{}.png'.format(output_folder, prompt))
                images.append(img)
  return images


def best_img_clicked(b, images, pdf, font, output):
  # Display the message within the output widget.
  global best_img
  best_img = int(b.description)
  with output:
    print("image {} was shosen".format(best_img))
    image_draw = ImageDraw.Draw(images[best_img])
    image_draw.text((x, y), title, fill=(0, 0, 0), font=font)
    images[best_img].save(f"{title}_title.png","PNG")
    pdf.add_page()
    pdf.image(f"{title}_title.png", x=0, y=0)
    print("best_img:", best_img)



def main_create_cover(title, USE_DALLE):
  title = make_space_for_title(title)
  output = 1 # chosen img
  model = "Regular"
  if model == "Premuim":
    USE_DALLE = True
  else:
    USE_DALLE = False

  # Prepare the mask for generating the image
  base_image = Image.new("RGB", (512, 512), 'white')
  # base_image.paste(Image.new('RGB', (256, 256), 'white'), (256, 256, 512, 512))
  # TODO add noise to the image

  image_draw = ImageDraw.Draw(base_image)
  fontsize = 60
  max_value = 205
  font = 'arial'
  if font == 'arial':
    font = ImageFont.truetype("arial.ttf", fontsize)
  else:
    font = ImageFont.truetype("baskerville.ttf", fontsize)
  # Add Text to an image
  y = 200
  x = 62
  position = (x, y)
  left, top, right, bottom = image_draw.textbbox(position, title, font=font)
  image_draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill="white")

  # prepare mask
  polygon = [(left - 5, top - 5), (right + 5, top - 5), (right + 5, bottom + 5), (left - 5, bottom + 5)]
  mask = Image.new('L', (512, 512), 0)
  ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
  mask = np.array(mask)

  if not USE_DALLE:
    strength = .9  # This controls the strength of our prompt relative to the init image.

    d = int(255 * (1 - strength))
    mask *= (255 - d)  # Converts our range from [0,1] to [0,255]
    mask += d
    mask[mask == 255] = max_value

  mask = Image.fromarray(mask)

  blur = GaussianBlur(11, 20)
  mask = blur(mask)

  # Generate the image
  output_folder = 'outputs'
  samples = 3

  prompt = "{}, {} ".format(Prompt, styles_dict[Style])  # , children's book
  print("Generating the cover images for you..")
  images = generate_img(prompt, base_image=base_image, mask=mask, samples=samples, dalle=USE_DALLE)

  # Choose the image
  # tb = widgets.TabBar([str(i) for i in range(samples)])
  for i in range(samples):
    # Only select the first 3 tabs, and render others in the background.
    image_draw = ImageDraw.Draw(images[i])
    image_draw.text((x, y), title, fill=(0, 0, 0), font=font)
    # with tb.output_to(i, select=(i < 3)):

    plt.figure(figsize=(10, 10), dpi=80)
    plt.imshow(images[i])
    plt.axis('off')
    plt.show()

  # print("What is your prefered version?")

  # create the PDF
  pdf = FPDF('P', 'mm', (180, 180))

  # buttons = []
  # for i in range(samples):
  #   buttons.append(ipywidgets.Button(description=str(i)))
  #
  # output = ipywidgets.Output()
  #
  # for button in buttons:
  #   button.on_click(best_img_clicked)
  #
  # display(*buttons, output)

  best_img_clicked(b, images, pdf, font, output)


if __name__ == '__main__':
  title = "The Fearless Penguin"  # @param {type:"string"}
  Prompt = "A small penguin standing on an iceberg, looking up at the night sky with a determined expression."  # @param {type:"string"}
  Style = "fantasy"  # @param ["fantasy", "superhero", "elegant"]
  # model = "Regular" #@param ["Regular", "Premuim"]
  # Color = "red" #@param ["red", "blue", "white"]

  USE_DALLE = False
  main_create_cover(title, USE_DALLE)