
import os
import abc
from os import path
from fonts.ttf import AmaticSC
import bidi.algorithm as bidi

try:
  from PIL import Image, ImageFont, ImageDraw
except:
  os._exit(00)

path_to_fonts = 'fonts'
if not path.exists(path_to_fonts):
  os.mkdir(path_to_fonts)


import io
import warnings

from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import GaussianBlur
from PIL import Image
import openai
import requests
from io import BytesIO

from story_text import StoryText

global styles_dict
styles_dict = \
    {
    'cute': "very cute illustration for a childrens book, digital art, highly detailed"
            " clear focus, very coherent"
            " concept, atmospheric, fluffy, vibrant colors, trending on artstation, foggy, sun flare",
    'fantasy': 'high quality, highly detailed, elegant, sharp focus, concept art, digital painting, artstation',
    'superhero': 'background hyper detailed, character concept, full body, dynamic pose,'
                 ' intricate, highly detailed, digital painting, artstation, concept art, smooth, sharp focus',
    'elegant': 'wind, sky, clouds, the moon, moonlight, stars, universe, fireflies, butterflies, lights, '
               'lens flares effects, swirly bokeh, brush effect'
  }

global negative_prompt
negative_prompt = 'Blurry, ugly'

#TODO: Move this to a separate file
OPENAI_API_kEY = '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'
openai.api_key = OPENAI_API_kEY
STABILITY_API_KEY = 'sk-2uM6WfcdKbfJCr9G0PV1rjBUI1KMC6tAWZB5U1u8nYE4V9F5'


class Page(metaclass=abc.ABCMeta):
    def __init__(self, page_size: tuple = (512, 512), text: str = None, text_background: str = None,
                 text_size: int = 50, text_position: tuple = (10, 10), style: str = 'fantasy', num_samples: int = 3):
        self.page_size = page_size
        self.text = text
        self.num_samples = num_samples

        self.image_engine = None
        self.text_engine = None

        self.page_text = None
        self.best_story_image = None
        self.other_story_images = None
        self.other_story_images_without_text = None

        self.img_mask = None
        self.prompt = None
        self.base_image = None

        self.text_size = text_size
        self.text_position = text_position

        self.style = style
        self.author = None

        self.text_background = text_background

    def add_text(self, text: StoryText):
        self.page_text = text
        self.page_text.set_font_size(self.text_size)
        self.page_text.position = self.text_position if self.text_position else text.position

    def make_spacing(self, split_by: str = 'len', row_len: int = 52, char_to_split: list = ['!', ',', '.']):
        # TODO make spacing with regard to the text size
        if split_by == 'len':  # number of chars
            words = self.page_text.content.split()
            curr_len = 0
            for i, word in enumerate(words):
                if curr_len + len(word) > row_len:
                    words[i] = '\n' + word
                    curr_len = 0
                curr_len += len(word) + 1
            self.page_text.spaced_content = ' '.join(words).replace(' \n', '\n')
        elif split_by == 'char':
            max_len = 42
            curr_text = self.page_text.content
            for char in char_to_split:
                curr_text = curr_text.replace(char, char + '\n')
            curr_text = curr_text.replace('\n\n', '\n')
            sentences = curr_text.split('\n')
            for i, sentence in enumerate(sentences):
                curr_len = 0
                words = sentence.split()
                for j, word in enumerate(words):
                    if curr_len + len(word) > max_len:
                        words[j] = '\n' + word
                        curr_len = 0
                    curr_len += len(word) + 1
                sentences[i] = ' '.join(words).replace(' \n', '\n')
            self.page_text.spaced_content = '\n'.join(sentences)

    def make_splitting(self, lines_max: int = 3, splits_max: int = 3):
        """
         split the spaced text into chunks by lines number
        :param lines_max:
        :return: update self.page_text.split_content.
        """
        split_content = []
        lines_list = self.page_text.spaced_content.split('\n')
        if len(lines_list) < lines_max:
            self.page_text.split_content = [self.page_text.spaced_content]
        else:
            num_splits = len(lines_list) // lines_max
            residual_lines = len(lines_list) % lines_max

            if num_splits > splits_max:
                raise ValueError(f"Can't split the text into {splits_max} splits")

            # if residual is > 2 - split by num_splits + 1
            if residual_lines >= 2:
                split_content = [lines_list[i * lines_max: (i + 1) * lines_max] for i in range(num_splits)]
                split_content.append(lines_list[num_splits * lines_max:])
            else:
                # 7 lines with max 3 splits : 3, 3, 1 --> 3, 2, 2
                # split so residual will be 2 - taking line from last full split
                split_content = [lines_list[i * lines_max: (i + 1) * lines_max] for i in range(num_splits -1)]
                split_content.append(lines_list[(num_splits -1) * lines_max: num_splits * lines_max - 1])
                split_content.append(lines_list[num_splits * lines_max - 1:])

            self.page_text.split_content = ['\n'.join(split) for split in split_content]


    def prepare_author_name(self):
        if self.author.content[0] > 'א' and self.author.content[0] < 'ת':
            self.author.spaced_content = bidi.get_display('סופר: {}\nמאייר: בינה מלאכותית'.format(self.author.content))
            self.author.content = bidi.get_display(self.author.content)
        else:
            self.author.spaced_content = 'Author: {}\nIllustrator: AI'.format(self.author.content)

    def add_prompt(self, prompt):
        self.prompt = prompt

    def add_mask(self, transparency_level=0.0, base_image_importance=0.0) -> None:
        # TODO: separate between the mask polygon and outside of mask polygon
        text_background = self.text_background
        if transparency_level < 0 or transparency_level > 1:
            print("transparency_level must be between 0 and 1")
            return
        if base_image_importance < 0 or base_image_importance > 1:
            print("base_image_importance must be between 0 and 1")
            return
        # base_image = Image.new("RGB", (512, 512), Color)
        # base_image = images[best_img]
        if self.base_image is None:
            base_image = Image.new("RGB", (512, 512), 'white')
        else:
            base_image = self.base_image
        # TODO add noise to the image

        image_draw = ImageDraw.Draw(base_image)
        img = Image.new('L', (512, 512), 0)
        font = ImageFont.truetype(AmaticSC, self.page_text.font_size)
        # Add text to an image

        # one position (no splits)
        position = self.page_text.position
        left, top, right, bottom = image_draw.textbbox(position, self.page_text.spaced_content, font=font)
        image_draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill=text_background)
        polygon_title = [(left - 5, top - 5), (right + 5, top - 5), (right + 5, bottom + 5), (left - 5, bottom + 5)]
        ImageDraw.Draw(img).polygon(polygon_title, outline=1, fill=1)

        if self.author:
            font = self.author.font
            position = self.author.position
            left, top, right, bottom = image_draw.textbbox(position, self.author.spaced_content, font=font)
            image_draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill=text_background)
            polygon_author = [(left - 5, top - 5), (right + 5, top - 5), (right + 5, bottom + 5), (left - 5, bottom + 5)]
            ImageDraw.Draw(img).polygon(polygon_author, outline=1, fill=1)


        mask = np.array(img)
        inverse_mask = 1 - mask

        mask = int((1 - transparency_level) * 255) * mask
        inverse_mask = int(base_image_importance * 255) * inverse_mask
        mask = mask + inverse_mask

        # strength = 1 - base_image_importance  # This controls the strength of our prompt relative to the init image.
        # d = int(255 * (1 - strength))
        # mask *= (255 - d)  # Converts our range from [0,1] to [0,255]
        # mask += int(d * (1 - transparency_level))

        mask = Image.fromarray(mask)

        blur = GaussianBlur(11, 20)
        mask = blur(mask)

        self.img_mask = mask

    def make_prompt_suggestion(self):
        pass

    def add_text_to_image(self, image, align='left'):
        image_draw = ImageDraw.Draw(image)
        # TODO: add color of text from the text object field
        image_draw.text(self.page_text.position, self.page_text.spaced_content,
                        fill=(0, 0, 0), font=self.page_text.font, align=align)
        if self.author:
            image_draw.text(self.author.position, self.author.spaced_content,
                            fill=(0, 0, 0),
                            font=self.author.font, align='left')
        return image

    def generate_images(self, prompt=None, engine="stable-diffusion-512-v2-1", img_size=(512, 512),
                        base_image=None, mask=None, transparency_level=0.0, base_image_importance=0.0):
        """
        Generates images from a prompt
        :param prompt: A string containing the prompt from which to generate images
        :param num_samples: The number of images to generate
        :param engine: The name of the engine to use (dalle or one og the diffusion engines)
        :param img_size: The size of the images to generate
        :param base_image: The base image to use for the background, it's important only when
                            base_image_importance is not 0
        :param mask: The mask to use for the text, it's important only when transparency_level is not 1
        :param transparency_level: The transparency level of the text background
        :param base_image_importance: The importance of the base image
        :return: A list of images
        """
        if prompt is None:
            print("No prompt given")
            return
        else:
            self.prompt = prompt

        styled_prompt = self.prompt + ' ' + styles_dict[self.style]

        if base_image is None:
            base_image = Image.new("RGB", (512, 512), 'white')
        self.base_image = base_image

        if mask is None:
            self.add_mask(transparency_level=transparency_level, base_image_importance=base_image_importance)

        if engine == 'dalle':
            print("Generating - DALLE")
            base_image.save("img.png", "PNG")
            mask = mask.rotate(angle=90.0)
            mask.save("mask.png", "PNG")
            # response = openai.Image.create_edit(
            #   image=open("img.png", "rb"),
            #   mask=open("mask.png", "rb"),
            #   prompt=prompt,
            #   n=1,
            #   size="512x512"
            # )
            response = openai.Image.create(
                prompt=styled_prompt,
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
            print("Generating - {}".format(engine))
            stability_api = client.StabilityInference(
                key=STABILITY_API_KEY,
                verbose=False,
                engine=engine,
            )
            styled_prompt = self.prompt + ' ' + styles_dict[self.style]
            # the object returned is a python generator
            try:
                answers = stability_api.generate(
                    prompt=
                    [
                        generation.Prompt(text=self.prompt, parameters=generation.PromptParameters(weight=2)),
                        generation.Prompt(text=styles_dict[self.style], parameters=generation.PromptParameters(weight=1)),
                        generation.Prompt(text=negative_prompt, parameters=generation.PromptParameters(weight=-2))
                    ],
                    samples=self.num_samples,
                    init_image=base_image,
                    mask_image=self.img_mask,
                )
                print("Done generating image with the prompt: {}".format(styled_prompt))
            except Exception as e:
                print("Try a different prompt")
                print(e)
                return

            images = []
            images_without_text = []
            # iterating over the generator produces the api response
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        warnings.warn(
                            "Your request activated the API's safety filters and could not be processed."
                            "Please modify the prompt and try again.")
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        img_with_text = self.add_text_to_image(image=img)
                        # save the image to disk
                        # img.save('{}/{}.png'.format(output_folder, prompt))
                        images.append(img_with_text)
                        images_without_text.append(img)

        self.other_story_images = images
        self.other_story_images_without_text = images_without_text

    def get_other_story_images(self):
        return self.other_story_images

    def choose_best_image(self, num_choice: int):
        self.best_story_image = self.other_story_images[num_choice - 1]

    def show_all_images_samples(self) -> None:
        for i in range(self.num_samples):
            # show PIL image
            plt.figure(figsize=(10, 10), dpi=80)
            plt.imshow(self.other_story_images[i])
            plt.axis('off')
            plt.show()

    # def find_best_text_position(self):
    #     # by best story image - the smoothest area
    #     import cv2
    #     orig_img = np.array()
    #     # gray scale --> blur the PIL image
    #     img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    #     img = cv2.GaussianBlur(img, (5, 5), 0)
    #     sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
    #
    #     # edges
    #     edges = cv2.Canny(img, 100, 200)
    #     # find contours
    #     contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



