import os
import abc
from os import path
from PIL import ImageFont
from fonts.ttf import AmaticSC, FredokaOne
from fontTools.ttLib import TTFont
# from fonts.Miriam_Libre import MiriamLibre-Regular.tts
# import fonts.ttf as ttf
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import bidi.algorithm as bidi
import argparse

import re
from PIL import Image, ImageFont, ImageDraw

path_to_fonts = 'fonts'
if not path.exists(path_to_fonts):
  os.mkdir(path_to_fonts)

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
styles_dict = \
    {
    'pop': 'Funko pop figurine, made of plastic, product studio shot, on a white background, '
           'diffused lighting, centered.',
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
negative_prompt = 'duplicates'

global story_font
story_font = AmaticSC # FredokaOne # AmaticSC
# story_font = TTFont('fonts/Miriam_Libre/MiriamLibre-Bold.ttf')

global show_text
show_text = True

#TODO: Move this to a separate file
OPENAI_API_kEY = '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'
openai.api_key = OPENAI_API_kEY
STABILITY_API_KEY = 'sk-2uM6WfcdKbfJCr9G0PV1rjBUI1KMC6tAWZB5U1u8nYE4V9F5'


class StoryText:
    def __init__(self, content, font=story_font, fontsize: int = 20, position: tuple = (10, 10), color=None):
        self.content = content
        self.spaced_content = None
        self.font_type = font
        self.font = ImageFont.truetype(font, fontsize)
        self.font_size = fontsize
        self.color = color #TODO: use
        self.position = position

    def __repr__(self):
        return f"StoryText: {self.content}"

    def set_font_size(self, font_size):
        self.font_size = font_size
        self.font = ImageFont.truetype(self.font_type, self.font_size)


class StoryImage:
    def __init__(self, prompt, img_size=(512, 512)):
        self.prompt = prompt
        self.img_size = img_size
        self.image = None

    def __repr__(self):
        plt.imshow(self.image)
        plt.show()
        return f"StoryImage(prompt={self.prompt}, img_size={self.img_size})"

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    def get_prompt(self):
        return self.prompt

    def set_prompt(self, prompt):
        self.prompt = prompt

    def get_img_size(self):
        return self.img_size

    def set_img_size(self, img_size):
        self.img_size = img_size


class Page(metaclass=abc.ABCMeta):
    def __init__(self, page_size: tuple = (512, 512), text: str = None, text_background: str = None,
                 text_size: int = 50, text_position: tuple = (10, 10), style: str = 'fantasy'):
        self.page_size = page_size
        self.text = text
        self.num_samples = num_samples

        self.image_engine = None
        self.text_engine = None

        self.page_text = None
        self.best_story_image = None
        self.other_story_images = None

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
        if split_by == 'len':
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
        font = ImageFont.truetype(story_font, self.page_text.font_size)
        # Add text to an image
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

    def generate_images(self, prompt=None, num_samples=2, engine="stable-diffusion-512-v2-1", img_size=(512, 512),
                        base_image=None, mask=None, transparency_level=0.0, base_image_importance=0.0,
                        path_to_images=None):
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
        :param path_to_images: The path to the folder in which to save the images
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
                        generation.Prompt(text=self.prompt, parameters=generation.PromptParameters(weight=10)),
                        # generation.Prompt(text=styles_dict[self.style], parameters=generation.PromptParameters(weight=1)),
                        generation.Prompt(text=negative_prompt, parameters=generation.PromptParameters(weight=-1))
                    ],
                    samples=self.num_samples,
                    init_image=base_image,
                    mask_image=self.img_mask,
                )
            except Exception as e:
                print("Try a different prompt")
                print(e)
                return

            images = []
            # iterating over the generator produces the api response
            output_folder = path_to_images
            if output_folder is None:
                output_folder = "generated_images"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            image_index = 1
            for i, resp in enumerate(answers):
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        warnings.warn(
                            "Your request activated the API's safety filters and could not be processed."
                            "Please modify the prompt and try again.")
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        if show_text:
                            img = self.add_text_to_image(image=img)
                        # save the image to disk
                        # if '{}/{}_{}.png' exists, increment the index until it doesn't
                        # while os.path.exists('{}/{}_{}.png'.format(output_folder, self.prompt, image_index)):
                        #     image_index += 1
                        img.save('{}/{}_{}.png'.format(output_folder, prompt, image_index))
                        image_index += 1
                        images.append(img)

        self.other_story_images = images

    def choose_best_image(self, num_choice: int):
        self.best_story_image = self.other_story_images[num_choice - 1]

    def show_all_images_samples(self) -> None:
        for i in range(self.num_samples):
            # show PIL image
            plt.figure(figsize=(10, 10), dpi=80)
            plt.imshow(self.other_story_images[i])
            plt.axis('off')
            plt.show()


class StoryPage(Page):
    def __init__(self, page_size: tuple = (512, 512), text: str = None, text_background: str = None,
                 text_size: int = 30, text_position: tuple = (10, 10), style: str = "fantasy"):
        super().__init__(page_size=page_size, text=text, text_background=text_background, text_size=text_size,
                         text_position=text_position, style=style)
        self.image = None
        if text:
            if text[0] > 'א' and text[0] < 'ת':
                text = bidi.get_display(text)
            self.add_text(StoryText(content=text, fontsize=text_size, position=text_position))

    def make_prompt_suggestion(self, engine: str = 'davinci'):
        response = openai.Completion.create(
            engine=engine,
            prompt="Make a figurative description of the image that can go with the following title"
                   "from a children's book. Title: {}, Image description: ".format(self.page_text.content),
        )
        return response.choices[0].text

    def make_spacing(self, split_by: str = 'char', row_len: int = 52, char_to_split: list = [',', '.', '!', '?']):
        return super().make_spacing(split_by=split_by, char_to_split=char_to_split)

    def add_text(self, text: StoryText):
        super().add_text(text=text)
        self.make_spacing()

    def add_text_to_image(self, image, align='left'):
        return super().add_text_to_image(image=image, align=align)


class CoverPage(Page):
    def __init__(self, page_size: tuple = (512, 512), title: str = None, text_background: str = None,
                 text_size: int = 200, text_position: tuple = (10, 10), style: str = "fantasy", author: StoryText = None):
        super().__init__(page_size=page_size, text_background=text_background,
                         text_position=text_position, text_size=text_size, style=style)
        self.title = title
        self.image = None
        self.mask = None
        self.base_image = None
        self.author = author
        self.prepare_author_name()
        if title:
            if title[0] > 'א' and title[0] < 'ת':
                title = bidi.get_display(title)
            self.add_text(StoryText(content=title, fontsize=text_size))

    def make_prompt_suggestion(self, engine='curie'):
        response = openai.Completion.create(
            engine=engine,
            prompt="Title: The sleepy bear and the alarm. "
                   "Image description: A cute bear lies in his bed, with an alarm on the desk behind him."
                   "Title: The kid that had everything"
                   "Image description: A kid in a room full of toys "
                   "Title: The Thirsty Dog "
                   "Image description: A black dog happily plays in a park, chasing a ball"
                   "Title: {}"
                   "Image description: ".format(self.page_text.content),
        )
        return response.choices[0].text

    def make_spacing(self, split_by: str = 'len', row_len: int = 13, char_to_split: str = '\n'):
        super().make_spacing(split_by=split_by, row_len=row_len)

    def add_text(self, text: StoryText):
        super(CoverPage, self).add_text(text=text)
        self.make_spacing()

    def add_author(self, author: StoryText):
        self.author = author
        self.prepare_author_name()

    def add_text_to_image(self, image, align='center'):
        return super().add_text_to_image(image=image, align=align)




class StoryBook:
    def __init__(self, title, author, num_pages,
                 text_size: int = 100, text_position: tuple = (10, 10), style: str = "fantasy"):

        self.title = title
        self.author = author
        self.pages = num_pages
        self.style = style
        self.pdf = FPDF('P', 'mm', (180,180))
        self.pages_list = []

    # def add_cover(self, content):

    def add_page(self, page: Page) -> None:
        self.pages_list.append(page)

    def generate_full_story(self):
        for page in self.pages_list:
            # save the image
            page.best_story_image.save("{}_page{}.png".format(self.title, self.pages_list.index(page)), "PNG")
            self.pdf.add_page()
            self.pdf.image("{}_page{}.png".format(self.title, self.pages_list.index(page)), 0, 0)
        # save the pdf, if it already exists, save it with a different name
        number = 1
        while os.path.exists(f"{self.title}.pdf"):
            self.title = f"{self.title}_{number}"
            number += 1
        self.pdf.output(f"{self.title}.pdf")
            # self.pdf.image(f"{Title}_title.png", x=0, y=0)


if __name__ == '__main__':
    # get args from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, default='generate_cover', help='generate_cover or generate_story',
                        choices=['generate_cover', 'generate_page'])
    parser.add_argument('--book_title', type=str, default='The Story Book', help='The title of the book')
    parser.add_argument('--author_name', type=str, default='The Author', help='The author of the book')
    parser.add_argument('--style', type=str, default='fantasy', help='The style of the book')
    parser.add_argument('--prompt', type=str, default='The cat that had everything', help='The prompt for the book')
    parser.add_argument('--text', type=str, default='The kid that had everything', help='The prompt for the book')
    parser.add_argument('--path_to_images', type=str, default='images', help='The path to the images folder')
    parser.add_argument('--num_samples', type=int, default=3, help='The number of images to generate')
    parser.add_argument('--suggest_prompt', type=bool, default=False, help='Whether to suggest a prompt with openai')
    parser.add_argument('--cover_text_size', type=int, default=70, help='The size of the cover text')
    parser.add_argument('--page_text_size', type=int, default=30, help='The size of the page text')
    parser.add_argument('--base_image_importance', type=float, default=0.0, help='The importance of the base image')
    parser.add_argument('--transparency_level', type=float, default=0.3, help='The transparency level')
    parser.add_argument('--text_background', type=str, default='white', help='The background of the text')
    parser.add_argument('--author_position', type=tuple, default=(30, 420), help='The position of the author')
    parser.add_argument('--author_text_size', type=int, default=20, help='The size of the author text')

    args = parser.parse_args()
    author_name = args.author_name
    book_title = args.book_title
    style = args.style
    prompt = args.prompt
    text = args.text
    command = args.command
    path_to_images = args.path_to_images
    num_samples = args.num_samples
    suggest_prompt = args.suggest_prompt
    cover_text_size = args.cover_text_size
    page_text_size = args.page_text_size
    base_image_importance = args.base_image_importance
    transparency_level = args.transparency_level
    text_background = args.text_background
    author_position = args.author_position
    author_text_size = args.author_text_size

    if command == 'generate_cover':
        author = StoryText(content=author_name, fontsize=author_text_size,
                           position=author_position)

        # Create the book cover
        cover_page = CoverPage(style=style, title=book_title, text_size=cover_text_size,
                               text_background=text_background, author=author)

        cover_page.generate_images(prompt=prompt, num_samples=num_samples, transparency_level=transparency_level,
                                   base_image_importance=base_image_importance, path_to_images=path_to_images)

    elif command == 'generate_page':
        new_page = StoryPage(style=style, text=text, text_size=page_text_size, text_background=text_background)

        # Create images from prompt
        new_page.generate_images(prompt=prompt, num_samples=num_samples, transparency_level=transparency_level,
                                 base_image_importance=base_image_importance, path_to_images=path_to_images)

