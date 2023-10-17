import os
from os import path

try:
  from PIL import Image, ImageFont, ImageDraw
except:
  os._exit(00)

path_to_fonts = 'fonts'
if not path.exists(path_to_fonts):
  os.mkdir(path_to_fonts)


from page import Page, styles_dict
from story_text import StoryText


import os
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


import numpy as np

from torchvision.transforms import GaussianBlur
from PIL import Image
import openai


class StoryPage(Page):
    def __init__(self, page_size: tuple = (512, 512), text: str = None, text_background: str = None,
                 text_size: int = 30, text_position: tuple = (10, 10), style: str = "fantasy", num_samples: int = 3):
        super().__init__(page_size=page_size, text=text, text_background=text_background, text_size=text_size,
                         text_position=text_position, style=style, num_samples=num_samples)
        self.image = None
        self.text_positions_list = []  # relevant to text splits

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

    def make_splitting(self, lines_max: int = 4):
        return super().make_splitting(lines_max=lines_max)

    def add_text(self, text: StoryText):
        super().add_text(text=text)
        self.make_spacing()
        self.make_splitting()
        self.gen_positions()

    def gen_positions(self):
        """
        random position for each split
        :return:
        """
        for i in range(len(self.page_text.split_content)):
            random_pos = (int(np.random.choice([0, 1])*self.page_size[1]/2) + 10,
                          np.random.randint(i*(self.page_size[0] - 10)/3 + i*20, (i+1)*(self.page_size[0] - 20)/3),
                          )
            self.text_positions_list.append(random_pos)
    #override

    def add_text_to_image(self, image, align='left'):
        # return super().add_text_to_image(image=image, align=align)
        image_draw = ImageDraw.Draw(image)
        # TODO: add color of text from the text object field
        for split, position in zip(self.page_text.split_content, self.text_positions_list):
            image_draw.text(position, split, fill=(0, 0, 0), font=self.page_text.font, align=align)

        return image

    #override
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

        for split, position in zip(self.page_text.split_content, self.text_positions_list):
            left, top, right, bottom = image_draw.textbbox(position, split, font=font)
            image_draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill=text_background)
            polygon_title = [(left - 5, top - 5), (right + 5, top - 5), (right + 5, bottom + 5), (left - 5, bottom + 5)]
            ImageDraw.Draw(img).polygon(polygon_title, outline=1, fill=1)


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