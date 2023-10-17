import os
from os import path
from fonts.ttf import AmaticSC

try:
  from PIL import Image, ImageFont, ImageDraw
except:
  os._exit(00)

path_to_fonts = 'fonts'
if not path.exists(path_to_fonts):
  os.mkdir(path_to_fonts)


class StoryText:
    def __init__(self, content, font=AmaticSC, fontsize: int = 20, position: tuple = (10, 10), color=None):
        self.content = content
        self.spaced_content = None
        self.font_type = font
        self.font = ImageFont.truetype(font, fontsize)
        self.font_size = fontsize
        self.color = color #TODO: use
        self.position = position
        self.spaced_content = None
        self.split_content = None  # after spacing the text, splitting in to chunks

    def __repr__(self):
        return f"StoryText: {self.content}"

    def set_font_size(self, font_size):
        self.font_size = font_size
        self.font = ImageFont.truetype(self.font_type, self.font_size)
