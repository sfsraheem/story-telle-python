import os
from os import path
from PIL import Image, ImageFont, ImageDraw

# !pip install --upgrade Pillow
# try:
#   from PIL import Image, ImageFont, ImageDraw
# except:
#   os._exit(00)

path_to_fonts = 'fonts'
if not path.exists(path_to_fonts):
  os.mkdir(path_to_fonts)


import getpass, os
import io
import warnings


import numpy as np
from matplotlib import cm
from torchvision.transforms import GaussianBlur
from fpdf import FPDF


from page import Page, styles_dict
from cover_page import CoverPage
from story_text import StoryText
from story_page import StoryPage
from story_image import StoryImage


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
    # Number of samples from image model for each prompt
    num_samples = 3
    mode = 'constant'  # 'constant' # 'user'
    # Make a prompt suggestion for the user by gpt3
    suggest_prompt = False
    cover_text_size = 70
    page_text_size = 30

    base_image_importance = 0.0
    transparency_level = 0.2  # mask [1 - fully transparent, 0 - fully opaque]
    text_background = 'cyan'
    author_position = (30, 420)
    author_text_size =30

    # Welcome message
    print("hello, welcome to storyTelly!")

    # Get author name and book title
    if mode == 'user':
        author_name = input("What is your name? ")
        book_title = input("What is the title of your story? ")
    else:
        author_name = 'John Doe'
        book_title = 'The first day at the Swimming pool'

    print("What style do you want your story to be? ")
    for i, key in enumerate(styles_dict.keys()):
        print("{}. {}".format(i + 1, key))
    is_valid = False
    while not is_valid:
        if mode == 'user':
            input_style = input("Please enter the number of the style you want: ")
        else:
            input_style = '1'
        try:
            style = int(input_style)
            if style < 0 or style > len(styles_dict.keys()) + 1:
                print("Invalid input, please enter a number between 1 and {}".format(len(styles_dict.keys()) + 1))
            else:
                is_valid = True
                style = list(styles_dict.keys())[style - 1]
        except ValueError:
            print("Invalid input, please enter a number")
    print("You have chosen style {}".format(style))

    print("How many pages do you want your story to have? ")
    if mode == 'user':
        input_num_pages = input("Please enter the number of pages you want: ")
    else:
        input_num_pages = '1'
    is_valid = False
    while not is_valid:
        try:
            num_pages = int(input_num_pages)
            if num_pages < 1:
                print("Invalid input, please enter a number greater than 0")
            elif num_pages > 10:
                print("You can have a maximum of 10 pages")
                num_pages = 10
                is_valid = True
            else:
                is_valid = True
        except ValueError:
            print("Invalid input, please enter a number greater than 0")
    print("You have chosen {} pages".format(num_pages))

    author = StoryText(content=author_name, fontsize=author_text_size,
                       position=author_position)

    # Create the book cover
    cover_page = CoverPage(style=style, title=book_title, text_size=cover_text_size,
                           text_background=text_background, author=author, num_samples=num_samples)

    # if suggest_prompt:
    #     suggested_prompt = cover_page.make_prompt_suggestion(engine='davinci')
    #     print("Suggested prompt: {}".format(suggested_prompt))
    #     if mode == 'user':
    #         input_prompt = input("Do you want to use this prompt? (y/n) ")
    #     else:
    #         input_prompt = 'n'
    #     if input_prompt == 'y':
    #         cover_page.add_prompt(suggested_prompt)
    #     else:
    #         if mode == 'user':
    #             input_prompt = input("Please enter your prompt: ")
    #         else:
    #             input_prompt = 'A cat sits on a chair, looking at the camera'
    # else:
    #     if mode == 'user':
    #         input_prompt = input("Please enter your prompt: ")
    #     else:
    #         input_prompt = "A sad boy with his cat"


    # Get number of pages and style (for prompt) from user

    if suggest_prompt:
        suggested_prompt = cover_page.make_prompt_suggestion(engine='curie')
        print("Suggested prompt: {}".format(suggested_prompt))
        ans = input("Do you want to use this prompt? (y/n) ")
        if ans == 'y':
            pass
        else:
            cover_prompt = input("Please enter your own prompt: ")
    elif mode == 'user':
        cover_prompt = input("Please enter a description for the photo: ")
    else:
        cover_prompt = 'A cat in a yellow swimming suit crying by the pool'

    # uncomment after debugging Mor
    # cover_page.generate_images(prompt=cover_prompt, transparency_level=transparency_level,
    #                              base_image_importance=base_image_importance)
    # cover_page.show_all_images_samples()
    # input_best_image = input("Please choose the best image for your page: ")
    # is_valid = False
    # while not is_valid:
    #     try:
    #         best_image = int(input_best_image)
    #         if best_image < 1 or best_image > num_samples:
    #             print("Invalid input, please enter a number between 1 and {}".format(num_samples))
    #         else:
    #             is_valid = True
    #     except ValueError:
    #         print("Invalid input, please enter a number between 1 and {}".format(num_samples))
    # cover_page.choose_best_image(best_image)



    # Create the book object
    new_book = StoryBook(title=book_title, author=author_name, num_pages=num_pages, style=style)
    # new_book.add_page(cover_page)

    # Add pages to the book
    for page_number in range(1, num_pages + 1):
        if mode == 'user':
            input_text = input("Please enter the text for page {}: ".format(page_number + 1))
        else:
            input_text = 'This is a story about Alon.\nAlon is a very happy kid.' \
                         '\nToday its his first swimming lesson.\nHe is afraid.\nHe doesnt know how to swim.'
        new_page = StoryPage(style=style, text=input_text, text_size=page_text_size, text_background=text_background, num_samples=num_samples)

        if suggest_prompt:
            suggested_prompt = new_page.make_prompt_suggestion(engine='curie')
            print("Suggested prompt: {}".format(suggested_prompt))
            input_prompt = input("Do you want to use this prompt? (y/n) ")
            if input_prompt == 'y':
                pass
            else:
                input_prompt = input("Please enter your own prompt: ")
        elif mode == 'user':
            input_prompt = input("Please enter a description for the photo: ")
        else:
            input_prompt = 'A boy in a yellow swimming suit walking by the swimming pool. ' \
                           'School mates are swimming in the background.' \


        # Create images from prompt
        new_page.generate_images(input_prompt, transparency_level=transparency_level,
                                 base_image_importance=base_image_importance)
        new_page.show_all_images_samples()
        input_best_image = input("Please choose the best image for your page: ")
        is_valid = False
        while not is_valid:
            try:
                best_image = int(input_best_image)
                if best_image < 1 or best_image > num_samples:
                    print("Invalid input, please enter a number between 1 and {}".format(num_samples))
                else:
                    is_valid = True
            except ValueError:
                print("Invalid input, please enter a number between 1 and {}".format(num_samples))
        new_page.choose_best_image(best_image)
        new_book.add_page(new_page)

    new_book.generate_full_story()
