import matplotlib.pyplot as plt
import json


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

