import abc
import numpy as np
from PIL import Image
from create_page import *
class character(metaclass=abc.ABCMeta):
    def __init__(self, name, img_size=(512, 512)):
        self.name = name
        self.img_size = img_size
        self.image = None

    def __repr__(self):
        return f"character(name={self.name}, img_size={self.img_size})"


    def show_image(self):
        self.image.show()

    def get_character_mask(self, background_color=None):
        """
        Returns a mask of the character's image
        Find all non-white pixels and return a mask of them with 255, else 0
        """
        if background_color is None:
            img_grayscale = self.image.convert('L')
            img_a = np.array(img_grayscale)

            mask = np.array(img_grayscale)
            mask[img_a < 150] = 0  # This is the area that will get painted, will show up as grey.
            mask[
                img_a >= 150] = 1  # This is the protected area, will show up white. Protected areas won't be affected by our generating.
            return mask
        elif background_color == 'black':
            filter_color = 0
        else:
            filter_color = 255

        image_grayscale = self.image.convert('L')
        np_image_grayscale = np.array(image_grayscale)
        np_image_grayscale[np_image_grayscale != filter_color] = 1
        np_image_grayscale[np_image_grayscale != 1] = 0
        return np_image_grayscale

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_img_size(self):
        return self.img_size

    def set_img_size(self, img_size):
        self.img_size = img_size


class PiterPan(character):
    def __init__(self, name, img_size=(512, 512)):
        super().__init__(name, img_size)

        self.image = Image.open("characters/peter2.png").resize(img_size)

    def __repr__(self):
        return f"PiterPan(name={self.name}, img_size={self.img_size})"


if __name__ == '__main__':
    peter = PiterPan("Peter Pan")
    prompt = 'Sitting on a tree and eating a cookie'
    mask = peter.get_character_mask(background_color='black')
    inverse_mask = 1 - mask

    transparency_level = .2  # This controls how transparent the mask is.
    base_image_importance = .0  # This controls how much of the base image is visible through the mask.

    mask = int((1 - transparency_level) * 255) * mask
    inverse_mask = int(base_image_importance * 255) * inverse_mask
    mask = mask + inverse_mask
    start_schedule = 0.0


    base_image = peter.get_image()
    # convert to RGB
    base_image = base_image.convert('RGB')
    # convert to grayscale
    # base_image = base_image_orig.convert('L')

    np_base_image = np.array(base_image)

    # add noise to the base image to make it more interesting
    random_noise = np.random.randint(0, 255, size=np_base_image.shape, dtype=np.uint8)
    # stack the mask to make it 3D
    mask = np.stack((mask, mask, mask), axis=2)
    random_noise[mask != 0] = 0

    np_base_image = np_base_image + random_noise

    # grayscale to RGB
    base_image = Image.fromarray(np_base_image, mode='RGB')

    mask = Image.fromarray(mask)
    # show the image in plt
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10), dpi=80)
    # plt.imshow(mask)
    # plt.axis('off')
    # plt.show()
    samples = 1
    prompt = [generation.Prompt(text="Sitting on a tree",
                                parameters=generation.PromptParameters(weight=1)),
              generation.Prompt(text="digital art, Pixar style",
                                parameters=generation.PromptParameters(weight=1)),
              generation.Prompt(text="Ugly, Blurry, Dark, Extra limbs, Extra eyes, Extra mouth",
                                parameters=generation.PromptParameters(weight=-1)),
              ]

    seed = 40
    images = generate_img(prompt, samples=samples, seed=seed, mask=mask,
                          base_image=base_image, start_schedule=start_schedule)
    for i in range(samples):
        plt.figure(figsize=(10, 10), dpi=80)
        plt.imshow(images[i])
        plt.axis('off')
        plt.show()

    prompt = [generation.Prompt(text="Eating a cookie",
                                parameters=generation.PromptParameters(weight=1)),
              generation.Prompt(text="digital art, Pixar style",
                                parameters=generation.PromptParameters(weight=1)),
              generation.Prompt(text="Ugly, Blurry, Dark, Extra limbs, Extra eyes, Extra mouth",
                                parameters=generation.PromptParameters(weight=-1)),
              ]

    images = generate_img(prompt, samples=samples, seed=seed, mask=mask,
                          base_image=base_image, start_schedule=start_schedule)
    for i in range(samples):
        plt.figure(figsize=(10, 10), dpi=80)
        plt.imshow(images[i])
        plt.axis('off')
        plt.show()

# Path: create_page.py
    print("Generating the images for you..")
