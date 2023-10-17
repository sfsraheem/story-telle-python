import os
from os import path

from PIL import Image, ImageFont, ImageDraw


path_to_fonts = 'fonts'
if not path.exists(path_to_fonts):
  os.mkdir(path_to_fonts)


from create_cover import *


def make_space_for_page(s):
    words = s.split()
    curr_len = 0
    for i, word in enumerate(words):
        if curr_len + len(word) > 52:
            words[i] = '\n' + word
            curr_len = 0
        curr_len += len(word) + 1
    return ' '.join(words).replace(' \n', '\n')


def prep_mask(text, x, y):
    # base_image = Image.new("RGB", (512, 512), Color)
    # base_image = images[best_img]
    base_image = Image.new("RGB", (512, 512), 'white')
    # TODO add noise to the image

    image_draw = ImageDraw.Draw(base_image)
    fontsize = 20

    font = ImageFont.truetype("arial.ttf", fontsize)
    # Add text to an image

    position = (x, y)
    left, top, right, bottom = image_draw.textbbox(position, text, font=font)
    image_draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill="white")

    # prepare mask
    polygon = [(left - 5, top - 5), (right + 5, top - 5), (right + 5, bottom + 5), (left - 5, bottom + 5)]
    img = Image.new('L', (512, 512), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)
    strength = .9  # This controls the strength of our prompt relative to the init image.

    d = int(255 * (1 - strength))
    mask *= (255 - d)  # Converts our range from [0,1] to [0,255]
    mask += d

    mask = Image.fromarray(mask)

    blur = GaussianBlur(11, 20)
    mask = blur(mask)
    return mask, base_image


def main_create_page(text, prompt, style, x, y):
    text = make_space_for_page(text)

    mask, base_image = prep_mask(text, x, y)

    output_folder = 'outputs'
    num_samples = 3
    prompt = "{}, {} ".format(prompt, styles_dict[style])  # , children's book
    print("Generating the images for you..")
    images = generate_img(prompt, base_image=base_image, mask=mask, samples=num_samples)

    # tb = widgets.TabBar([str(i) for i in range(samples)])
    for i in range(num_samples):
        # Only select the first 3 tabs, and render others in the background.
        image_draw = ImageDraw.Draw(images[i])
        image_draw.text((x, y), text, fill=(0, 0, 0), font=font)
        # with tb.output_to(i, select=(i < 3)):
        plt.figure(figsize=(10, 10), dpi=80)
        plt.imshow(images[i])
        plt.axis('off')
        plt.show()
        
if __name__ == '__main__':
    y = 64
    x = 12
    text = "So the moral of the story, is clear to see, There's nothing scary, about the dark and the night. It's full of wonders, and beauty to behold, So don't be afraid, and embrace the cold."
    prompt = "A small penguin standing on an iceberg, looking up at the night sky with a determined expression."  # @param {type:"string"}
    style = "fantasy"
    main_create_page(text, prompt, style, x, y)
    

