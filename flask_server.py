import io
from flask import Flask, request, send_file, jsonify
from base64 import encodebytes
from PIL import Image
from cover_page import CoverPage
from story_page import StoryPage
from story_text import StoryText
from story_book import StoryBook
from story_image import StoryImage
from page import Page
import json as json
import time

app = Flask(__name__)



# @app.route('/story', methods=['GET'])
# def send_image():
#     return send_file("dogs{}.png".format(request.args.get("pic_num")), mimetype='image/png')

@app.route('/generate_cover', methods=['POST'])
def generate_cover():
# def generate_cover(input_cover_json):
    """
    :param input_cover_json:
        book_title: str = None
        author_name: str = None
        prompt: str = None

        style: str = "fantasy" - optional
        num_samples: int = 3 - optional
        cover_text_size: int = 200 - optional
        page_size: tuple = (512, 512) - optional
        author_text_size: int = 200 - optional
        author_position: tuple = (10, 512 - 100) - optional
        engine="stable-diffusion-512-v2-1" - optional
        base_image=None - optional
        mask=None - optional
        transparency_level=0.0 - optional
        base_image_importance=0.0 - optional
    :return: json
    """
    input_cover_json = request.get_json()
    print(input_cover_json)
    input_cover = input_cover_json
    if 'author_name' in input_cover:
        author_name = input_cover['author_name']
    else:
        raise Exception("author_name is required")
    if 'book_title' in input_cover:
        book_title = input_cover['book_title']
    else:
        raise Exception("book_title is required")
    if 'prompt' in input_cover:
        cover_prompt = input_cover['prompt']
    else:
        raise Exception("prompt is required")

    if 'style' in input_cover:
        style = input_cover['style']
    else:
        style = "fantasy"
    if 'num_samples' in input_cover:
        num_samples = input_cover['num_samples']
    else:
        num_samples = 3
    if 'cover_text_size' in input_cover:
        cover_text_size = input_cover['cover_text_size']
    else:
        cover_text_size = 100
    if 'page_size' in input_cover:
        page_size = input_cover['page_size']
    else:
        page_size = (512, 512)
    if 'author_text_size' in input_cover:
        author_text_size = input_cover['author_text_size']
    else:
        author_text_size = 40
    if 'author_position' in input_cover:
        author_position = input_cover['author_position']
    else:
        author_position = (10, 512 - 100)
    if 'engine' in input_cover:
        engine = input_cover['engine']
    else:
        engine = "stable-diffusion-512-v2-1"
    if 'base_image' in input_cover:
        base_image = input_cover['base_image']
    else:
        base_image = None
    if 'mask' in input_cover:
        mask = input_cover['mask']
    else:
        mask = None
    if 'transparency_level' in input_cover:
        transparency_level = input_cover['transparency_level']
    else:
        transparency_level = 0.4
    if 'base_image_importance' in input_cover:
        base_image_importance = input_cover['base_image_importance']
    else:
        base_image_importance = 0.0
    if 'text_background' in input_cover:
        text_background = input_cover['text_background']
    else:
        text_background = 'white'




    # author_text_size = input_cover_json['author_text_size']
    # author_position = input_cover_json['author_position']
    #
    # text_background = input_cover_json['text_background']
    # cover_text_size = input_cover_json['cover_text_size']
    # style = input_cover_json['style']
    # num_samples = input_cover_json['num_samples']
    # cover_prompt = input_cover_json['cover_prompt']
    # transparency_level = input_cover_json['transparency_level']
    # base_image_importance = input_cover_json['base_image_importance']


    author = StoryText(content=author_name, fontsize=author_text_size,
                       position=author_position)
    cover_page = CoverPage(style=style, title=book_title, text_size=cover_text_size,
                           text_background=text_background, author=author, num_samples=num_samples)
    cover_page.generate_images(prompt=cover_prompt, transparency_level=transparency_level,
                               base_image_importance=base_image_importance)
    # TODO implement autocomplete or make suggestions
    images = cover_page.get_other_story_images()

    encoded_images = []
    for image in images:
        # show the image
        # image.show()
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='PNG')  # convert the PIL image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
        encoded_images.append(encoded_img)
    return jsonify({'result': encoded_images})

    # encoded_images_json = json.dumps(encoded_images, default=lambda o: o.__bytes__())
    # return encoded_images_json
    



    # remove font due to json serialization error
    # cover_page.page_text.font = None
    # cover_page.other_story_images = None
    # cover_page.other_story_images_without_text = None
    # cover_page.author.font = None
    # cover_page.image = None
    # cover_page.base_image = None
    # cover_page.mask = None

    # cover_page_json = json.dumps(cover_page, default=lambda o: o.__dict__)

    # return jsonify({'encoded_images': encoded_images})

# if __name__ == '__main__':
#     res = generate_cover(input_cover_json={
#         "author_name": "John Doe",
#         "book_title": "The Book of John",
#         "prompt": "A big blue house",
#         "transparency_level": .4,
#         "base_image_importance": 0.0,
#                                     })
#     print(res)


@app.route('/generate_page', methods=['POST'])
def generate_page():
# def generate_page(input_page_json):
    """

    :param input_page_json:
    {
        prompt: str = None
        input_text: str = None

        text_background: str = "white" - optional
        style: str = "fantasy" - optional
        num_samples: int = 3 - optional
        page_size: tuple = (512, 512) - optional
        engine="stable-diffusion-512-v2-1" - optional
        base_image=None - optional
        mask=None - optional
        transparency_level=0.0 - optional
        base_image_importance=0.0 - optional

    :return:
    """
    input_page_json = request.get_json()
    # input_page = json.loads(input_page_json)
    input_page = input_page_json
    if 'prompt' in input_page:
        prompt = input_page['prompt']
    else:
        raise Exception("prompt is required")
    if 'input_text' in input_page:
        input_text = input_page['input_text']
    else:
        raise Exception("input_text is required")
    if 'text_background' in input_page:
        text_background = input_page['text_background']
    else:
        text_background = "white"
    if 'style' in input_page:
        style = input_page['style']
    else:
        style = "fantasy"
    if 'page_text_size' in input_page:
        page_text_size = input_page['page_text_size']
    else:
        page_text_size = 30
    if 'num_samples' in input_page:
        num_samples = input_page['num_samples']
    else:
        num_samples = 3
    if 'page_size' in input_page:
        page_size = input_page['page_size']
    else:
        page_size = (512, 512)
    if 'engine' in input_page:
        engine = input_page['engine']
    else:
        engine = "stable-diffusion-512-v2-1"
    if 'base_image' in input_page:
        base_image = input_page['base_image']
    else:
        base_image = None
    if 'mask' in input_page:
        mask = input_page['mask']
    else:
        mask = None
    if 'transparency_level' in input_page:
        transparency_level = input_page['transparency_level']
    else:
        transparency_level = 0.2
    if 'base_image_importance' in input_page:
        base_image_importance = input_page['base_image_importance']
    else:
        base_image_importance = 0.0


    new_page = StoryPage(style=style, text=input_text, text_size=page_text_size, text_background=text_background,
                         num_samples=num_samples)
    new_page.generate_images(prompt, transparency_level=transparency_level,
                                 base_image_importance=base_image_importance)
    # TODO implement autocomplete or make suggestions
    images = new_page.get_other_story_images()

    encoded_images = []
    for image in images:
        # show the image
        # image.show()
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='PNG')
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
        encoded_images.append(encoded_img)
    return jsonify({'result': encoded_images})
    # encoded_images_json = json.dumps(encoded_images)
    # return encoded_images_json

# if __name__ == '__main__':
#     res = generate_page(input_page_json={
#         "prompt": "A big blue house",
#         "input_text": "This is a test",
#         "text_background": "white",
#         "style": "fantasy",
#     })
#     print(res)


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    byte_arr.seek(0)
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/get_images',methods=['POST'])
def get_images():
    input_cover_json = request.get_json()
    print(input_cover_json)
    time.sleep(5)
    ##reuslt  contains list of path images
    result = ['dogs1.png', 'dogs2.png', 'dogs3.png']
    encoded_images = []
    for image_path in result:
        encoded_images.append(get_response_image(image_path))
    return jsonify({'result': encoded_images})


@app.route('/test_fun',methods=['POST'])
def test_fun():
    input_cover_json = request.get_json()
    print(input_cover_json)
    time.sleep(5)
    ##reuslt  contains list of path images
    result = ['dogs1.png', 'dogs2.png', 'dogs3.png']
    encoded_images = []
    for image_path in result:
        encoded_images.append(get_response_image(image_path))
    return jsonify({'result': encoded_images})



@app.route('/',methods=['GET'])
def default_fun():
    ##reuslt  contains list of path images
    result = ['dogs1.png', 'dogs2.png', 'dogs3.png']
    encoded_images = []
    for image_path in result:
        encoded_images.append(get_response_image(image_path))
    return jsonify({'result': encoded_images})

@app.route('/', methods=['GET'])
def index():
    return "StoryTelle server:" + request.args.get("test") + request.args.get("test2")

@app.route('/json', methods=['POST'])
def handle_json():
    data = request.json
    print(data.get('prompt'))
    print(data.get('name'))
    return data

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)


