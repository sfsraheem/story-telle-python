from page import Page
from story_text import StoryText
import openai
import bidi.algorithm as bidi


class CoverPage(Page):
    def __init__(self, page_size: tuple = (512, 512), title: str = None, text_background: str = None,
                 text_size: int = 200, text_position: tuple = (10, 10), style: str = "fantasy", author: StoryText = None,
                 num_samples: int = 3):
        super().__init__(page_size=page_size, text_background=text_background,
                         text_position=text_position, text_size=text_size, style=style, num_samples=num_samples)
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
