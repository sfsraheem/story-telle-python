# StoryTellE_py

## Tasks:
- [ ] separate between the mask polygon and outside of mask polygon
- [ ] Check the last page image as the base image for the next page for consistency
- [ ] Add the code for segmentation from Colab
- [ ] If text length is 0, then generate a new image and return it
- [ ] Watermarks

For next meeting:
- [ ] Download the images as PDF - Asaf
- [ ] Another try - generate more images
- [ ] Login and DB - Asaf
- 
- [ ] General page - Create a new page for all the books that are generated
- [ ] Create a profile for the person

# StoryTellE_py README
For next meeting: Mor + Michael + Asaf
https://storytellepy/generate_cover JSON{author, title..} ; return JSON{...} 
https://storytellepy/generate_page JSON{prompt, page_text} ; return JSON{...}
https://storytellepy/generate_character JSON{photo} ; return JSON{...}

Next steps:
https://storytellepy/generate_full_text JSON{author, title, prompt, page_text} ; return JSON{...}
https://storytellepy/generate_full_story JSON{author, title, prompt, page_text} ; return JSON{[]}
