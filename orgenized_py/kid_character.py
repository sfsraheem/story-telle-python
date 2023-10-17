from page import *

engine="stable-diffusion-512-v2-1"
stability_api = client.StabilityInference(
    key=STABILITY_API_KEY,
    verbose=False,
    engine=engine,
)

# stable diffusion image 2 image:
init_img = Image.open(r"C:\Users\ventu\Downloads\Telegram Desktop\mikey_baby.jpg")
init_img = init_img.resize((512, 512))
img_prompt = "digital art, disney, bear",
answers = stability_api.generate(
    prompt=img_prompt,
    init_image=init_img, # Assign our previously generated img as our Initial Image for transformation.
    start_schedule=0.3, # Set the strength of our prompt in relation to our initial image.
    seed=123467458, # If attempting to transform an image that was previously generated with our API,
                    # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
    steps=30, # Amount of inference steps performed on image generation. Defaults to 30.
    cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                   # Setting this value higher increases the strength in which it tries to match your prompt.
                   # Defaults to 7.0 if not specified.
    width=512, # Generation width, defaults to 512 if not included.
    height=512, # Generation height, defaults to 512 if not included.
    sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                 # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                 # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
)
images = []
# iterating over the generator produces the api response
for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            # save the image to disk
            img.save('{}.png'.format(img_prompt))
            images.append(img)
plt.figure(figsize=(10, 10), dpi=80)
plt.imshow(images[0])
plt.axis('off')
plt.show()
