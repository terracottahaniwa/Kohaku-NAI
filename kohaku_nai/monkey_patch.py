import io
import sys
import base64
import asyncio
import zipfile
from typing import Any, Tuple, List, Dict

import gradio as gr
from gradio.components.base import (
    Component,
)
import PIL
from PIL.Image import Image, LANCZOS

from modules import shared
from modules.processing import (
    StableDiffusionProcessingImg2Img,
)

import kohaku_nai.utils


def append_info(is_img2img: bool) -> None:
    gr.Markdown(
        "Extra noise multiplier for img2img and hires fix "
        "(`img2img_extra_noise`) is used for Noise parameter",
        visible=is_img2img)


def append_ui(is_img2img: bool) -> Tuple[Component]:
    add_original_image = gr.Checkbox(
        True, label="Add original image", visible=is_img2img)
    free_only = gr.Checkbox(True, label="Free only")
    return (add_original_image, free_only)


def director_tools_ui(is_img2img: bool) -> Tuple[Component]:
    with gr.Accordion(
        label="Director tools", 
        open=True,
        visible=True if is_img2img else False
    ):
        with gr.Row():
            req_type = gr.Dropdown(
                choices=[
                    "disable",
                    "bg_removal",  # keyname with hyphen dosen't work well.
                    "lineart",
                    "sketch",
                    "colorize",
                    "emotion",
                    "declutter",
                ],
                value="disable",
                label="req_type (bg_removal always consumes anlas)",
                interactive=True,
            )
            emotion = gr.Dropdown(
                choices=[
                    "neutral;;",
                    "happy;;",
                    "sad;;",
                    "angry;;",
                    "scared;;",
                    "surprised;;",
                    "tired;;",
                    "excited;;",
                    "nervous;;",
                    "thinking;;",
                    "confused;;",
                    "shy;;",
                    "disgusted;;",
                    "smug;;",
                    "bored;;",
                    "laughing;;",
                    "irritated;;",
                    "aroused;;",
                    "embarrassed;;",
                    "worried;;",
                    "love;;",
                    "determined;;",
                    "hurt;;",
                    "playful;;",
                ],
                value="neutral;;",
                label="emotion (Work on req_type: emotion)",
                interactive=True,
            )
        defry = gr.Slider(
            minimum=0,
            maximum=5,
            value=0,
            step=1,
            label=("defry (Additonal prompt and defry work "
                   "on req_type: colorize and emotion)")
        )
    return (req_type, emotion, defry)


def stop_non_free(width, height, steps, req_type) -> None:
    if kohaku_nai.utils.free_check(width, height, steps) == False:
        raise Exception("Free settings exceeded.")
    if req_type == "bg_removal":
        raise Exception("bg_removal always consumes anlas.")


def image_to_base64png(image: Image) -> str:
    bytes_io = io.BytesIO()
    image.save(bytes_io, format="PNG")
    return base64.b64encode(bytes_io.getvalue()).decode("ascii")


def director_tools(p, req_type, emotion, defry) -> (
        Tuple[List[Image], List[str]]):
    req_type = req_type.replace("_", "-")
    width, height = p.width, p.height
    image = p.init_images[0]
    if not image:
        raise Exception("No image found.")
    image = image.resize((width, height), LANCZOS)
    image = image_to_base64png(image)
    prompt = p.prompt
    if req_type == "emotion":
        prompt = emotion + prompt

    payload = {
        "defry": defry,
        "height": height,
        "image": image,
        "prompt": prompt,
        "req_type": req_type,
        "width": width,
    }
    if req_type not in ("colorize", "emotion"):
        del payload["defry"]
        del payload["prompt"]

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    loop.run_until_complete(
        kohaku_nai.utils.set_client(
            shared.opts.knai_http_backend,
            token=shared.opts.knai_token.strip()
        )
    )
    response = loop.run_until_complete(
        kohaku_nai.utils.global_client.post(
            f"{kohaku_nai.utils.API_IMAGE_URL}/ai/augment-image",
            json=payload
        )
    )
    del payload["image"]
    content = io.BytesIO(response.content)
    with zipfile.ZipFile(content) as archive:
        images = []
        infotexts = []
        for file_name in archive.namelist():
            stream = archive.read(file_name)
            image = PIL.Image.open(io.BytesIO(stream))
            images.append(image)
            infotexts.append(f"{payload=}, {file_name=}")
        return images, infotexts


def i2i_support(payload: dict, **kwargs) -> Dict[str, Any]:
    assert "monkey_patch_context" in kwargs
    i, p, add_original_image = kwargs["monkey_patch_context"]
    is_img2img = isinstance(p, StableDiffusionProcessingImg2Img)
    is_infill = is_img2img and p.image_mask != None

    if is_img2img:
        payload["action"] = "img2img"
        payload["parameters"].update(
            {
                "extra_noise_seed": payload["parameters"]["seed"],
                "image": image_to_base64png(p.init_images[i]),
                "noise": shared.opts.img2img_extra_noise,
                "strength": p.denoising_strength,
            }
        )
    if is_infill:
        payload["model"] = "nai-diffusion-3-inpainting"
        payload["action"]= "infill"
        payload["parameters"].update(
            {
                "add_original_image": add_original_image,
                "mask": image_to_base64png(p.image_mask),
            }
        )
    return payload
