from typing import List

import numpy as np
import pyperclip
import skimage.transform
import streamlit as st
from PIL import Image
from streamlit_image_select import image_select

from src.copy_button import copy_mime_button
from src.horde import list_models
from src.utils import (
    image_to_np,
    downscale,
    generate_palette,
    make_seamless,
    get_color_palette,
    remap_image,
    encode_file,
)


class Texture:
    def __init__(self, image: np.array):
        self.raw_image: np.array = image
        self.image: np.array = None
        self.palette: np.array = None
        self._upscaled_image = None

        self.update()

    def update(self):
        self.image = postprocess(self.raw_image)
        self.palette = get_color_palette(self.image)
        self._upscaled_image = None

    @property
    def upscaled_image(self):
        if self._upscaled_image is None:
            self._upscaled_image = skimage.transform.resize(
                texture.image, (512, 512, 4), order=0
            )
        return self._upscaled_image

    @property
    def rgb(self):
        alpha = self.image[:, :, 3] < 128
        color = self.image[:, :, :3].copy()
        color[alpha] = 255
        return color


class Settings:
    def __init__(self):
        self.width = 16
        self.height = 16
        self.remove_background = False
        self.seamless = False
        self.seamless_algorithm = "watershed"
        self.models = ["AlbedoBase XL (SDXL)"]
        self.loras = ["Pixel Art XL"]
        self.colors = 16
        self.color_similarity = 4
        self.palette = ""

    def update(self):
        changes = False
        for key, value in self.__dict__.items():
            if key in st.session_state and st.session_state[key] != value:
                setattr(self, key, st.session_state[key])
                changes = True
        return changes


if "settings" not in st.session_state:
    st.session_state.settings = Settings()
settings = st.session_state.settings


def generate():
    path = "notebooks/test.png"
    image = Image.open(path)
    image = image_to_np(image)
    image = downscale(image)
    return image


@st.cache_data
def postprocess(image: np.array) -> np.array:
    image = generate_palette(
        image,
        max_count=256 if settings.colors <= 1 else settings.colors,
        min_distance=settings.color_similarity,
    )
    if settings.seamless:
        image = make_seamless(image, algorithm=settings.seamless_algorithm)
    if settings.palette:
        image = remap_image(image, convert_palette(settings.palette))
    image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    return image


def regenerate_images():
    for textures in st.session_state.textures:
        textures.update()


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    if len(h) == 6:
        h = h + "ff"
    return list(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4, 6))


def convert_palette(param: str):
    try:
        return np.asarray(
            [hex_to_rgb(color.strip()) for color in param.split(",") if color]
        )
    except ValueError:
        st.error("Invalid palette format!")
        return np.asarray([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])


def rgb_to_hex(rgb: List[int]) -> str:
    return "#{:02x}{:02x}{:02x}{:02x}".format(*rgb)


def colors_to_palette(rgb_list):
    hex_list = []
    for rgb in rgb_list:
        hex_list.append(rgb_to_hex(rgb))
    return ", ".join(hex_list)


def on_change():
    if settings.update():
        regenerate_images()


if "textures" not in st.session_state:
    st.session_state.textures = []

st.set_page_config(
    page_title="Pixelart Texture Generator", page_icon="🪨", layout="centered"
)

st.title(
    "Pixelart Texture Generator",
    help="Describe the texture you want, adapt the general settings on the left,"
    "and hit generate. After that, you can still modify the generation/use of palettes.",
)

st.text_area("prompt", placeholder="A mossy cobblestone wall")
st.text_input("negative prompt")

cols = st.columns(3)
with cols[0]:
    if st.button("Generate", use_container_width=True, type="primary"):
        with st.spinner("Generating ..."):
            texture = generate()
            st.session_state.textures.append(Texture(texture))
with cols[1]:
    if st.button("Remove all", use_container_width=True):
        st.session_state.textures = []


if st.session_state.textures:
    index = image_select(
        f"Generated Textures",
        images=[texture.rgb for texture in st.session_state.textures],
        return_value="index",
        use_container_width=False
    )
    texture = st.session_state.textures[index]

    data = encode_file(texture.image)

    cols = st.columns([1, 1, 1.5, 3, 3, 5])
    with cols[0]:
        if st.button("❌", help="Delete"):
            st.session_state.last_selected_texture -= 1
            st.session_state.textures.pop(index)
            st.rerun()
    with cols[1]:
        st.download_button("💾", data, "image.png", mime="image/png")
    with cols[2]:
        copy_mime_button("📋", data, "image/png")
    with cols[3]:
        if st.button("Copy Palette"):
            pyperclip.copy(colors_to_palette(texture.palette))
    with cols[4]:
        if settings.palette:
            if st.button("Clear Palette"):
                st.session_state.palette = ""
                on_change()
        else:
            if st.button(
                "Use Palette",
                help="Copy palette to the settings and apply it to all other textures.",
            ):
                st.session_state.palette = colors_to_palette(texture.palette)
                on_change()
    with cols[5]:
        st.write(f"Palette has {len(texture.palette)} colors")

    st.image(
        texture.upscaled_image,
        output_format="PNG",
        use_column_width="always",
    )


with st.sidebar:
    with st.expander("General", expanded=True):
        st.slider("Width", 16, 128, step=8, key="width", on_change=on_change)
        st.slider("Height", 16, 128, step=8, key="height", on_change=on_change)
        if st.toggle(
            "Generate seamless texture", key="seamless", on_change=on_change, value=True
        ):
            settings.seamless_algorithm = st.radio(
                "Tiling algorithm",
                ["watershed", "slic"],
                key="seamless_algorithm",
                on_change=on_change,
            )
        settings.remove_background = st.toggle(
            "Remove background", key="remove_background", on_change=on_change
        )

    with st.expander("Model", expanded=True):
        st.multiselect(
            "Models",
            list_models(),
            default=["AlbedoBase XL (SDXL)"],
            key="models",
            on_change=on_change,
        )
        st.multiselect(
            "LoRAS",
            [
                "Pixel Art XL",
                "SXZ Texture Bringer",
                "Detail Tweaker XL",
            ],
            default=["Pixel Art XL"],
            key="loras",
            on_change=on_change,
        )
        st.text_input(
            "Additional LoRAS",
            placeholder="<lora:weight>",
            key="additional_loras",
            on_change=on_change,
        )

    with st.expander("Palette", expanded=True):
        st.slider(
            "Similarity",
            min_value=0,
            max_value=32,
            step=1,
            value=10,
            help="Minimum similarity between colors in the palette, its recommended to prefer this constraint over "
            "hard limiting the colors.",
            key="color_similarity",
            on_change=on_change,
        )
        st.slider(
            "Colors",
            min_value=1,
            max_value=32,
            step=1,
            value=1,
            help="Maximum amount of colors in the palette, 1 to allow as many as needed.",
            key="colors",
            on_change=on_change,
        )
        st.text_area(
            "Palette",
            "",
            help="A comma separated list of hex colors to map the generated textures to. "
            "Palette will be generated if left empty. "
            "Copy the palette from the generated texture. ",
            key="palette",
            on_change=on_change,
        )

on_change()
