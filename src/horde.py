import os
from functools import cache
from typing import List

import requests
from PIL.Image import Image
from dotenv import load_dotenv
from horde_sdk.ai_horde_api import KNOWN_SAMPLERS
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPISimpleClient
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    ImageGenerationInputPayload,
    LorasPayloadEntry,
)

load_dotenv()

API_KEY = os.getenv("HORDE_API_KEY")
simple_client = AIHordeAPISimpleClient()


def generate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    n: int = 1,
    steps: int = 30,
    models=None,
    loras: List[str] | None = None,
) -> List[Image]:
    if models is None:
        models = ["AlbedoBase XL (SDXL)"]

    status_response, job_id = simple_client.image_generate_request(
        ImageGenerateAsyncRequest(
            apikey=API_KEY,
            params=ImageGenerationInputPayload(
                sampler_name=KNOWN_SAMPLERS.k_euler,
                width=width,
                height=height,
                steps=steps,
                use_nsfw_censor=False,
                n=n,
                loras=[LorasPayloadEntry(name=n) for n in loras],
            ),
            prompt=prompt,
            models=models,
        ),
    )

    if len(status_response.generations) == 0:
        raise Exception("No generations found")

    return [g.image for g in status_response.generations if not g.censored]


@cache
def list_models() -> List[str]:
    url = "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-image-model-reference/main/stable_diffusion.json"
    response = requests.get(url).json()
    filtered_models = []
    for name, model in response.items():
        if model["baseline"] == "stable_diffusion_xl":
            filtered_models.append(name)
    return filtered_models
