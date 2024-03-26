import io

import PIL
import numpy as np
from PIL.Image import Image
from scipy.spatial.distance import pdist, squareform, cdist
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import watershed
from skimage.util import view_as_blocks
from sklearn.cluster import KMeans


def image_to_np(image: Image) -> np.ndarray:
    return np.array(image, dtype=np.float32) / 255.0


def downscale(image: np.ndarray, factor: int = 8) -> np.ndarray:
    # Make sure it has exactly 3 channels
    image = image.squeeze()
    if len(image.shape) == 2:
        image = np.stack([image] * 4, axis=-1)
        image[:, :, 3] = 1.0
    elif image.shape[2] == 1:
        image = np.stack([image[:, :, 0]] * 4, axis=-1)
        image[:, :, 3] = 1.0
    elif image.shape[2] == 3:
        image = np.concatenate([image, np.ones_like(image[:, :, :1])], axis=-1)

    # Downscale using median
    blocks = view_as_blocks(image, (factor, factor, 4)).squeeze()
    image = blocks.mean(axis=(2, 3))

    return image


def merge_clusters(
    clusters: np.array, min_distance: float, max_count: int = 256
) -> np.array:
    # Compute pairwise distances
    distances = squareform(pdist(clusters))
    np.fill_diagonal(distances, np.inf)

    # Iteratively merge the two closest clusters
    while True:
        a, b = np.unravel_index(np.argmin(distances), distances.shape)
        if distances[a, b] < min_distance or clusters.shape[0] > max_count:
            clusters[a, :] = (clusters[a, :] + clusters[b, :]) / 2
            clusters = np.delete(clusters, b, axis=0)
            distances = np.delete(distances, b, axis=0)
            distances = np.delete(distances, b, axis=1)
        else:
            return clusters


def generate_palette(
    image: np.ndarray,
    min_distance: float = 4.0,
    max_count: int = 256,
    initial_count: int = 256,
) -> np.ndarray:
    """
    Performs one KMeans clustering step on the image in CIE Lab color space to generate a possible palette,
    then performs a second step with a subset of colors constrained by the minimum distance.
    """
    alpha = image[:, :, 3].reshape(-1, 1)
    color = rgb2lab(image[:, :, :3]).reshape((-1, 3))
    data = np.concatenate([color, alpha], axis=1)

    # Find initial clusters
    k = KMeans(initial_count, random_state=42)
    k.fit(data)
    centers = k.cluster_centers_

    # Merge clusters that are too close
    centers = merge_clusters(centers, min_distance, max_count)

    # Find final clusters
    clusters = KMeans(centers.shape[0], init=centers, random_state=42).fit_predict(data)
    clusters = clusters.reshape(image.shape[:2])

    new_image = np.zeros_like(image)
    for i in range(max_count):
        mask = clusters == i
        if mask.sum() > 0:
            new_color = np.median(image[mask], axis=0)
            new_image[mask] = new_color

    return new_image


def make_seamless(
    image: np.ndarray,
    algorithm: str = "watershed",
    axis: int = None,
    blend: float = 0.5,
    compactness: float = 5.0,
    n_segments: int = 500,
    dither_mask: bool = True,
) -> np.ndarray:
    if axis is None:
        for axis in [0, 1]:
            image = make_seamless(
                image, algorithm, axis, blend, compactness, n_segments
            )
        return image

    # Vertical column
    gradient = np.linspace(np.zeros(128), np.ones(128), 128)
    gradient = np.absolute(gradient - 0.5) * 2.0

    if axis == 1:
        gradient = gradient.T

    if algorithm == "watershed":
        # Sobel edge detection to get cleaner cuts
        edges = np.array(sobel(rgb2gray(image[:, :, :3])))
        edges = np.minimum(1, edges / edges.mean() * 0.5)

        # The depth-mask is now the center-focused gradient blended with edges
        mask = gradient * (1.0 - blend) + edges * blend

        # Use watershed to extract clusters of connected pixels
        clusters = watershed(mask)
    elif algorithm == "slic":
        # Slic is a color and position sensitive clustering algorithm which works better in some scenarios
        clusters = slic(image, compactness=compactness, n_segments=n_segments)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Separate clusters into two groups
    mask = np.zeros_like(clusters)
    for i in range(clusters.max() + 1):
        m = clusters == i
        if m.sum() > 0 and gradient[m].mean() < 0.5:
            mask[m] = 1

    # Dither the mask's border, which works nicely for pixel art
    if dither_mask:
        dither = np.diff(mask, axis=axis, append=0) != 0
        dither[:, -1] = 0
        dither[-1, :] = 0
        dither[::2, ::2] = 0
        dither[1::2, 1::2] = 0
        mask[dither] = 0
        mask[dither] = 1

    # Blend the two halves
    image_shifted = np.roll(image, image.shape[0] // 2, axis=axis)
    image = image_shifted * mask[:, :, None] + image * (1.0 - mask[:, :, None])

    return image


def get_color_palette(image: np.array) -> np.array:
    return np.unique(image.reshape(-1, 4), axis=0)


def match_colors(
    source_colors: np.array, target_colors: np.array, prefer_unique: bool
) -> np.array:
    """
    Returns the mapping from source color to the closest target color.
    If target < source, the mapping will be many-to-one, optionally preferring unique colors.
    """
    distances = cdist(source_colors, target_colors)
    mapping = np.zeros((source_colors.shape[0],), dtype=np.int32)

    for _ in range(source_colors.shape[0]):
        i, j = np.unravel_index(np.argmin(distances), distances.shape)

        if prefer_unique:
            distances[:, j] += 1

        distances[i, :] = np.inf
        mapping[i] = j

    return mapping


def remap_image(
    image: np.array, target_palette: np.array, prefer_unique: bool = True
) -> np.array:
    source_palette = get_color_palette(image)
    mapping = match_colors(source_palette, target_palette, prefer_unique)
    remapped_image = np.zeros_like(image)

    for i in range(len(mapping)):
        c = source_palette[i]
        mask = (
            (image[:, :, 0] == c[0])
            & (image[:, :, 1] == c[1])
            & (image[:, :, 2] == c[2])
            & (image[:, :, 3] == c[3])
        )
        remapped_image[mask] = target_palette[mapping[i]]

    return remapped_image


def encode_file(image: np.array) -> bytes:
    pil_img = PIL.Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()
