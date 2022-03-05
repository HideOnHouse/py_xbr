import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

WGT1 = 0.129633
WGT2 = 0.175068
W1 = (-WGT1)
W2 = (WGT1 + 0.5)
W3 = (-WGT2)
W4 = (WGT2 + 0.5)


def clamp(x, low, high):
    return max(min(x, high), low)


def df(a, b):
    return abs(a - b)


def diagonal_edge(mat: np.ndarray, wp):
    dw1 = (
            wp[0] * (df(mat[0, 2], mat[1, 1]) + df(mat[1, 1], mat[2, 0]) +
                     df(mat[1, 3], mat[2, 2]) + df(mat[2, 2], mat[3, 1])) +
            wp[1] * (df(mat[0, 3], mat[1, 2]) + df(mat[2, 1], mat[3, 0])) +
            wp[2] * (df(mat[0, 3], mat[2, 1]) + df(mat[1, 2], mat[3, 0])) +
            wp[3] * df(mat[1, 2], mat[2, 1]) +
            wp[4] * (df(mat[0, 2], mat[2, 0]) + df(mat[1, 3], mat[3, 1])) +
            wp[5] * (df(mat[0, 1], mat[1, 0]) + df(mat[2, 3], mat[3, 2]))
    )

    dw2 = (
            wp[0] * (df(mat[0, 1], mat[1, 2]) + df(mat[1, 2], mat[2, 3]) +
                     df(mat[1, 0], mat[2, 1]) + df(mat[2, 1], mat[3, 2])) +
            wp[1] * (df(mat[0, 0], mat[1, 1]) + df(mat[2, 2], mat[3, 3])) +
            wp[2] * (df(mat[0, 0], mat[2, 2]) + df(mat[1, 1], mat[3, 3])) +
            wp[3] * df(mat[1, 1], mat[2, 2]) +
            wp[4] * (df(mat[1, 0], mat[3, 2]) + df(mat[0, 1], mat[2, 3])) +
            wp[5] * (df(mat[0, 2], mat[1, 3]) + df(mat[2, 0], mat[3, 1]))
    )

    return dw1 - dw2


def generate_patch(luma, wp, rgba: np.ndarray) -> np.ndarray:
    d_edge = diagonal_edge(luma, wp)
    # generate and write result
    if d_edge <= 0:
        patch = W1 * (rgba[0, 3] + rgba[3, 0]) + W2 * (rgba[1, 2] + rgba[2, 1])
    else:
        patch = W1 * (rgba[0, 0] + rgba[3, 3]) + W2 * (rgba[1, 1] + rgba[2, 2])
    # anti-ringing, clamp
    low, high = np.min(rgba, axis=(0, 1)), np.max(rgba, axis=(0, 1))
    patch = np.clip(patch, low, high)
    patch = np.clip(np.ceil(patch), 0, 255)
    return patch


def super_xbr(data: np.ndarray) -> np.ndarray:
    factor = 2
    h, w, c = data.shape
    out_h, out_w = h * factor, w * factor
    out = np.empty((out_h, out_w, c), dtype=np.uint8)
    rgba = np.empty((4, 4, c))
    luma = np.empty((4, 4))

    # first pass
    wp = (2, 1, -1, 4, -1, 1)
    for y in range(0, out_h, 2):
        for x in range(0, out_w, 2):
            cx, cy = x // factor, y // factor  # central pixels on original images
            # sample supporting pixels in original image
            for sx in range(-1, 3):
                for sy in range(-1, 3):
                    csy = int(clamp(cy + sy, 0, h - 1))
                    csx = int(clamp(cx + sx, 0, w - 1))

                    # sample & add weighted components
                    sample = data[csy][csx]
                    rgba[sx + 1][sy + 1] = sample
                    luma[sx + 1][sy + 1] = 0.2126 * sample[0] + 0.7152 * sample[1] + 0.0722 * sample[2]

            out[y, x] = out[y, x + 1] = out[y + 1, x] = data[cy][cx]
            patch = generate_patch(luma, wp, rgba)
            out[y + 1, x + 1] = patch

    # second pass
    wp = (2, 0, 0, 0, 0, 0)
    for y in range(0, out_h, 2):
        for x in range(0, out_w, 2):
            # sample supporting pixels in original image
            for sx in range(-1, 3):
                for sy in range(-1, 3):
                    # clamp pixel locations
                    csy = int(clamp(sx - sy + y, 0, factor * h - 1))
                    csx = int(clamp(sx + sy + x, 0, factor * w - 1))

                    # sample & add weighted components
                    sample = out[csy][csx]
                    rgba[sx + 1][sy + 1] = sample
                    luma[sx + 1][sy + 1] = 0.2126 * sample[0] + 0.7152 * sample[1] + 0.0722 * sample[2]

            patch = generate_patch(luma, wp, rgba)
            out[y, x + 1] = patch

            for sx in range(-1, 3):
                for sy in range(-1, 3):
                    # clamp pixel locations
                    csy = int(clamp(sx - sy + 1 + y, 0, factor * h - 1))
                    csx = int(clamp(sx + sy - 1 + x, 0, factor * w - 1))

                    # sample and add weighted components
                    sample = out[csy][csx]
                    rgba[sx + 1][sy + 1] = sample
                    luma[sx + 1][sy + 1] = 0.2126 * sample[0] + 0.7152 * sample[1] + 0.0722 * sample[2]

            patch = generate_patch(luma, wp, rgba)
            out[y + 1][x] = patch

    # third pass
    wp = (2, 1, -1, 4, -1, 1)
    for y in range(out_h - 1, -1, -1):
        for x in range(out_w - 1, -1, -1):
            for sx in range(-2, 2):
                for sy in range(-2, 2):
                    # clamp pixel locations
                    csy = int(clamp(sy + y, 0, factor * h - 1))
                    csx = int(clamp(sx + x, 0, factor * w - 1))

                    # sample and add weighted components
                    sample = out[csy, csx]
                    rgba[sx + 2][sy + 2] = sample
                    luma[sx + 2][sy + 2] = 0.2126 * sample[0] + 0.7152 * sample[1] + 0.0722 * sample[2]

            patch = generate_patch(luma, wp, rgba)
            out[y][x] = patch
    return out


def main(args):
    if len(args) != 3:
        print("Usage:\n python super_xbr.pt [image path or directory] [output directory]")
        return 0

    if os.path.isdir(args[1]):
        image_list = [os.path.join(args[1], i) for i in os.listdir(args[1])]
    elif os.path.isfile(args[1]):
        image_list = [args[1]]
    else:
        print("Invalid path or directory")
        return 1

    if not os.path.exists(args[2]):
        os.makedirs(args[2])

    for image_path in tqdm(image_list):
        img = Image.open(image_path)
        data = np.asarray(img)
        processed = super_xbr(data)
        processed = Image.fromarray(processed)
        processed.save(os.path.join(args[2], os.path.basename(image_path)))


if __name__ == '__main__':
    main(sys.argv)
