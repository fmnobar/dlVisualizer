#!/usr/bin/env python3

from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


SIZE = 1024
PANEL_MARGIN = 56
PANEL_RADIUS = 230

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "Assets"
CONCEPTS_DIR = ASSETS_DIR / "IconConcepts"
ICONSET_DIR = ASSETS_DIR / "AppIcon.iconset"
ICNS_PATH = ASSETS_DIR / "AppIcon.icns"


def rgba(hex_value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[index:index + 2], 16) for index in (0, 2, 4)) + (alpha,)


def make_canvas() -> tuple[Image.Image, tuple[int, int, int, int]]:
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    panel_bounds = (
        PANEL_MARGIN,
        PANEL_MARGIN,
        SIZE - PANEL_MARGIN,
        SIZE - PANEL_MARGIN,
    )
    return canvas, panel_bounds


def add_shadow(canvas: Image.Image, panel_bounds: tuple[int, int, int, int]) -> None:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (
            panel_bounds[0],
            panel_bounds[1] + 18,
            panel_bounds[2],
            panel_bounds[3] + 18,
        ),
        radius=PANEL_RADIUS,
        fill=(18, 31, 58, 36),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(24))
    canvas.alpha_composite(shadow)


def fill_panel_gradient(
    canvas: Image.Image,
    panel_bounds: tuple[int, int, int, int],
    top: tuple[int, int, int, int],
    bottom: tuple[int, int, int, int],
    accent: tuple[int, int, int, int],
) -> None:
    width = panel_bounds[2] - panel_bounds[0]
    height = panel_bounds[3] - panel_bounds[1]
    panel = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = panel.load()

    for y in range(height):
        t = y / max(height - 1, 1)
        base = tuple(int(top[index] * (1 - t) + bottom[index] * t) for index in range(4))
        for x in range(width):
            dx = (x - (width * 0.76)) / width
            dy = (y - (height * 0.18)) / height
            glow = max(0.0, 1.0 - math.sqrt((dx * dx * 1.8) + (dy * dy * 4.0)) * 3.0)
            pixels[x, y] = tuple(
                min(255, int(base[index] + accent[index] * glow * 0.38))
                for index in range(4)
            )

    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width, height), radius=PANEL_RADIUS, fill=255)
    panel.putalpha(mask)
    canvas.alpha_composite(panel, dest=(panel_bounds[0], panel_bounds[1]))


def add_glow(
    canvas: Image.Image,
    position: tuple[float, float],
    radius: int,
    color: tuple[int, int, int, int],
) -> None:
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    x, y = position
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(radius // 2))
    canvas.alpha_composite(glow)


def draw_node(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: int, fill_color: tuple[int, int, int, int]) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill_color)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 255, 255, 160), width=2)


def draw_curve(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.line(points, fill=color, width=width, joint="curve")


def concept_network_curve() -> Image.Image:
    canvas, panel_bounds = make_canvas()
    add_shadow(canvas, panel_bounds)
    fill_panel_gradient(
        canvas,
        panel_bounds,
        rgba("#F8FBFF"),
        rgba("#E7EEFF"),
        rgba("#B5D7FF"),
    )

    add_glow(canvas, (780, 250), 190, rgba("#7BC7FF", 78))
    add_glow(canvas, (270, 760), 180, rgba("#FF9D84", 56))

    draw = ImageDraw.Draw(canvas)
    input_nodes = [(250, 320), (250, 510), (250, 700)]
    hidden_nodes = [(430, 270), (430, 450), (430, 630), (430, 810)]
    output_nodes = [(605, 530)]

    for from_node in input_nodes:
        for to_node in hidden_nodes:
            draw.line((from_node, to_node), fill=rgba("#7CA7E5", 92), width=9)
    for from_node in hidden_nodes:
        draw.line((from_node, output_nodes[0]), fill=rgba("#FF8F7B", 86), width=8)

    for center in input_nodes:
        draw_node(draw, center, 22, rgba("#4D8FFF"))
    for center in hidden_nodes:
        draw_node(draw, center, 18, rgba("#87B7FF"))
    for center in output_nodes:
        draw_node(draw, center, 24, rgba("#FF7F66"))

    curve_points: list[tuple[float, float]] = []
    for index in range(180):
        t = index / 179.0
        x = 610 + t * 230
        hump = math.exp(-((t - 0.34) ** 2) / 0.07)
        y = 700 - (hump * 250) - (t * 30)
        curve_points.append((x, y))

    draw_curve(draw, curve_points, rgba("#157CF0"), 17)

    rng = random.Random(19)
    for t in [0.08, 0.15, 0.24, 0.32, 0.41, 0.53, 0.65, 0.77, 0.88]:
        x = 610 + t * 230 + rng.uniform(-6, 6)
        hump = math.exp(-((t - 0.34) ** 2) / 0.07)
        y = 700 - (hump * 250) - (t * 30) + rng.uniform(-18, 18)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=rgba("#FF655E"))

    return canvas


def concept_heatmap_curve() -> Image.Image:
    canvas, panel_bounds = make_canvas()
    add_shadow(canvas, panel_bounds)
    fill_panel_gradient(
        canvas,
        panel_bounds,
        rgba("#FFF9F3"),
        rgba("#FFECDC"),
        rgba("#FFC18A"),
    )

    add_glow(canvas, (330, 300), 210, rgba("#FFA86B", 70))
    add_glow(canvas, (740, 780), 180, rgba("#86B6FF", 64))
    draw = ImageDraw.Draw(canvas)

    start_x = 180
    start_y = 220
    cell = 112
    gap = 18
    values = [
        [0.15, 0.35, 0.55, 0.8],
        [0.22, 0.45, 0.7, 0.92],
        [0.3, 0.58, 0.82, 0.95],
        [0.18, 0.4, 0.63, 0.78],
    ]
    for row, row_values in enumerate(values):
        for column, value in enumerate(row_values):
            x0 = start_x + column * (cell + gap)
            y0 = start_y + row * (cell + gap)
            blend = (
                int(72 + (180 * value)),
                int(110 + (70 * (1 - value))),
                int(220 - (90 * value)),
                230,
            )
            draw.rounded_rectangle((x0, y0, x0 + cell, y0 + cell), radius=26, fill=blend)

    axis_color = rgba("#3F556E", 90)
    draw.line((170, 835, 870, 835), fill=axis_color, width=5)
    draw.line((170, 170, 170, 835), fill=axis_color, width=5)

    curve_points: list[tuple[float, float]] = []
    for index in range(170):
        t = index / 169.0
        x = 230 + t * 560
        y = 710 - (math.sin((t * 2.1) + 0.2) * 165) - (t * 70)
        curve_points.append((x, y))
    draw_curve(draw, curve_points, rgba("#155EEA"), 18)

    for index in range(7):
        t = 0.08 + index * 0.13
        x = 230 + t * 560
        y = 710 - (math.sin((t * 2.1) + 0.2) * 165) - (t * 70)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=rgba("#FF704D"))

    return canvas


def concept_playground_graph() -> Image.Image:
    canvas, panel_bounds = make_canvas()
    add_shadow(canvas, panel_bounds)
    fill_panel_gradient(
        canvas,
        panel_bounds,
        rgba("#F7FAFF"),
        rgba("#ECF1FA"),
        rgba("#B7CBFF"),
    )

    add_glow(canvas, (790, 220), 170, rgba("#FF9D8B", 58))
    add_glow(canvas, (260, 710), 180, rgba("#7EC7FF", 64))

    draw = ImageDraw.Draw(canvas)
    columns = [
        [(210, 300), (210, 500), (210, 700)],
        [(400, 250), (400, 400), (400, 550), (400, 700)],
        [(610, 330), (610, 530), (610, 730)],
        [(805, 510)],
    ]

    line_palette = [rgba("#5D8FFF", 90), rgba("#FF8A6B", 88)]
    for layer_index in range(len(columns) - 1):
        for source_index, source in enumerate(columns[layer_index]):
            for target_index, target in enumerate(columns[layer_index + 1]):
                color = line_palette[(source_index + target_index + layer_index) % len(line_palette)]
                draw.line((source, target), fill=color, width=7)

    node_colors = [rgba("#2E7BFF"), rgba("#8CB7FF"), rgba("#6AA2FF"), rgba("#FF7B63")]
    node_sizes = [24, 20, 22, 28]
    for layer_index, layer in enumerate(columns):
        for center in layer:
            draw_node(draw, center, node_sizes[layer_index], node_colors[layer_index])

    curve_points: list[tuple[float, float]] = []
    for index in range(120):
        t = index / 119.0
        x = 715 + t * 150
        y = 670 - (math.exp(-((t - 0.22) ** 2) / 0.02) * 80) + (t * 18)
        curve_points.append((x, y))
    draw_curve(draw, curve_points, rgba("#0E73ED"), 15)

    return canvas


def export_iconset(base_image: Image.Image) -> None:
    ICONSET_DIR.mkdir(parents=True, exist_ok=True)
    sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    for file_name, size in sizes.items():
        resized = base_image.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(ICONSET_DIR / file_name)

    subprocess.run(
        ["iconutil", "-c", "icns", str(ICONSET_DIR), "-o", str(ICNS_PATH)],
        check=True,
    )


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)

    concepts = {
        "dl-visualizer-icon-network-curve.png": concept_network_curve(),
        "dl-visualizer-icon-heatmap-curve.png": concept_heatmap_curve(),
        "dl-visualizer-icon-playground-graph.png": concept_playground_graph(),
    }

    for file_name, image in concepts.items():
        image.save(CONCEPTS_DIR / file_name)

    export_iconset(concepts["dl-visualizer-icon-network-curve.png"])


if __name__ == "__main__":
    main()
