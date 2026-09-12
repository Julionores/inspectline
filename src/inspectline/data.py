"""Generateur d'images synthetiques pour un controle qualite industriel :
plusieurs pieces geometriques (cercle, carre, triangle, etoile) disposees sur
un tapis de convoyeur bruite, avec leurs boites englobantes et labels."""

import numpy as np
from PIL import Image, ImageDraw

SHAPE_NAMES = ["cercle", "carre", "triangle", "etoile"]

COLORS = [
    (220, 60, 60),
    (60, 140, 220),
    (60, 200, 100),
    (230, 180, 40),
    (170, 90, 200),
    (240, 130, 40),
]


def _draw_star(draw, cx, cy, r, color):
    points = []
    for i in range(10):
        angle = np.pi / 2 + i * np.pi / 5
        radius = r if i % 2 == 0 else r * 0.45
        points.append((cx + radius * np.cos(angle), cy - radius * np.sin(angle)))
    draw.polygon(points, fill=color)


def generate_shape_image(size=160, n_shapes_range=(2, 4), seed=None):
    """Genere une image (PIL) avec plusieurs formes, et retourne
    (image, boxes [x1,y1,x2,y2], labels [0..3])."""
    rng = np.random.default_rng(seed)
    bg = tuple(int(v) for v in rng.integers(190, 230, size=3))
    img = Image.new("RGB", (size, size), color=bg)
    draw = ImageDraw.Draw(img)

    # bruit de fond (tapis de convoyeur texture)
    for _ in range(40):
        x, y = rng.integers(0, size, 2)
        shade = int(rng.integers(-15, 15))
        c = tuple(max(0, min(255, v + shade)) for v in bg)
        draw.ellipse([x, y, x + 2, y + 2], fill=c)

    n_shapes = rng.integers(*n_shapes_range)
    boxes, labels = [], []
    placed = []

    attempts = 0
    while len(boxes) < n_shapes and attempts < n_shapes * 20:
        attempts += 1
        r = rng.integers(size // 10, size // 6)
        cx = rng.integers(r + 2, size - r - 2)
        cy = rng.integers(r + 2, size - r - 2)

        overlap = any(
            (cx - ocx) ** 2 + (cy - ocy) ** 2 < (r + orad) ** 2
            for ocx, ocy, orad in placed
        )
        if overlap:
            continue
        placed.append((cx, cy, r))

        shape_idx = rng.integers(0, 4)
        color = tuple(int(c) for c in COLORS[rng.integers(0, len(COLORS))])

        if shape_idx == 0:  # cercle
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        elif shape_idx == 1:  # carre
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
        elif shape_idx == 2:  # triangle
            pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
            draw.polygon(pts, fill=color)
        else:  # etoile
            _draw_star(draw, cx, cy, r, color)

        boxes.append([cx - r, cy - r, cx + r, cy + r])
        labels.append(shape_idx)

    return img, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def generate_dataset(n_images, size=160, seed=0):
    """Genere n_images (image, boxes, labels), de facon reproductible."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, n_images)
    return [generate_shape_image(size=size, seed=int(s)) for s in seeds]
