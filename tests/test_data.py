import numpy as np

from inspectline import generate_shape_image, generate_dataset, SHAPE_NAMES


def test_generate_shape_image_returns_consistent_boxes_and_labels():
    img, boxes, labels = generate_shape_image(size=160, seed=1)
    assert img.size == (160, 160)
    assert boxes.shape[0] == labels.shape[0]
    assert boxes.shape[1] == 4  # x1, y1, x2, y2
    assert all(0 <= label < len(SHAPE_NAMES) for label in labels)


def test_boxes_are_within_image_bounds():
    img, boxes, _ = generate_shape_image(size=160, seed=2)
    assert np.all(boxes[:, 0] >= 0) and np.all(boxes[:, 2] <= 160)
    assert np.all(boxes[:, 1] >= 0) and np.all(boxes[:, 3] <= 160)


def test_boxes_have_positive_area():
    _, boxes, _ = generate_shape_image(size=160, seed=3)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    assert np.all(widths > 0) and np.all(heights > 0)


def test_generate_shape_image_is_reproducible():
    img1, boxes1, labels1 = generate_shape_image(size=160, seed=42)
    img2, boxes2, labels2 = generate_shape_image(size=160, seed=42)
    np.testing.assert_array_equal(boxes1, boxes2)
    np.testing.assert_array_equal(labels1, labels2)


def test_generate_dataset_returns_requested_count():
    dataset = generate_dataset(10, size=160, seed=0)
    assert len(dataset) == 10
