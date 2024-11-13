import cv2
import pytest
from image_utils import to_grayscale, apply_gaussian_blur, detect_edges, find_contours

# test to check everything is working

def test_to_grayscale():
    image = cv2.imread('data/input/sample_scan.png')
    gray_image = to_grayscale(image)
    assert gray_image is not None, "Grayscale conversion returned None"
    assert len(gray_image.shape) == 2, "Image should be in grayscale (single channel)"

def test_apply_gaussian_blur():
    image = cv2.imread('data/input/sample_scan.png')
    gray_image = to_grayscale(image)
    blurred_image = apply_gaussian_blur(gray_image)
    assert blurred_image is not None, "Gaussian blur returned None"
    assert blurred_image.shape == gray_image.shape, "Blurred image should have the same shape as the input grayscale image"

def test_detect_edges():
    image = cv2.imread('data/input/sample_scan.png')
    gray_image = to_grayscale(image)
    blurred_image = apply_gaussian_blur(gray_image)
    edge_image = detect_edges(blurred_image)
    assert edge_image is not None, "Edge detection returned None"
    assert len(edge_image.shape) == 2, "Edge-detected image should be a single channel (grayscale-like)"

def test_find_contours():
    image = cv2.imread('data/input/sample_scan.png')
    gray_image = to_grayscale(image)
    blurred_image = apply_gaussian_blur(gray_image)
    edge_image = detect_edges(blurred_image)
    contours = find_contours(edge_image)
    assert contours, "No contours found, card detection failed"
    assert all(len(c) > 0 for c in contours), "Some contours found have no points"
