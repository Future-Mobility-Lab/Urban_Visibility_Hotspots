"""Forward viewing areas and vectorised point-in-shape tests.

The driver's view is a 60° sector reaching 100 m ahead. Shapes compared:
    circle    r = 50 m, centred 50 m ahead (touches the vehicle, reaches 100 m)   one distance test
    sector    r = 100 m, 60°, apex at the vehicle                                  distance + angle
    triangle  apex at the vehicle, base 100 m ahead and 115.5 m wide              three sign tests
Points (px, py) broadcast against vehicle states (x, y, bearing).
"""
import math

import numpy as np

RANGE_M = 100.0
OPENING_DEG = 60.0
CIRCLE_R = CIRCLE_OFFSET = RANGE_M / 2
TRI_BASE = 2 * RANGE_M * math.tan(math.radians(OPENING_DEG / 2))
SHAPES = ("circle", "sector", "triangle")


def circle_centre(x, y, bearing):
    th = np.radians(bearing)
    return x + CIRCLE_OFFSET * np.sin(th), y + CIRCLE_OFFSET * np.cos(th)


def in_circle(px, py, x, y, bearing):
    cx, cy = circle_centre(x, y, bearing)
    return (px - cx) ** 2 + (py - cy) ** 2 <= CIRCLE_R ** 2


def in_sector(px, py, x, y, bearing):
    dx, dy = px - x, py - y
    off_axis = np.abs((np.degrees(np.arctan2(dx, dy)) - bearing + 180) % 360 - 180)
    return (dx * dx + dy * dy <= RANGE_M ** 2) & (off_axis <= OPENING_DEG / 2)


def triangle_vertices(x, y, bearing):
    th = np.radians(bearing)
    fx, fy, rx, ry = np.sin(th), np.cos(th), np.cos(th), -np.sin(th)       # forward and right unit vectors
    bx, by = x + RANGE_M * fx, y + RANGE_M * fy
    h = TRI_BASE / 2
    return (x, y), (bx - h * rx, by - h * ry), (bx + h * rx, by + h * ry)


def in_triangle(px, py, x, y, bearing):
    verts = triangle_vertices(x, y, bearing)
    sides = [(bx - ax) * (py - ay) - (by - ay) * (px - ax)                  # cross product per edge
             for (ax, ay), (bx, by) in zip(verts, verts[1:] + verts[:1])]
    return np.all([s >= 0 for s in sides], axis=0) | np.all([s <= 0 for s in sides], axis=0)


TESTS = {"circle": in_circle, "sector": in_sector, "triangle": in_triangle}


def query_disc(shape, x, y, bearing):
    """Centre and radius of the smallest disc around which a radius query covers the shape.

    For the circle the disc is the shape itself; sector and triangle need a disc around the vehicle
    (4x and 5.3x the circle's area) plus the exact test.
    """
    if shape == "circle":
        return (*circle_centre(x, y, bearing), CIRCLE_R)
    return x, y, RANGE_M if shape == "sector" else math.hypot(RANGE_M, TRI_BASE / 2)
