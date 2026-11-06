"""Local metric coordinates and headings.

An equirectangular projection around a reference point is accurate to well under a metre over a few kilometres.
Bearings are degrees clockwise from north (0 = north, 90 = east).
"""
import math

import numpy as np

M_PER_DEG = 111_320.0


class LocalProjection:
    """lon/lat <-> metres east/north of (lat0, lon0)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.kx = M_PER_DEG * math.cos(math.radians(self.lat0))

    def to_xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * M_PER_DEG

    def to_lonlat(self, x, y):
        return self.lon0 + np.asarray(x) / self.kx, self.lat0 + np.asarray(y) / M_PER_DEG


def bearing_deg(dx, dy):
    """Heading of a displacement, in [0, 360)."""
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)


def ahead(x, y, bearing, distance):
    """Point `distance` metres ahead along `bearing`."""
    th = np.radians(bearing)
    return x + distance * np.sin(th), y + distance * np.cos(th)
