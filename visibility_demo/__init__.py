"""Façade exposure from vehicle trajectories.

Modules, in the order the method uses them:
    geo        local metric projection, bearings
    buildings  outlines -> densified façade points
    shapes     viewing areas and point-in-shape tests
    exposure   BallTree index and exposure counting
    synthetic  demo trips on the road network
    data       loaders for data/
    plotting   shared figure style
"""
SAMPLE_INTERVAL_S = 5.0   # one exposure count = 5 s in view
