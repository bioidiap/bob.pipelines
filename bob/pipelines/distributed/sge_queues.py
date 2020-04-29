#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""This queue setup has a DEMANDING arrangement.

For CPU jobs, it prioritizes q_1day and io_big This HAS to be the
default
"""
QUEUE_DEFAULT = {
    "default": {
        "queue": "q_1day",
        "memory": "8GB",
        "io_big": True,
        "resource_spec": "",
        "max_jobs": 48,
        "resources": "",
    },
    "q_1week": {
        "queue": "q_1week",
        "memory": "4GB",
        "io_big": True,
        "resource_spec": "",
        "max_jobs": 24,
        "resources": {"q_1week": 1},
    },
    "q_short_gpu": {
        "queue": "q_short_gpu",
        "memory": "30GB",
        "io_big": False,
        "resource_spec": "",
        "max_jobs": 45,
        "resources": {"q_short_gpu": 1},
    },
    "q_gpu": {
        "queue": "q_gpu",
        "memory": "30GB",
        "io_big": False,
        "resource_spec": "",
        "resources": {"q_gpu": 1},
    },
    "q_long_gpu": {
        "queue": "q_long_gpu",
        "memory": "30GB",
        "io_big": False,
        "resource_spec": "",
        "resources": {"q_long_gpu": 1},
    },
}


"""
This queue setup has a light arrangement.
For CPU jobs, it prioritizes all.q and not io_big
"""
QUEUE_LIGHT = {
    "default": {
        "queue": "all.q",
        "memory": "4GB",
        "io_big": False,
        "resource_spec": "",
        "max_jobs": 96,
        "resources": "",
    },
    "q_1day": {
        "queue": "q_1day",
        "memory": "4GB",
        "io_big": False,
        "resource_spec": "",
        "max_jobs": 48,
        "resources": {"q_1day": 1},
    },
    "q_1week": {
        "queue": "q_1week",
        "memory": "4GB",
        "io_big": True,
        "resource_spec": "",
        "resources": {"q_1week": 1},
    },
    "q_short_gpu": {
        "queue": "q_short_gpu",
        "memory": "30GB",
        "io_big": False,
        "resource_spec": "",
        "max_jobs": 45,
        "resources": {"q_short_gpu": 1},
    },
    "q_gpu": {
        "queue": "q_gpu",
        "memory": "30GB",
        "io_big": False,
        "resource_spec": "",
        "resources": {"q_gpu": 1},
    },
    "q_long_gpu": {
        "queue": "q_long_gpu",
        "memory": "30GB",
        "io_big": False,
        "resource_spec": "",
        "resources": {"q_long_gpu": 1},
    },
}
