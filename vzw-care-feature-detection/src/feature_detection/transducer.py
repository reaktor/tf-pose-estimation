# -*- coding: utf-8 -*-
"""Recognizers for features.
"""
import logging

from enum import Enum

class Transducer:
    class State(Enum):
        Running, Done = range(2)

    def next():
        return NotImplemented
