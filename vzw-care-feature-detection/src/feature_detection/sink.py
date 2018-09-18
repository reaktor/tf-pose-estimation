# -*- coding: utf-8 -*-
"""Recognizers for features.
"""
import logging

import json
from sys import stdout
from enum import Enum

from feature_detection.transducer import Transducer

class OutputSink(Transducer):
    """Construct an input source from a file stream or STDOUT. 
        Args:
            filename: filename or None for STDOUT
    """
    EOF = []

    def __init__(self, filename=None):
        self.file = stdout if filename is None else open(filename, 'w')
        self.state_counter = 0
        self.condition = self.State.Running

        self.filename = "STDOUT" if filename is None else filename
        logging.info("Writing to %s", self.filename)

    def is_done(self):
        return self.condition == self.State.Done

    def next(self, in_record):
        logging.debug("Writing to %s", self.filename)
        line = json.dumps(in_record)
        self.file.write(line + "\n")
        self.file.flush()

        self.state_counter += 1
        return None
