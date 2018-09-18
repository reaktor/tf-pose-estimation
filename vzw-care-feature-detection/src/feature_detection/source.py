# -*- coding: utf-8 -*-
"""Recognizers for features.
"""
import logging

import json
from sys import stdin
from enum import Enum

from feature_detection.transducer import Transducer

class InputSource(Transducer):
    """Construct an input source from a file stream or STDIN. 
        Args:
            filename: filename or None for STDIN
    """
    EOF = []

    def __init__(self, filename=None):
        self.file = stdin if filename is None else open(filename, 'r')
        self.reader = (line for line in self.file)
        self.state_counter = 0
        self.condition = self.State.Running

        self.filename = "STDIN" if filename is None else filename
        logging.info("Reading from %s", self.filename)

    def is_done(self):
        return self.condition == self.State.Done

    def next(self):
        logging.debug("Reading from %s", self.filename)
        line = next(self.reader, self.EOF)
        if line is self.EOF:
            logging.debug(" -- reached EOF.")
            self.condition = self.State.Done
            return self.EOF

        datum = json.loads(line)
        self.state_counter += 1
        return datum
