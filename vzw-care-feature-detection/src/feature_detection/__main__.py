#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import argparse
import sys
import json
import logging

from sys import stdin

from feature_detection import __version__
from feature_detection.arm_raising import ArmRaised
from feature_detection.source import InputSource
from feature_detection.sink import OutputSink

_logger = logging.getLogger(__name__)

def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Detect features")
    parser.add_argument(
        '--version',
        action='version',
        version='vzw-care-feature-detection {ver}'.format(ver=__version__))
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stderr, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

def main(args):
    """Runs a feature detector off of STDIN outputting features to STDOUT.
    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    reader = InputSource()
    recognizer_right = ArmRaised.make('right')
    recognizer_left = ArmRaised.make('left')
    writer = OutputSink()

    while True:
        datum = reader.next()
        if datum is InputSource.EOF:
            break
        
        if len(datum) < 1:
            continue
        datum_single = datum[0]
       
        frame_count = reader.state_counter
        output = recognizer_left.next(datum_single)
        if output is not None:
            output['frame'] = frame_count
            writer.next(output)
        
        output = recognizer_right.next(datum_single)
        if output is not None:
            output['frame'] = frame_count
            writer.next(output)

def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
