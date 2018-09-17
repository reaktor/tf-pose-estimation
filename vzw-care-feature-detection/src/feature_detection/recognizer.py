# -*- coding: utf-8 -*-
"""Recognizers for features.
"""
import logging

from enum import Enum

class Recognizer:
    """Construct a recognizer with a given structure.
        Args:
            transitions:
                dict-like structure mapping states to functions of the signature
                    
                    def t(mutable_context, state, input_data):
                        --> state, output_data

                chosen for the given state.

            initial_state:
                initial state for state structure. Does not have to be in
                transitions, but should be unless trivial.

            context_init:
                function to initialize context

            states:
                explicit set of states for constraining behavior.
    """
    def __init__(
        self, transitions=None, initial_state=None, context_init=None, states=None,
        params=None, kparams=None
    ):
        params = params or ()
        kparams = kparams or {}
        self.state = initial_state
        self.context = dict() if context_init is None else context_init(*params, **kparams)

        # TODO: Check transitions for sane structure
        self.transitions = transitions

        # TODO: Handle explicit states

    def next(self, in_record):
        logging.debug("Processing:")
        #logging.debug(" -- input %.100s", str(in_record))
        in_state = self.state
        logging.debug(" -- current state (%s)", in_state)
        #logging.debug(" -- context state %s", str(self.context))
      
        if self.transitions is None:
            out_state, out_record = in_state, in_record

        elif in_state in self.transitions:
            t = self.transitions[in_state]
            result = t(self.context, in_state, in_record)
            if result is None:
                out_state, out_record = in_state, None
            else:
                out_state, out_record = result

        else:
            logging.debug(" -- reached terminal state.")
            return None

        logging.debug(" -- new state (%s)", out_state)
        #logging.debug(" -- outputting %.100s", str(out_record))
        self.state = out_state
        logging.info(">> State >> %s", out_state)
        return out_record
