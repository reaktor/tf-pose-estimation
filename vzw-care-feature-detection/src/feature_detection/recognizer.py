# -*- coding: utf-8 -*-
"""Recognizers for features.
"""
import logging

from enum import Enum

from feature_detection.transducer import Transducer

class Recognizer(Transducer):
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

        self.state_counter = 0
        self.condition = self.State.Running

        # TODO: Check transitions for sane structure
        self.transitions = transitions

        # TODO: Handle explicit states

    def is_done(self):
        return self.condition == self.State.Done

    def next(self, in_record):
        logging.debug("Processing:")
        
        in_state = self.state
        logging.debug(" -- current state (%s)", in_state)

        # This is the case for using the recognizer symbolically
        if self.transitions is None:
            out_state, out_record = in_state, in_record

        elif in_state in self.transitions:
            t = self.transitions[in_state]
            result = t(self.context, in_state, in_record)
            out_state, out_record = (in_state, None) if result is None else result

        else:
            logging.debug(" -- reached terminal state.")
            self.condition = self.State.Done
            return None

        logging.debug(" -- new state (%s)", out_state)
        logging.info("%8d : State = %s", self.state_counter, out_state)

        self.state = out_state
        self.state_counter += 1
        return out_record
