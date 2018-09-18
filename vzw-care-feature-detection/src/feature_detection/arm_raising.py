# -*- coding: utf-8 -*-
"""Specific recognizers for arm raises.
"""
import logging

from enum import Enum

from feature_detection.recognizer import Recognizer

kSideMap = {
    'left': {
        'wrist': 'LWrist',
        'shoulder': 'LShoulder',
        'hip': 'LHip'
    },
    'right': {
        'wrist': 'RWrist',
        'shoulder': 'RShoulder',
        'hip': 'RHip'
    }
}

kFieldMap = {
    'belt': 'MidHip'
}

def field_mapper(side):
    side_map = kSideMap[side]
    side_map_keys = set(side_map.keys())
    field_map_keys = set(kFieldMap.keys())
    def apply(name):
        if name in side_map_keys:
            return side_map[name]
        if name in field_map_keys:
            return kFieldMap[name]
        return name
    return apply

def extractor(mapper):
    def extract(data, *fields):
        return tuple(data.get(mapper(field)) for field in fields)

    def extract_all(data, *fields):
        row = extract(data, *fields)
        return row if all(v is not None for v in row) else None

    return dict(any=extract, all=extract_all)

class ArmRaised:
    class State(Enum):
        Unknown, Stable, Raised = range(3)

    def __init__(
            self, side,
            close_to_belt=0.2,
            raise_thresh=0.4,
            drop_thresh=0.3
    ):
        self.side = side
        self.field_mapper = field_mapper(side)
        self.extractor = extractor(self.field_mapper)
        self.close_to_belt = close_to_belt
        self.raise_thresh = raise_thresh
        self.drop_thresh = drop_thresh

    @staticmethod
    def _raise_phase(wrist, belt, shoulder):
        torso = abs(belt['y'] - shoulder['y'])
        assert torso > 0.0, "DEGENERATE BODY!"
        return 1.0 - min(1.0, max(0.0, wrist['y'] - shoulder['y'])/torso)

    def unknown(self, in_state, in_data):
        row = self.extractor['all'](in_data, 'wrist', 'belt', 'shoulder')
        if row is None:
            return None

        wrist, belt, shoulder = row 
        torso = abs(belt['y'] - shoulder['y'])
        assert torso > 0.0, "DEGENERATE BODY!"

        wrist_from_belt = abs(wrist['y'] - belt['y'])
        quiescents = wrist_from_belt / torso
        if quiescents < self.close_to_belt:
            return self.State.Stable, {
                "time": in_data['time'],
                "event": "Stablized",
                "arm": self.side
            }
        
        return None

    def stable(self, in_state, in_data):
        row = self.extractor['all'](in_data, 'wrist', 'belt', 'shoulder')
        if row is None:
            return None

        raise_phase = self._raise_phase(*row)
        if raise_phase >= self.raise_thresh:
            return self.State.Raised, {
                "time": in_data['time'],
                "event": "Raised",
                "arm": self.side
            }
        
        return None

    def raised(self, in_state, in_data):
        row = self.extractor['all'](in_data, 'wrist', 'belt', 'shoulder')
        if row is None:
            return None
        
        raise_phase = self._raise_phase(*row)
        if raise_phase < self.drop_thresh:
            return self.State.Stable, {
                "time": in_data['time'],
                "event": "Dropped",
                "arm": self.side
            }
        
        return None

    transitions = {
        State.Unknown: unknown,
        State.Stable: stable,
        State.Raised: raised
    }

    init_state = State.Unknown

    @classmethod
    def make(Self, *params, **kparams):
        return Recognizer(
            Self.transitions, Self.init_state, Self,
            params=params, kparams=kparams
        )
