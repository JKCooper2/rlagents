import numpy as np


def decay(_decay):
    if not callable(getattr(_decay, "update", None)):
        raise NotImplementedError("Decay must implement .update()")

    if getattr(_decay, "value", None) is None:
        raise NotImplementedError("Decay must contain 'value' property")


def exploration(_exploration):
    if getattr(_exploration, "value", None) is None:
        raise NotImplementedError("Exploration must contain 'value' property")

    if not callable(getattr(_exploration, "update", None)):
        raise NotImplementedError("Exploration must implement .update()")

    if not callable(getattr(_exploration, "choose_action", None)):
        raise NotImplementedError("Exploration must implement .choose_action()")


def action_space(_action_space):
    if not callable(getattr(_action_space, "sample", None)):
        raise NotImplementedError("Action Space must implement .sample()")

    if not callable(getattr(_action_space, "contains", None)):
        raise NotImplementedError("Action Space must implement .contains()")


def model(_model):
    if not callable(getattr(_model, "action", None)):
        raise NotImplementedError("Model must implement .action(observation)")

    if not callable(getattr(_model, "action_value", None)):
        raise NotImplementedError("Model must implement .action_value(observation)")


def action(_action_space, _action):
    if not _action_space.contains(_action):
        raise ValueError("Action is not contained within action_space")


def number_range(_number, minimum=-np.inf, maximum=np.inf, min_eq=False, max_eq=False):
    if maximum < minimum:
        raise ValueError("Max must be greater than min")

    if (min_eq and _number < minimum) or (not min_eq and _number <= minimum):
        raise ValueError("Number is below the accepted range")

    if (max_eq and _number > maximum) or (not max_eq and _number >= maximum):
        raise ValueError("Number is above the accepted range")


def observation_fa(_observation_fa):
    if not callable(getattr(_observation_fa, "convert", None)):
        raise NotImplementedError("Observation FA must implement .convert(observation)")
