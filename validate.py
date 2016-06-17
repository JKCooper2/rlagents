import numpy as np


def decay(_decay):
    if not callable(getattr(_decay, "update", None)):
        raise NotImplementedError("Decay must implement .update()")

    if getattr(_decay, "value", None) is None:
        raise NotImplementedError("Decay must contain 'value' property")


def epsilon_greedy(_epsilon_greedy):
    if getattr(_epsilon_greedy, "epsilon", None) is None:
        raise NotImplementedError("Epsilon Greedy must contain 'epsilon' property")

    if not callable(getattr(_epsilon_greedy, "update", None)):
        raise NotImplementedError("Epsilon Greedy must implement .update()")

    if not callable(getattr(_epsilon_greedy, "choose_action", None)):
        raise NotImplementedError("Epsilon Greedy must implement .choose_action()")


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

    if (min_eq and _number < minimum) or _number < minimum:
        raise ValueError("Number is below the accepted range")

    if (max_eq and _number > maximum) or _number >= maximum:
        raise ValueError("Number is above the accepted range")
