import rlagents

def epsilongreedy_decay():
    return rlagents.functions.decay.FixedDecay(0.1, 1, 0.1)