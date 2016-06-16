import numpy as np
from rlagents.functions.decay import FixedDecay


class SimulatedAnnealing:
    def __init__(self, temperature=None, shift=None, bias=None):
        self.temperature = self.__set_temperature(temperature)
        self.testing_vals = None
        self.testing_result = 0
        self.shift = self.__set_shift(shift)
        self.bias = self.__set_bias(bias)

    @staticmethod
    def __set_temperature(temperature):
        if temperature is None:
            return FixedDecay(10, decay=0.997, minimum=0.1)

        return temperature

    @staticmethod
    def __set_bias(bias):
        if bias is None:
            return FixedDecay(1, decay=0.995, minimum=0.01)

        return bias

    @staticmethod
    def __set_shift(shift):
        if shift is None:
            return FixedDecay(1, decay=0.995, minimum=0.01)

        return shift

    def __set_vals(self, testing, result):
        self.testing_vals = testing.copy()
        self.testing_result = result

    def next_generation(self, batch, results):
        if len(batch) > 1:
            print "Only the first sample is used in simulated annealing"

        batch_vals = batch[0].export_values()
        result = results[0]

        # No saved value to compare against
        if self.testing_vals is None:
            self.__set_vals(batch_vals, result)

        acceptance_probability = np.e ** ((result - self.testing_result) / self.temperature.value)
        print self.testing_result, result, self.shift.value, self.temperature.value, acceptance_probability

        if np.random.uniform() < acceptance_probability:
            self.__set_vals(batch_vals, result)
            batch[0].import_values(self.__create_next_test(batch_vals, result))

        else:
            batch[0].import_values(self.__create_next_test(self.testing_vals.copy(), self.testing_result))

        # print self.testing_vals, batch[0].export_values()
        self.temperature.update()
        self.bias.update()
        self.shift.update()

        print self.testing_vals is batch_vals

        return batch

    def __create_next_test(self, batch_vals, result):
        choice = np.random.randint(len(batch_vals))

        batch_vals[choice] = np.random.normal(batch_vals[choice], self.shift.value)

        if self.bias.value != 0:
            batch_vals[choice] += np.random.normal(0, self.bias.value)

        return batch_vals
