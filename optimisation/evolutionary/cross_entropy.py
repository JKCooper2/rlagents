import numpy as np


class CrossEntropy:
    def __init__(self, elite=0.2):
        self.elite = elite  # Percentage of samples selected to generate next batch

    def next_generation(self, batch, results):
        elite_n = int(len(batch) * self.elite)

        batch_vals = [b.export_values() for b in batch]

        # Select the x best scoring samples based on reverse score order
        best = np.array([batch_vals[b] for b in np.argsort(results)[-elite_n:]])

        # Update the mean/std based on new values
        mean = best.mean(axis=0)
        std = best.std(axis=0)

        for i in range(len(batch)):
            batch[i].import_values(np.random.normal(mean, std))

        return batch
