class ModelBase(object):
    def score(self, observation):
        raise NotImplementedError

    def export_values(self):
        raise NotImplementedError

    def import_values(self, values):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError