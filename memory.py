import pandas as pd


class MemoryItemBase(object):
    def __init__(self, size):
        self.size = size

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, s):
        if not isinstance(s, int):
            raise TypeError("Memory size must be an int")

        if s < 1:
            raise ValueError("Memory size must be greater than 1")

        self._size = s

    def count(self):
        return NotImplementedError

    def store(self, item):
        raise NotImplementedError

    def fetch_last(self, key):
        raise NotImplementedError


class ListItem(MemoryItemBase):
    def __init__(self, size):
        MemoryItemBase.__init__(self, size)
        self.items = []

    def store(self, item):
        self.items.append(item)

        if len(self.items) > self.size:
            self.items = self.items[-self.size:]

    def fetch_last(self, i):
        return self.items[-i:]

    def count(self):
        return len(self.items)


class MemoryBase(object):
    def __init__(self, size=100, columns=None):
        self.size = size

        if columns is not None:
            self.new(columns)

    def new(self, name):
        """
        Adds new columns to memory

        Parameters:
            string or list of strings
        """
        raise NotImplementedError

    def store(self, d):
        """
        Takes in a dictionary or list of dictionaries and adds to memory
        """
        raise NotImplementedError

    def fetch_last(self, i, name=None, return_type="dict"):
        """
        Returns last i rows for column if name, else table
        return_type:
            "dict" - Dictionary
            "numpy" - Numpy array (values only)
            "pandas" - Pandas DataFrame
        """
        raise NotImplementedError

    def count(self, name=None):
        """
        If name is a string returns count stored for that column
        Else returns max stored columns
        """
        raise NotImplementedError

    def update(self, d):
        """
        Takes in dictionary and updates most recent entry
        """
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

class ListMemory(MemoryBase):
    def __init__(self, size=100, columns=None):
        self.items = {}  # Must go before Base init as Base may call it

        MemoryBase.__init__(self, size, columns)

    def new(self, name):
        if isinstance(name, str):
            self.items[name] = ListItem(size=self.size)

        else:
            for n in name:
                self.items[n] = ListItem(size=self.size)

    def store(self, d):
        if isinstance(d, dict):
            for key, value in d.iteritems():
                self.items[key].store(value)

            mem_keys = [key for key in self.items.keys() if key not in d.keys()]

            for mem_key in mem_keys:
                self.items[mem_key].store(None)

        else:
            for m in d:
                for key, value in m.iteritems():
                    self.items[key].store(value)

                mem_keys = [key for key in self.items.keys() if key not in m.keys()]

                for mem_key in mem_keys:
                    self.items[mem_key].store(None)

    def fetch_last(self, i, name=None, return_type="dict"):
        if return_type not in ["dict"]:
            raise TypeError("Return type not currently supported")

        if name is not None:
            return self.items[name].fetch_last(i)

        else:
            ret_dict = {}

            for key in self.items.keys():
                ret_dict[key] = self.items[key].fetch_last(i)

            return ret_dict

    def count(self, name=None):
        if name is not None:
            return self.items[name].count()

        else:
            return max([self.items[key].count() for key in self.items.keys()])

    def update(self, d):
        for key, value in d.iteritems():
            self.items[key].items[-1] = value

    def export(self):
        return {"Type": "List Memory",
                "Size": self.size,
                "Columns": self.items.keys()}


class PandasMemory(MemoryBase):
    def __init__(self, size=100, columns=None):
        self.df = pd.DataFrame()    # Has to go before base init as based may call it

        MemoryBase.__init__(self, size, columns)

    def new(self, name):
        if isinstance(name, str):
            self.df[name] = None

        else:
            for n in name:
                self.df[n] = None

    def store(self, d):
        if isinstance(d, dict):
            self.df = self.df.append(d, ignore_index=True)

        else:
            for m in d:
                self.df = self.df.append(m, ignore_index=True)

    def fetch_last(self, i, name=None, return_type="pandas"):
        if return_type not in ["pandas"]:
            raise TypeError("Return type not currently supported")

        if return_type == "pandas":
            ret_df = self.df.ix[-i:, name] if name is not None else self.df.iloc[-i:]
            ret_df.reset_index(drop=True, inplace=True)
            return ret_df

    def count(self, name=None):
        if name is not None:
            return self.df[name].count()

        else:
            return len(self.df.index)

    def update(self, d):
        curr_index = self.count()
        for key, value in d.iteritems():
            self.df.ix[curr_index-1, key] = value

    def export(self):
        return {"Type": "Pandas",
                "Size": self.size,
                "Columns": self.df.columns()}
