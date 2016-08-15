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

    def retrieve_last(self, key):
        raise NotImplementedError


class ListItem(MemoryItemBase):
    def __init__(self, size):
        MemoryItemBase.__init__(self, size)
        self.items = []

    def store(self, item):
        self.items.append(item)

        if len(self.items) > self.size:
            self.items = self.items[-self.size:]

    def retrieve_last(self, i):
        return self.items[-i:]

    def count(self):
        return len(self.items)


class MemoryBase(object):
    def __init__(self, size):
        self.size = size

    def new(self, name):
        raise NotImplementedError

    def store(self, name, item):
        raise NotImplementedError

    def retrieve_last(self, name, i):
        raise NotImplementedError

    def count(self, name):
        raise NotImplementedError


class ListMemory(MemoryBase):
    def __init__(self, size=100):
        MemoryBase.__init__(self, size)
        self.items = {}

    def new(self, name):
        if isinstance(name, str):
            self.items[name] = ListItem(size=self.size)

        else:
            for n in name:
                self.items[n] = ListItem(size=self.size)

    def store(self, name, item):
        self.items[name].store(item)

    def retrieve_last(self, name, i):
        self.items[name].retrieve_last(i)

    def count(self, name):
        self.items[name].count()