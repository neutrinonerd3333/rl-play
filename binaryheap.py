import numpy


def parent(ind):
    return (ind - 1) // 2

def children(ind):
    left = 2 * ind + 1
    return (left, left + 1)


class HeapElement:
    def __init__(self, name, key):
        self._name = name
        self._key = key

    def name(self):
        return self._name

    def key(self):
        return self._key

    def __ge__(self, other):
        return self._key >= other._key

    def __le__(self, other):
        return self._key <= other._key

class BinaryHeap:
    def __init__(self):
        self._array = []

    def _compare(self, priority1, priority2):
        return priority1 >= priority2

    def insert(self, item, priority):
        self._array.append(HeapElement(item, priority))
        ind = len(self._array) - 1
        self._up_heap(ind)

    def decrease_key(self, ind, priority):
        cur_item = self._array[ind].name()
        self._array[ind] = HeapElement(cur_item, priority)
        self._heapify(ind)

    def _up_heap(self, ind):
        if ind == 0:
            return

        parent_ind = parent(ind)
        cur_elt = self._array[ind]
        parent_elt = self._array[parent_ind]
        if not self._compare(parent_elt, cur_elt):
            self._array[ind] = parent_elt
            self._array[parent_ind] = cur_elt
            self._up_heap(parent_ind)

    def _heapify(self, ind):
        left, right = children(ind)
        if left >= len(self._array):
            return  # leaf node!
        elif right >= len(self._array):
            child_ind = right
        else:
            child_ind = left if self._compare(self._array[left], self._array[right]) else right

        cur_elt, child_elt = self._array[ind], self._array[child_ind]
        # check heap invariant
        if self._compare(cur_elt, child_elt):
            return
        
        self._array[ind], self._array[child_ind] = child_elt, cur_elt
        self._heapify(child_ind)

    def trim(self):
        return self._array.pop()

    def array(self):
        return list(map(lambda x: x.name(), self._array))

    def _heapcheck(self):
        return all([self._compare(self._array[parent(i)], self._array[i]) for i in range(1, len(self._array))])

    def sort(self):
        self._array.sort(key=lambda x: -x.key())

    def __len__(self):
        return len(self._array)
