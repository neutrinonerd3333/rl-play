import random
import numpy

import itertools
import functools

def parent(ind):
    return (ind - 1) // 2

def children(ind):
    left = 2 * ind + 1
    return (left, left + 1)

class HeapLookupEntry:
    """
    A barebones implementation of an index-priority tuple for our heap.
    """
    def __init__(self, index, priority):
        self._index = index
        self._priority = priority

    def index(self):
        return self._index

    def priority(self):
        return self._priority

    def set_index(self, index):
        self._index = index

    def set_priority(self, priority):
        self._priority = priority

    def __repr__(self):
        return str((self._index, self._priority))


@functools.lru_cache(maxsize=128, typed=False)
def rank_based_partition(length: int, bins: int):
    """
    Partition the Riemann sum 1/1 + 1/2 + ... + 1/length into approximately
    equal-size bins, and return a list of the right-endpoints.
    """
    assert length >= bins

    probability_normalization = sum(map(lambda x: 1/x, range(1, length + 1)))
    bin_size = probability_normalization / bins

    partial_sum = 0
    right_endpoints = []
    for x in range(1, length + 1):
        partial_sum += (1 / x)
        if partial_sum >= bin_size:
            partial_sum -= bin_size
            right_endpoints.append(x)

    # handle floating point errors
    if right_endpoints[-1] != length:
        right_endpoints.append(length)

    # sanity check
    if len(right_endpoints) != bins:
        print(right_endpoints)
        raise RuntimeError("Length mismatch. length = {}, bins = {}".format(length, bins))
    return right_endpoints


# taken from https://docs.python.org/3/library/itertools.html#recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class BinaryHeap:
    """
    Max-heap implementation.
    """

    def __init__(self):
        self._array = []
        self._ind_lookup = dict()

    def _compare(self, item1, item2):
        items = [item1, item2]
        priorities = list(map(lambda i: self._ind_lookup[i].priority(), items))
        return priorities[0] >= priorities[1]

    def insert(self, item, priority):
        # should not call insert on things we already have
        assert item not in self._ind_lookup

        ind = len(self._array)
        self._ind_lookup[item] = HeapLookupEntry(ind, priority)
        self._array.append(item)
        self._up_heap(ind)

    def change_priority(self, item, priority):
        ind_lookup = self._ind_lookup[item]
        ind = ind_lookup.index()
        cur_priority = ind_lookup.priority()
        assert item == self._array[ind]

        if cur_priority == priority:
            return

        # update priority and maintain heap invariant
        self._ind_lookup[item].set_priority(priority)
        if cur_priority > priority:
            self._heapify(ind)
        elif cur_priority < priority:
            self._up_heap(ind)

    def _up_heap(self, ind):
        # basecase: root of the heap
        if ind == 0:
            return

        parent_ind = parent(ind)

        item = self._array[ind]
        parent_item = self._array[parent_ind]

        # maintain heap invariant if broken
        if not self._compare(parent_item, item):
            self._array[ind] = parent_item
            self._array[parent_ind] = item

            # swap in lookup array
            self._ind_lookup[item].set_index(parent_ind)
            self._ind_lookup[parent_item].set_index(ind)

            # recurse
            self._up_heap(parent_ind)

    def _heapify(self, ind):
        left, right = children(ind)
        if left >= len(self._array):
            return  # leaf node!
        elif right >= len(self._array):
            child_ind = left
        else:
            child_ind = left if self._compare(self._array[left], self._array[right]) else right

        item, child_item = self._array[ind], self._array[child_ind]
        # check heap invariant
        if self._compare(item, child_item):
            return
        
        self._array[ind], self._array[child_ind] = child_item, item
        self._ind_lookup[item].set_index(child_ind)
        self._ind_lookup[child_item].set_index(ind)

        self._heapify(child_ind)

    def trim(self):
        trimmed_item = self._array.pop()
        del self._ind_lookup[trimmed_item]
        return trimmed_item

    def array(self):
        return list(self._array)

    def _check_rep(self):
        if len(self._array) != len(self._ind_lookup):
            raise RuntimeError("Array and lookup table size mismatch")
        elif any([not self._compare(self._array[parent(i)], self._array[i]) for i in range(1, len(self._array))]):
            raise RuntimeError("Heap invariant violated! Internal array: {}".format(self._array))
        elif any([self._ind_lookup[item].index() != index for (index, item) in enumerate(self._array)]):
            raise RuntimeError("Lookup table corrupted; table: {}; array: {}".format(self._ind_lookup, self._array))

    def sort(self):
        self._array.sort(key=lambda x: -self._ind_lookup[x].priority())
        for (ind, item) in enumerate(self._array):
            self._ind_lookup[item].set_index(ind)

    def sample(self, n):
        assert len(self._array) >= n

        partitions = pairwise(itertools.chain([0], rank_based_partition(len(self._array), n)))
        return [random.choice(self._array[left:right])
            for (left, right) in partitions]

        # indices = [random.choice(range(left, right)) for (left, right) in partitions]
        # ret = [self._array[ind] for ind in indices]
        # priorities = [self._ind_lookup[item].priority() for item in ret]
        # return ret

        # # python cartpole-run.py --deep --batch-size 32 --gamma 0.9 --gamma-final 0.99 --anneal 100 -v --prioritize

    def max_priority(self):
        try:
            return self._ind_lookup[self._array[0]].priority()
        except IndexError:
            return 1  # if no elements currently in array

    def __len__(self):
        return len(self._array)
