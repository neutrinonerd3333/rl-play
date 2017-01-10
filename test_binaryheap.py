import unittest

from binaryheap import BinaryHeap, rank_based_partition

# class Widget:
# 	def __init__(self, name, key):
# 		self.name = name
# 		self.key = key

# 	def __lt__(self, other):
# 		return self.key < other.key

# 	def __gt__(self, other):
# 		return self.key > other.key

wa = "a"
wb = "b"
wc = "c"
wd = "d"
we = "e"
very_big_number = 987654321

class TestBinaryHeap(unittest.TestCase):
	def setUp(self):
		self.heap = BinaryHeap()

	# insert: 
	def test_insert(self):
		self.heap.insert(wa, 100)
		self.heap._check_rep()
		self.assertEqual(self.heap.array(), [wa])

	def test_insert_two(self):
		self.heap.insert(wa, 100)
		self.heap.insert(wb, 50)
		self.heap._check_rep()
		self.assertEqual(self.heap.array(), [wa, wb])

	def test_insert_out_of_order(self):
		self.heap.insert(wa, 50)
		self.heap.insert(wb, 100)
		self.heap._check_rep()
		self.assertEqual(self.heap.array(), [wb, wa])

	def test_negative_priority(self):
		custom_heap = BinaryHeap()
		custom_heap.insert(wa, -50)
		custom_heap.insert(wb, -100)
		self.heap._check_rep()
		self.assertEqual(custom_heap.array(), [wa, wb])

	def test_insert_many(self):
		self.heap.insert(wa, -100)
		self.heap.insert(wb, -50)
		self.heap.insert(wc, 0)
		self.heap.insert(wd, 50)
		self.heap.insert(we, 100)
		self.assertEqual(set(self.heap.array()), {wa, wb, wc, wd, we})
		self.heap._check_rep()

	def test_sort(self):
		self.heap.insert(wa, -100)
		self.heap.insert(wb, 50)
		self.heap.insert(wc, 0)
		self.heap.insert(wd, -50)
		self.heap.insert(we, 100)
		self.heap.sort()
		self.heap._check_rep()
		self.assertEqual(self.heap.array(), [we, wb, wc, wd, wa])

	def test_decrease_key(self):
		self.heap.insert(wa, -100)
		self.heap.insert(wb, 50)
		self.heap.insert(wc, 0)
		self.heap.insert(wd, -50)
		self.heap.insert(we, 100)
		self.heap.sort()
		self.heap._check_rep()
		self.assertEqual(self.heap.array(), [we, wb, wc, wd, wa])

		self.heap.change_priority(wb, -2000)
		self.assertNotEqual(self.heap.array(), [we, wb, wc, wd, wa])
		self.heap._check_rep()

		self.heap.sort()
		self.assertEqual(self.heap.array(), [we, wc, wd, wa, wb])
		self.heap._check_rep()

	def test_increase_key(self):
		self.heap.insert(wa, -100)
		self.heap.insert(wb, 50)
		self.heap.insert(wc, 0)
		self.heap.insert(wd, -50)
		self.heap.insert(we, 100)
		self.heap.sort()
		self.heap._check_rep()
		self.assertEqual(self.heap.array(), [we, wb, wc, wd, wa])

		self.heap.change_priority(wb, 2000)
		self.assertNotEqual(self.heap.array(), [we, wb, wc, wd, wa])
		self.heap._check_rep()
		self.heap.sort()
		self.assertEqual(self.heap.array(), [wb, we, wc, wd, wa])

	def test_max_priority(self):
		self.heap.insert(wa, very_big_number)
		self.assertEqual(self.heap.max_priority(), very_big_number)

	def test_max_priority_multiple(self):
		self.heap.insert(wc, 20)
		self.heap.insert(wa, very_big_number)
		self.heap.insert(we, -500)
		self.assertEqual(self.heap.max_priority(), very_big_number)

class TestRankPartition(unittest.TestCase):
	
	def test_trivial(self):
		right_endpoints = rank_based_partition(1, 1)
		self.assertEqual(right_endpoints, [1])

	def test_one_bin(self):
		right_endpoints = rank_based_partition(5, 1)
		self.assertEqual(right_endpoints, [5])

	def test_two_bins(self):
		right_endpoints = rank_based_partition(3, 2)
		self.assertEqual(right_endpoints, [1, 3])

	def test_many_bins(self):
		right_endpoints = rank_based_partition(10, 3)
		self.assertEqual(right_endpoints, [1, 4, 10])

	def test_too_many_bins_regression(self):
		"""
		This is a regression test.
		"""
		right_endpoints = rank_based_partition(14, 14)
		self.assertEqual(right_endpoints, list(range(1, 15)))
