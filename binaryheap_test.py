import unittest

from binaryheap import BinaryHeap

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

class TestBinaryHeap(unittest.TestCase):
	def setUp(self):
		self.heap = BinaryHeap()

	# insert: 
	def test_insert(self):
		self.heap.insert(wa, 100)
		self.assertEqual(self.heap.array(), [wa])

	def test_insert_two(self):
		self.heap.insert(wa, 100)
		self.heap.insert(wb, 50)
		self.assertEqual(self.heap.array(), [wa, wb])

	def test_insert_out_of_order(self):
		self.heap.insert(wa, 50)
		self.heap.insert(wb, 100)
		self.assertEqual(self.heap.array(), [wb, wa])

	def test_negative_priority(self):
		custom_heap = BinaryHeap()
		custom_heap.insert(wa, -50)
		custom_heap.insert(wb, -100)
		self.assertEqual(custom_heap.array(), [wa, wb])

	def test_insert_many(self):
		self.heap.insert(wa, -100)
		self.heap.insert(wb, -50)
		self.heap.insert(wc, 0)
		self.heap.insert(wd, 50)
		self.heap.insert(we, 100)
		self.assertEqual(set(self.heap.array()), {wa, wb, wc, wd, we})
		self.assertTrue(self.heap._heapcheck())

	def test_sort(self):
		self.heap.insert(wa, -100)
		self.heap.insert(wb, 50)
		self.heap.insert(wc, 0)
		self.heap.insert(wd, -50)
		self.heap.insert(we, 100)
		self.heap.sort()
		self.assertEqual(self.heap.array(), [we, wb, wc, wd, wa])

	def test_decrease_key(self):
		self.heap.insert(wa, -100)
		self.heap.insert(wb, 50)
		self.heap.insert(wc, 0)
		self.heap.insert(wd, -50)
		self.heap.insert(we, 100)
		self.heap.sort()
		self.assertEqual(self.heap.array(), [we, wb, wc, wd, wa])
		self.heap.decrease_key(1, -2000)
		self.assertNotEqual(self.heap.array(), [we, wb, wc, wd, wa])
		self.heap.sort()
		self.assertEqual(self.heap.array(), [we, wc, wd, wa, wb])
