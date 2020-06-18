class LinearMap(object):

	def __init__(self):
		self.items = []

	def add(self, k, v):
		self.items.append((k, v))

	def get(self, k):
		val_list = []
		for key, val in self.items:
			if key == k:
				val_list.append(val)
		# if len(val_list) == 0:
		# 	raise KeyError
		return val_list

	# def get(self, k):
	# 	for key, val in self.items:
	# 		if key == k:
	# 			return val
	# 	raise KeyError


class BetterMap(object):

	def __init__(self, n=100):
		self.maps = []
		for i in range(n):
			self.maps.append(LinearMap())

	def find_map(self, k):
		index = hash(k) % len(self.maps)
		return self.maps[index]

	def add(self, k, v):
		m = self.find_map(k)
		m.add(k, v)

	def get(self, k):
		m = self.find_map(k)
		return m.get(k)


class HashMap(object):
	def __init__(self, maplen):
		self.maps = BetterMap(maplen)
		self.num = 0

	def get(self, k):
		return self.maps.get(k)

	# def add(self, k, v):
	# 	if self.num == len(self.maps.maps):
	# 		self.resize()
	#
	# 	self.maps.add(k, v)
	# 	self.num += 1

	def add(self, k, v):
		if self.num == len(self.maps.maps):
			self.sub_resize(v)
			if self.num == len(self.maps.maps):
				self.resize()

		self.maps.add(k, v)
		self.num += 1

	def resize(self):
		new_map = BetterMap(self.num * 2)

		for m in self.maps.maps:
			for k, v in m.items:
				new_map.add(k, v)

		self.maps = new_map

	def sub_resize(self, last_v):
		for m in self.maps.maps:
			for i in range(len(m.items)):
				# if last_v - m.items[i][1] > 15:
				if last_v == m.items[i][1]:
					m.items.pop(i)
					self.num -= 1

	def key(self):
		keylist = []

		for m in self.maps.maps:
			for k, _ in m.items:
				keylist.append(k)
		return keylist

	def print_map(self):
		maplist = []

		for m in self.maps.maps:
			for k, v in m.items:
				maplist.append([k, v])
		return maplist


# def main(script):
# 	import string
#
# 	m = HashMap(2)
# 	s = string.ascii_lowercase
#
# 	for k, v in enumerate(s):
# 		m.add(k, v)
#
# 	for k in range(len(s)):
# 		print(k, m.get(k))
#
#
# if __name__ == '__main__':
# 	import sys
# 	main(*sys.argv)