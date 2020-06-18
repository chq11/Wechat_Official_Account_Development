class image_Handle(object):
	def GET(self, name):
		try:
			f = open('./DRN/received_image/'+name+'.jpg', 'rb')
			return f.read()

		except Exception:
			return ('please try again later!')