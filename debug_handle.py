import web
import hashlib

render = web.template.render('./templates/')

class image_Handle(object):
	def GET(self, name):
		try:
			# i = web.input(name=None)
			return name


			# data = web.input(name)
			# print(data)
			# return 'asdsdf'
			# print(data.keys())
			# if len(data) == 0:
			# 	f = open('/home/haiquan/deep_learning/wechat_chat_robot_python3/DRN/debug/babyx4.png', 'rb')
			# 	return f.read()
				# return "hello, this is image ip!"

		except Exception:
			return