# -*- coding: utf-8 -*-
# filename: debug_main.py
import web
from debug_handle import image_Handle

# urls = (
#     '/image/(.*)&(.*)', 'image_Handle',
# )

urls = (
    '/image/(.*)', 'image_Handle',
)

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()

# import web
# render = web.template.render('templates/')
# urls = (
#     '/(.*)', 'index'
# )
#
# class index:
#     def GET(self,name):
#         # i=web.input(name=None)
#         print(name)
#         return render.index(name)
#         #return "Hello, world!"
#
# if __name__ == "__main__":
#     app = web.application(urls, globals())
#     app.run()