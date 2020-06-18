# -*- coding: utf-8 -*-
# filename: main.py
import web
from handle import Handle
from image_result import image_Handle

urls = (
    '/wx', 'Handle',
    '/image/(.*)', 'image_Handle',
)

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()
