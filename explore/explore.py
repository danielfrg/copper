import copper

import tornado.web
import tornado.ioloop
import tornado.autoreload

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

def start(port=8000):
    application = tornado.web.Application([
        (r"/", MainHandler),
    ])
    application.listen(8000)
    tornado.autoreload.start()
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    copper.config.path = '../examples/donors'
    start(8000)

