import os
import json
import copper
import tornado.web
import tornado.ioloop

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        # self.write("Hello, world")
        self.render("index.html")

class FoldersHandler(tornado.web.RequestHandler):
    def get(self):
        ans = [{'folder': folder} for folder in os.listdir(copper.config.path)]
        print(ans)
        self.write(json.dumps(ans))

settings = {
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
}

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/folders", FoldersHandler),
], debug=True, **settings)

if __name__ == "__main__":
    copper.config.path = '../project'
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
