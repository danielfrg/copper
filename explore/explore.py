import os
import copper

import tornado.web
import tornado.ioloop
import tornado.autoreload

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("<a href='explore'>explore</a>")

class ExploreHandler(tornado.web.RequestHandler):
    def get(self):
        files = os.listdir(copper.config.export)
        self.render("templates/explore.html", files=files, data=None)

class ExploreFileHandler(tornado.web.RequestHandler):
    def get(self, ftarget):
        if ftarget == '':
            files = os.listdir(copper.config.export)
            self.render("templates/explore.html", files=files, data=None)
        else:
            import pandas as pd
            import json
            f = os.path.join(copper.config.export, ftarget)
            df = pd.read_csv(f)
            d = [
                dict([
                    (colname, row[i])
                    for i,colname in enumerate(df.columns)
                ])
                for row in df.values
            ]
            # return json.dumps(d)
            self.write(json.dumps(d))

def start(port=8000):
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "static"),
    }

    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/explore/(.*)", ExploreFileHandler),
        (r"/explore", ExploreHandler),
        (r"/()", tornado.web.StaticFileHandler,
                                            dict(path=settings['static_path'])),
    ], debug=True, **settings)
    application.listen(8000)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    copper.config.path = '../examples/donors'
    start(8000)

