import os
import json
import numpy as np
import pandas as pd
import tornado.web
import tornado.ioloop

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class ColumnsHandler(tornado.web.RequestHandler):
    def get(self):
        df = pd.read_csv('explore.csv')
        ans = [{'id': i, 'name': column} for i, column in enumerate(df.columns)]
        self.write(json.dumps(ans))

class HistogramHandler(tornado.web.RequestHandler):
    def get(self, col_id):
        df = pd.read_csv('explore.csv')
        col = df[df.columns[int(col_id)]]
        nans = int(len(col) - col.count())
        col = col.dropna()

        ans = {}
        ans['col_name'] = col.name
        ans['nans'] = nans
        ans['values'] = col.values.tolist()
        self.write(json.dumps(ans))

settings = {
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
}

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/hist/([0-9]+)", HistogramHandler),
    (r"/columns", ColumnsHandler),
], debug=True, **settings)

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
