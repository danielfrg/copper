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
        df = pd.read_csv('train.csv')
        ans = [{'id': i, 'name': column} for i, column in enumerate(df.columns)]
        self.write(json.dumps(ans))

class HistogramHandler(tornado.web.RequestHandler):
    def get(self, col_id):
        df = pd.read_csv('train.csv')
        col = df[df.columns[int(col_id)]]
        ans = {}
        bins = 20

        ans['col_name'] = col.name
        nans = int(len(col) - col.count())
        ans['nans'] = nans
        col = col.dropna()
        count, divis = np.histogram(col.values, bins=bins)
        ans['x_min'] = float(divis[0])
        ans['x_max'] = float(divis[-1])
        y_max = max(nans, float(max(count)))
        ans['y_max'] = y_max
        # labels = [{'count': int(c), 'init': float(i), 'final': float(f)}
                            # for c, i, f in zip(count, divis[:-1], divis[1:])]
        # ans['labels'] = labels
        # ans['count'] = count.tolist()
        # ans['data'] = [{'y': int(c), 'x': int(x)}
                                            # for c, x in zip(count, divis[:-1])]
        ans['values'] = col.values.tolist()
        # print(col)
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
