import os
import copper
import pandas as pd
import numpy as np

import unittest
from copper.tests.CopperTest import CopperTest

class TransformsTest(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        # suite.addTest(TransformsTest('test_to_number'))
        # suite.addTest(TransformsTest('test_strptime'))
        # suite.addTest(TransformsTest('test_date2number'))
        # suite.addTest(TransformsTest('test_category2ml'))
        # suite.addTest(TransformsTest('test_category2number'))
        # suite.addTest(TransformsTest('test_category_labels'))
        # suite.addTest(TransformsTest('test_inputs2ml'))
        suite.addTest(TransformsTest('test_target2ml'))
        return suite

    def test_to_number(self):
        self.setUpData()
        rands = np.round(np.random.rand(5) * 100)
        money = ['$%d'%num for num in rands]
        dic = { 'Money': money,
                'Cat.1': ['A', 'B', 'A', 'A', 'B'],
                'Num.as.Cat': ['1', '0', '1', '0', '1'],
                'Num.1': np.random.rand(5)}
        df = pd.DataFrame(dic)

        dic = { 'Money': rands,
                'Cat.1': [np.nan, np.nan, np.nan, np.nan, np.nan],
                'Num.as.Cat': [1, 0, 1, 0, 1]}
        sol = pd.DataFrame(dic)

        tr = df['Num.as.Cat'].apply(copper.transform.to_number)
        self.assertEqual(tr.dtype, float)
        self.assertEqual(tr, sol['Num.as.Cat'])
        tr = df['Cat.1'].apply(copper.transform.to_number)
        self.assertEqual(tr.dtype, float)
        self.assertEqual(tr, sol['Cat.1'])
        tr = df['Money'].apply(copper.transform.to_number)
        self.assertEqual(tr.dtype, float)
        self.assertEqual(tr, sol['Money'])

    def test_strptime(self):
        from datetime import datetime
        dates_1 = ['2000-1-12', '2001-12-31', '2002-3-3']
        dates_2 = ['2000.1.12', '2001.12.31', '2002.3.3']
        dates_3 = ['20000112', '20011231', '20020303']
        dates_4 = ['12/1/00', '31/12/01', '3/3/02']
        dic = {1:dates_1, 2:dates_2, 3:dates_3, 4:dates_4}
        df = pd.DataFrame(dic)

        sol = [datetime(2000,1,12), datetime(2001,12,31), datetime(2002,3,3)]
        t1 = df[1].apply(copper.transform.strptime, args='%Y-%m-%d')
        t2 = df[2].apply(copper.transform.strptime, args='%Y.%m.%d')
        t3 = df[3].apply(copper.transform.strptime, args='%Y%m%d')
        t4 = df[4].apply(copper.transform.strptime, args='%d/%m/%y')

        self.assertEqual(t1.values.tolist(), sol)
        self.assertEqual(t2.values.tolist(), sol)
        self.assertEqual(t3.values.tolist(), sol)
        self.assertEqual(t4.values.tolist(), sol)

    def test_date2number(self):
        '''
        Requires:
            transforms.strptime
        '''
        from datetime import datetime
        dates1 = [datetime(1970,1,1), datetime(1970,1,2), datetime(1971,1,1)]
        dates2 = [datetime(2000,1,1), datetime(2001,12,31), datetime(2002,3,3)]
        dates = dates1 + dates2
        df = pd.DataFrame(dates)

        t = df[0].apply(copper.transform.date_to_number)

        self.assertEqual(t[0], 0)
        self.assertEqual(t[1], 1)
        self.assertEqual(t[2], 365)
        self.assertEqual(t[3], 10957)
        self.assertEqual(t[4], 11687)
        self.assertEqual(t[5], 11749)

        # Custom startdate
        copper.transform.start_date = datetime(2000, 1, 1)
        t = df[0].apply(copper.transform.date_to_number)
        
        self.assertEqual(t[0], -10957)
        self.assertEqual(t[1], -10956)
        self.assertEqual(t[2], -10592)
        self.assertEqual(t[3], 0)
        self.assertEqual(t[4], 730)
        self.assertEqual(t[5], 792)
    
    def test_category2ml(self):
        d = ['A', 'B', 'A', 'A', 'C', 'A', 'B', 'A', 'C', 'B']
        df = pd.DataFrame(d)
        sol = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1],
                        [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]] )
        sol = pd.DataFrame(sol, columns=['0 [A]', '0 [B]', '0 [C]'])
        
        tr = copper.transform.category2ml(df[0])
        self.assertEqual(tr, sol)

    def test_category2number(self):
        d = ['A','B','A','A','C','A','B','A','C','B','D','A']
        df = pd.DataFrame(d)
        sol = [0,1,0,0,2,0,1,0,2,1,3,0]
        sol = pd.DataFrame(sol)

        tr = copper.transform.category2number(df[0])
        self.assertEqual(tr, sol[0])

    def test_category_labels(self):
        d = ['A','B','A','C','B','C','B','D','A','Z','A','X','D']
        df = pd.DataFrame(d)
        sol = ['A','B','C','D','X','Z']

        labels = copper.transform.category_labels(df[0])
        tr = copper.transform.category2number(df[0])
        self.assertEqual(labels.tolist(), sol)

    def test_inputs2ml(self):
        dic = { 'Cat.1': ['A','B','A','A','B'],
                'Cat.2' :['f','g','h','g','f'],
                'Num.1': np.random.rand(5),
                'Num.2': np.random.rand(5)}
        ds = copper.Dataset(pd.DataFrame(dic))

        sol = pd.DataFrame(index=ds.index)
        sol['Cat.1 [A]'] = [1,0,1,1,0]
        sol['Cat.1 [B]'] = [0,1,0,0,1]
        sol['Cat.2 [f]'] = [1,0,0,0,1]
        sol['Cat.2 [g]'] = [0,1,0,1,0]
        sol['Cat.2 [h]'] = [0,0,1,0,0]
        sol['Num.1'] = ds['Num.1']
        sol['Num.2'] = ds['Num.2']

        tr = copper.transform.inputs2ml(ds)
        self.assertEqual(tr, sol)

        # Change the role of a column to REJECT
        ds.role['Cat.1'] = ds.REJECT
        sol = pd.DataFrame(index=ds.index)
        sol['Cat.2 [f]'] = [1,0,0,0,1]
        sol['Cat.2 [g]'] = [0,1,0,1,0]
        sol['Cat.2 [h]'] = [0,0,1,0,0]
        sol['Num.1'] = ds['Num.1']
        sol['Num.2'] = ds['Num.2']

        tr = copper.transform.inputs2ml(ds)
        self.assertEqual(tr, sol)

        # Change the role of a column to TARGET
        ds.role['Cat.2'] = ds.TARGET
        sol = pd.DataFrame(index=ds.index)
        sol['Num.1'] = ds['Num.1']
        sol['Num.2'] = ds['Num.2']

        tr = copper.transform.inputs2ml(ds)
        self.assertEqual(tr, sol)

    def test_target2ml(self):
        dic = { 'Cat.1': ['A','B','A','A','B'],
                'Cat.2' :['f','g','h','g','f'],
                'Num.1': np.random.rand(5),
                'Num.2': np.random.rand(5)}
        ds = copper.Dataset(pd.DataFrame(dic))
        
        tr = copper.transform.target2ml(ds)
        self.assertEqual(tr, None)

        # Numerical target
        ds.role['Num.1'] = ds.TARGET
        tr = copper.transform.target2ml(ds)
        self.assertEqual(tr, ds['Num.1'])

        # Categorical target
        ds.role['Num.1'] = ds.INPUT
        ds.role['Cat.1'] = ds.TARGET
        sol = pd.DataFrame(index=ds.index)
        sol['Cat.1'] = [0,1,0,0,1]
        tr = copper.transform.target2ml(ds)
        self.assertEqual(tr, sol['Cat.1'])
        

if __name__ == '__main__':
    suite = TransformsTest().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)