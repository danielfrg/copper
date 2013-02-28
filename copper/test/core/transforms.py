import os
import copper
import pandas as pd

import unittest
from copper.test.CopperTest import CopperTest

class TransformsTest(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        # suite.addTest(TransformsTest(d'to_number'))
        suite.addTest(TransformsTest('strptime'))
        suite.addTest(TransformsTest('date2number'))
        return suite

    def to_number(self):
        self.setUpData()
        data = copper.read_csv('transforms/1/data.csv')
        sol = copper.read_csv('transforms/1/transformed.csv')

        t1 = data['Number.1'].apply(copper.transform.to_number)
        self.assertEqual(t1, sol['Number.1'])
        t2 = data['Number.2'].apply(copper.transform.to_number)
        self.assertEqual(t2, sol['Number.2'])

    def strptime(self):
        self.setUpData()
        data = copper.read_csv('transforms/1/data.csv')
        sol = copper.read_csv('transforms/1/transformed.csv')

        dates1 = data['Date.1'].apply(copper.transform.strptime, args='%Y-%m-%d')
        dates2 = data['Date.2'].apply(copper.transform.strptime, args='%Y/%m/%d')
        dates3 = data['Date.3'].apply(copper.transform.strptime, args=('%m/%d/%y'))
        dates1, dates2, dates3 = dates1.dropna(), dates2.dropna(), dates3.dropna()
        
        self.assertEqual(len(dates1), 12)
        self.assertEqual(dates1.values, dates2.values)
        self.assertEqual(dates2.values, dates3.values)
        self.assertEqual(dates1.values, dates3.values)

    def date2number(self):
        '''
        Requires:
            transforms.strptime
        '''
        self.setUpData()
        data = copper.read_csv('transforms/1/data.csv')
        sol = copper.read_csv('transforms/1/transformed.csv')

        # Default startdate

        dates1 = data['Date.1'].apply(copper.transform.strptime, args='%Y-%m-%d')
        dates2 = data['Date.2'].apply(copper.transform.strptime, args='%Y/%m/%d')
        dates3 = data['Date.3'].apply(copper.transform.strptime, args='%m/%d/%y')
        nums1 = dates1.apply(copper.transform.date_to_number)
        nums2 = dates2.apply(copper.transform.date_to_number)
        nums3 = dates3.apply(copper.transform.date_to_number)
        self.assertEqual(nums1.values, nums3.values)
        self.assertEqual(nums2.values, nums3.values)
        self.assertEqual(nums1.values, nums3.values)
        ans_1 = 13879
        self.assertEqual(nums1[0], ans_1)
        self.assertEqual(nums2[0], ans_1)
        self.assertEqual(nums3[0], ans_1)
        
        # Custom startdate
        from datetime import datetime
        copper.transform.start_date = datetime(2000, 1, 1)
        nums1_2 = dates1.apply(copper.transform.date_to_number)
        nums2_2 = dates2.apply(copper.transform.date_to_number)
        nums3_2 = dates3.apply(copper.transform.date_to_number)

        self.assertEqual(nums1_2.values, nums2_2.values)
        self.assertEqual(nums2_2.values, nums3_2.values)
        self.assertEqual(nums1_2.values, nums3_2.values)
        ans_1_2 = 2922
        self.assertEqual(nums1_2[0], ans_1_2)
        self.assertEqual(nums2_2[0], ans_1_2)
        self.assertEqual(nums3_2[0], ans_1_2)
        self.assertNotEqual(nums1[0], nums1_2[0])
        self.assertNotEqual(nums2[0], nums1_2[0])
        self.assertNotEqual(nums3[0], nums1_2[0])

if __name__ == '__main__':
    # unittest.main()
    suite = TransformsTest().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
