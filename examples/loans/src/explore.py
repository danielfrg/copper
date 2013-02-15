import copper
import numpy as np
import matplotlib.pyplot as plt

copper.project.path = '..'

loans = copper.load('loans')
loans.role['Interest.Rate'] = loans.TARGET

# print (loans.columns)
# print (loans.metadata)
# print (loans.corr())

# print (loans)
# loans.histogram('Loan.Length')


# copper.plot.scatter(loans, 'Interest.Rate', 'FICO.Range', reg=True, s=50, alpha=0.1)
# copper.plot.scatter(loans, 'Interest.Rate', 'FICO.Range', 'Loan.Length')
# copper.plot.show()

loans.fillna(method='mean')
print (loans.variance_explained())
loans.role['Amount.Requested'] = loans.REJECTED
plt.draw()
print (loans.variance_explained())
plt.show()
# U, s, V = np.linalg.svd(values)
# var = np.square(s) / sum(np.square(s))
# print(var)
# xlocations = np.array(range(len(var)))+0.5
# width = 0.99
# plt.bar(xlocations, var, width=width)
# plt.show()
