import copper
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

copper.project.path = '..'
loans = copper.load('loans')
loans.fix_names()
loans.fillna(method='mean')

# print(loans.frame)
# print (loans.metadata)
# loans.role['InterestRate'] = loans.TARGET
# print (loans.frame.skew())
# print (loans.corr())

# loans.histogram('Employment.Length')
# plt.draw()
# plt.figure()

# loans.histogram('MonthlyIncome')
# plt.show()


# mod = sm.ols(formula='InterestRate ~ FICORange + LoanLength', data=loans.frame)
# mod = sm.ols(formula='InterestRate ~ FICORange + LoanLength + C(LoanPurpose)', data=loans.frame)
# mod = sm.ols(formula='InterestRate ~ C(LoanPurpose)', data=loans.frame)
# res = mod.fit()
# print (res.summary())
# print (res.pvalues)


plt.subplot(131)
print(loans.variance_explained(plot=True))
plt.subplot(132)
copper.plot.scatter(loans, 'FICORange', 'InterestRate', reg=True, s=50, alpha=0.1)
plt.ylabel('Interest Rate')
plt.xlabel('FICO Range')
plt.subplot(133)
copper.plot.scatter(loans, 'AmountFundedByInvestors', 'InterestRate', 'LoanPurpose')
plt.ylabel('Interest Rate')
plt.xlabel('Amount Funded By Investors')
plt.show()

