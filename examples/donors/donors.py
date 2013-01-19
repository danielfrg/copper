import copper

copper.config.data_dir = './'
ds = copper.load('donors.csv')
ds.role['DemMedIncome'] = 'Rejected'
print(ds)

# ds.export(name='frame', format='csv', w='frame')
# ds.export(name='inputs', format='csv', w='inputs')
# ds.export(name='target', format='csv', w='target')

# Compare

original = ds._oframe[['GiftAvgLast', 'GiftAvg36', 'GiftAvgAll', 'GiftAvgCard36']][0:10]
new = ds.frame[['GiftAvgLast', 'GiftAvg36', 'GiftAvgAll', 'GiftAvgCard36']][0:10]
# print(original)
# print(new)

original = ds._oframe[['DemGender']][0:10]
new = ds.inputs[['DemGender [F]', 'DemGender [M]', 'DemGender [U]']][0:10]
print(original)
print(new)
