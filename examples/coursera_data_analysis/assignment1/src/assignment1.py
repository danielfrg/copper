import copper

copper.project.path = '..'

dataset = copper.Dataset()
dataset.load('loansData.csv')
# data = copper.read_csv('loansData.csv')

print dataset
