setwd('./')
getwd()

loans = read.csv('loansData.csv')

plot(loans$Amount.Requested, loans$Interest.Rate)
plot(loans$Loan.Length, loans$Interest.Rate)

loans$Interest.Rate
