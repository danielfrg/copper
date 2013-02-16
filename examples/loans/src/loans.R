setwd('/home/dfrodriguez/projects/copper/examples/loans/src/')

data <- read.csv("../data/exported/loans.clean.csv")

library(imputation)
data.imputed = kNNImpute(data, 5)

lm1 = lm(formula = data$Interest.Rate ~ data$FICO.Range + data$Loan.Length + data$Amount.Funded.By.Investors)
summary(lm1)