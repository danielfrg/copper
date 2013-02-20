setwd('/home/dfrodriguez/Dropbox/Documents/Coursera/Data Analysis/Quizes/quiz5/')

# Question 1
data(warpbreaks)
aov1 <- aov(warpbreaks$breaks ~ warpbreaks$tension + warpbreaks$wool)
summary(aov1)

# Question 2
log(0.2/(1-0.2))

# Question 3
library(glm2)
data(crabs)
glm1 <- glm(crabs$Satellites ~ crabs$Width, family='poisson')
glm1
exp(glm1$coefficients[2])

# Question 4
exp(glm1$coefficients[1]) * exp(glm1$coefficients[2] * 22)

# Question 5
library(MASS)
data(quine)

lm1 <- lm(Days+1 ~ ., data=quine)
aicFormula <- step(lm1)

