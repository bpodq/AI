library(ggplot2)

X_train = read.csv('X_train.csv')
Y_train = read.csv('Y_train.csv')
X_test = read.csv('X_test.csv')
Y_test = read.csv('Y_test.csv')

data = data.frame(X_train, Y_train)

# qplot(X_train[, 1], X_train[, 2], color='blue')
ggplot(data, aes(x1, x2, color=y)) +
	geom_point()

data_test = data.frame(X_test, Y_test)	
ggplot(data_test, aes(x1, x2, color=y)) + geom_point()

