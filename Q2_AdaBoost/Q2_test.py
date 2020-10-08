import numpy as np
from Q2_AdaBoost.AdaBoost import AdaBoost

print('==========Train==========')
X = np.random.rand(10,3)
Y = np.random.randint(2, size=10)
Y[Y == 0] = -1
classifier = AdaBoost(T=5)
print('==========X_train==========')
print(X)
print('==========Y_train==========')
print(Y)
print('==========Training==========')
classifier.train(X,Y)
X_test = np.random.rand(1,3)
print('==========Test==========')
print('Test sample:', X_test)
y_pred = classifier.predict(X_test)
print('Predict:', y_pred)
