from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import metrics

#Carregando os dados do data set.
cancer = load_breast_cancer()

#fazendoa separação de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3)

#Fazendo a classificação
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print('Acurácia do KNN no conjunto de treinamento: {:.3f}'.format(knn.score(X_train,y_train)))
print('Acurácia do KNN no conjunto de teste: {:.3f}'.format(knn.score(X_test,y_test)))

print("Accuracy: {:.3f}".format(metrics.accuracy_score(y_test, y_pred)))