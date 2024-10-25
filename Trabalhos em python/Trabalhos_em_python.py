import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carrega o dataset CIFAR-10
cifar10 = fetch_openml('cifar_10', version=1, cache=True)

# Separa os dados e os r�tulos
X = cifar10.data
y = cifar10.target

# Converte os r�tulos para inteiros
y = y.astype(np.uint8)

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza os dados
X_train = X_train / 255.0
X_test = X_test / 255.0

# Treinamento e avalia��o de tr�s classificadores
classifiers = {
    'SVM': SVC(),
    'RandomForest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

# Nomes das classes CIFAR-10
class_nome = ['Avi�o', 'Autom�vel', 'P�ssaro', 'Gato', 'Cervo', 'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminh�o']

# La�o para treinar e avaliar os classificadores
for name, clf in classifiers.items():
    # Treina o classificador
    clf.fit(X_train, y_train)
    
    # Faz previs�es
    y_pred = clf.predict(X_test)
    
    # Avalia��o de m�tricas
    accuracy = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Classificador: {name}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(confusion_matrix(y_test, y_pred))

    
    # Avalia��o de m�tricas
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classificador: {name}")
    print(f"Accuracy: {accuracy}")
    print("Matriz de confus�o:")
    print(confusion_matrix(y_test, y_pred))
    print("Relat�rio de classifica��o:")
    print(classification_report(y_test, y_pred))

# Relat�rio de classifica��o em porcentagens com os nomes das classes
print("Relat�rio de classifica��o (em %):")
for i, label in enumerate(class_nome):
        metrics = report[str(i)]
        precision = metrics['precision'] * 100
        recall = metrics['recall'] * 100
        f1_score = metrics['f1-score'] * 100
        print(f"Classe {label} - Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-Score: {f1_score:.2f}%")


'''
SVM obteve uma accuracy de 55.23%, bom desempenho para classes como Avi�o e Navio.
RandomForest teve um melhor desempenho geral, com uma accuracy de 66.89%, e foi forte nas classes Navio e Caminh�o.
KNN apresentou a menor accuracy 50.34%, com desempenhos medianos nas classes Autom�vel�e�Navio.
'''