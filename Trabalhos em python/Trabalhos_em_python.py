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

# Separa os dados e os rótulos
X = cifar10.data
y = cifar10.target

# Converte os rótulos para inteiros
y = y.astype(np.uint8)

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza os dados
X_train = X_train / 255.0
X_test = X_test / 255.0

# Treinamento e avaliação de três classificadores
classifiers = {
    'SVM': SVC(),
    'RandomForest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

# Nomes das classes CIFAR-10
class_nome = ['Avião', 'Automóvel', 'Pássaro', 'Gato', 'Cervo', 'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']

# Laço para treinar e avaliar os classificadores
for name, clf in classifiers.items():
    # Treina o classificador
    clf.fit(X_train, y_train)
    
    # Faz previsões
    y_pred = clf.predict(X_test)
    
    # Avaliação de métricas
    accuracy = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Classificador: {name}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(confusion_matrix(y_test, y_pred))

    
    # Avaliação de métricas
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classificador: {name}")
    print(f"Accuracy: {accuracy}")
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("Relatório de classificação:")
    print(classification_report(y_test, y_pred))

# Relatório de classificação em porcentagens com os nomes das classes
print("Relatório de classificação (em %):")
for i, label in enumerate(class_nome):
        metrics = report[str(i)]
        precision = metrics['precision'] * 100
        recall = metrics['recall'] * 100
        f1_score = metrics['f1-score'] * 100
        print(f"Classe {label} - Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-Score: {f1_score:.2f}%")


'''
SVM obteve uma accuracy de 55.23%, bom desempenho para classes como Avião e Navio.
RandomForest teve um melhor desempenho geral, com uma accuracy de 66.89%, e foi forte nas classes Navio e Caminhão.
KNN apresentou a menor accuracy 50.34%, com desempenhos medianos nas classes Automóvel e Navio.
'''