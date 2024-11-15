# ResNet Simplificada

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    return x

inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = residual_block(x, 32)
x = MaxPooling2D()(x)
x = residual_block(x, 64)
x = MaxPooling2D()(x)
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)

resnet_simple = Model(inputs, outputs)
resnet_simple.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

resnet_simple.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Def para calcular as métricas

def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro') * 100
    recall = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100

'''
Código acima: ResNet Simplificada 

É capaz de extrair características complexas sem o problema de gradientes desaparecendo. Isso frequentemente se traduz em métricas de accuracy, recall e F1-score superiores, especialmente em classes difíceis. 
Mesmo em uma versão simplificada, ResNet ainda demanda mais tempo de treinamento e poder de processamento devido às camadas residuais.

Lenet

É uma arquitetura menos complexa, com poucas camadas convolucionais e totalmente conectadas, o que reduz o custo computacional e permite treinar rapidamente o modelo.
Comparado às redes mais profundas, LeNet frequentemente apresenta precisão, recall e F1-score inferiores, especialmente para classes com variações complexas.

MiniVGGNes 
É mais custosa em termos computacionais, levando mais tempo e poder de processamento para treinar devido à quantidade de camadas e parâmetros.
Graças a suas camadas convolucionais empilhadas e regularização (com Dropout e Batch Normalization), MiniVGGNet costuma ter uma maior acurácia, precisão e F1-score do que LeNet.



'''