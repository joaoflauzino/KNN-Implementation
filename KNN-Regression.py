import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#Funcao para media
def distance(x, y):

    sum_sqr_distance = 0

    for i in range(len(x)):
        sum_sqr_distance += math.pow(x[i] - y[i], 2)

    return math.sqrt(sum_sqr_distance)

#Função para media
def mean(x):

    return sum(x)/len(x)

#Função para knn-regression
def knn_regression(train, observation, k):

    neighbor_dist_and_ind = [] #Lista para armazenar distancias e indices

    for index, object in enumerate(train):

        distance_of_objects =  distance(object[1:], observation) #Calculando distancia do objeto dos demais
        neighbor_dist_and_ind.append((distance_of_objects, index)) #Armazenando distancia e indice

    sorted_neighbor_dist_and_ind = sorted(neighbor_dist_and_ind)
    k_nearest_dist_and_ind = sorted_neighbor_dist_and_ind[:k]
    k_nearest_labels = [train[i][0] for distance, i in k_nearest_dist_and_ind]
    x = mean(k_nearest_labels)

    return x

#Função para transformar os dados
def transform_data(df):

    #Remove categorial variables
    columns = ['origin', 'car name', 'model year']

    for element in columns:
        df.drop(element, axis = 1, inplace = True)

    #Substituindo valores nulos pela moda
    df['horsepower'] = df['horsepower'].replace({'?': df['horsepower'].mode().apply(float)[0]}).apply(float)

    #Dividindo a base em treino e validacao
    ind = np.random.rand(len(df)) < 0.8
    train = df[ind] #80% para treino
    validate = df[~ind] #20% para validacao

    #Transformando em objetos
    train = train.values.tolist()
    validate = validate.values.tolist()

    return train, validate #retornando base para traino e validação

if __name__ == '__main__':

    #Passando local do arquivo
    path = LOCAL_FILE
    results = 'Resultados.csv'
    path_results = LOCAL_RESULTS
    chart = 'Resultados.png'

    #Importando base de dados
    df = pd.read_csv(path, sep=';')

    #Adquando base para rodar o modelo
    train, validate = transform_data(df)

    #Valor de K
    k = 3

    #Aplicando a base de validação no modelo para tentar estimar o valor 'mpg' (milhas por galão)
    predictions = [knn_regression(train,validate[element][1:],k) for element in range(len(validate))]

    #Avaliando a assertividade do knn regression
    x = []
    y = []
    for element in range(len(predictions)):
        x.append(predictions[element]) #Guardando estimativas realizadas pelo modelo
        y.append(validate[element][0]) #Guardando valores reais contidos na base de validacao

    df2 = pd.DataFrame({'Estimativa': x, 'Valor Real': y})
    df2.to_csv(path_results + results, sep = ';', index=False, decimal=",") #Exportando resultados pra um csv
    df2.plot(figsize=(10,5)) #Plotando resultados Valor Real vs Estimativa realizada pelo modelo
    plt.savefig(path_results + chart) #Salvando imagem
