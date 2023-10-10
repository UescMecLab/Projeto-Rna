import numpy as np
from network import Network
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats as st

def DenormalizeData(Data, Orig):
  return (Data * Orig.std() + Orig.mean())

def PlotCosts(fig, Hist):
  ax = fig.add_subplot(1, 2, 1)
  k = np.linspace(0, len(Hist), len(Hist), dtype=int)
  ax.plot(k, Hist)
  ax.set_xlabel('Iterações (k)')
  ax.set_ylabel('Custo')
  ax.set_title("Histórico de custo da Função MSE")
  ax.grid(True)

def PlotTrainingResult(fig, network):
  y_pred = network.predict(X)
  ax = fig.add_subplot(1, 2, 2, projection='3d')
  ax.scatter(X[:, 0], X[:, 1], Y, c='blue', marker='o', label='Dados Reais')
  ax.scatter(X[:, 0], X[:, 1], y_pred, c='red', marker='^', label='Previsões da Rede Neural')
  ax.set_xlabel('X1')
  ax.set_ylabel('X2')
  ax.set_zlabel('Saída Esperada (y)')
  ax.legend()

def CreateNetwork(w_h=None, b_h=None, w_o=None, b_o=None, up=True):
    network = Network()
    hiddenLayer = network.add(2, 2, 'tanh')
    outputLayer = network.add(2, 1, 'linear')
    if (up == True):
        updateNetwork(hiddenLayer, outputLayer, w_h, b_h, w_o, b_o)
    print("W_h: ", (hiddenLayer.get_weights()).T, "B_h:", (hiddenLayer.get_bias()).T)
    print("W_o: ", (outputLayer.get_weights()).T, "B_o:", (outputLayer.get_bias()).T)
    return (network)

def updateNetwork(hiddenLayer, outputLayer, w_h, b_h, w_o, b_o):
    hiddenLayer.set_weights(w_h)
    hiddenLayer.set_bias(b_h)
    outputLayer.set_weights(w_o)
    outputLayer.set_bias(b_o)

def TestNetwork(network, learning_rate=0.001, epochs=10, print_epochs=False):
    fig = plt.figure(figsize=(12, 6))
    network.train(X, Y, epochs, learning_rate, print_epochs)
    PlotCosts(fig, network.HistCost)
    PlotTrainingResult(fig, network)

if __name__ == '__main__':
    path = "/home/smodesto/uesc/Ic/Projeto-Rna/Fundamentos-de-RNAS/Trabalho6/dados/Trabalho5dados.xlsx"
    df = pd.read_excel(path)
    x1 = df['x1']
    x2 = df['x2']
    y = df['y']

    XDenorm = np.vstack([x1, x2]).T
    X = st.zscore(XDenorm)
    Y = st.zscore(y)
    net = CreateNetwork(
                        w_h=np.array([[1.0, 1.0], [1.0, 1.0]]), 
                        b_h=np.array([1.0 , 1.0]),
                        w_o=np.array([[1.0, 1.0]]),
                        b_o=np.array([1.0]))
    TestNetwork(net, print_epochs=True)