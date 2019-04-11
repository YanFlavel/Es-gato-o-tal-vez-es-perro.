import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 9
plt.imshow(train_set_x_orig[index])
print("y = "+str(train_set_y[:,index])   +", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+"picture.")
print("ACTIVIDAD 1")
print("Cantidad de datos de entrenamiento: ",len(train_set_x_orig))
print("Cantidad de datos test", len(test_set_x_orig))
#X_flatten = X.reshape (X.shape [0], -1) .T
#train_set_x_flatten =
print("Forma 2")
m_train=train_set_x_orig.shape[0]
print("# Datos entrenamiento: ",m_train)
m_test=test_set_x_orig.shape[0]
print("# Datos test: ",m_test)
print("Tamaño de la imagen: 64 x 64")
      
#Redimension
train_set_x_orig_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T

test_set_x_orig_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
print("ACTIVIDAD 2")
print("Nuevo tamanio entrenamiento: ",len(train_set_x_orig_flatten))
print("Nuevo tamanio test: ",len(test_set_x_orig_flatten))

#Normalizaciónde datos
train_set_x_orig_flatten=train_set_x_orig_flatten/255
test_set_x_orig_flatten=test_set_x_orig_flatten/255
nump_px=64
#Inicializacion en zeros
print("ACTIVIDAD 3")
print("w=np.zeros(dim)")
print("b=0")
def inicializate_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w,b


t,y=inicializate_with_zeros(64)
#inicializate_with_zeros(10)
print("ACTIVIDAD 4")
print("s= 1 / (1 + np.exp(-z))")
def sigmoid(z):
    s= 1 / (1 + np.exp(-z))

    return s

def propagate(w,b,X,Y):
    m=X.shape[1]
    A=sigmoid( np.dot(w.T,X)+b)
    
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))
    
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,  "db": db}
    return grads,cost


w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=[]
    for i in range(num_iterations):
        grads,cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w= w - dw*learning_rate
        b= b - db*learning_rate
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

        
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

def predict(w,b,X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    Y_DESTINO = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    #Obtener la predicción sin usar for
    Y_DESTINO = np.round(A)
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if (A[0,i] <= 0.5):
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    assert(Y_prediction.shape == (1, m))
    print("Mi prediccion: ",Y_DESTINO)
    return Y_prediction

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))





