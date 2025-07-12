from typing import List, Optional, Tuple
import numpy as np
import copy
from cards import load_cards_from_csv
import json 


class NeuralNetwork:

    def __init__(self, layer_dims : List[int], learning_rate : float=0.01) -> None:
        self.learning_rate = learning_rate
        self.layer_dims : List[int] = layer_dims
        self.layers : List = []
        self.L = len(layer_dims) 

        self.Ws = []
        self.bs = []

        for l in range(1, len(layer_dims)):
            self.Ws.append(np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01)
            self.bs.append(np.zeros((layer_dims[l], 1)))

    @staticmethod
    def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache
    
    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    @staticmethod
    def linear_activation_forward_relu(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        Z, linear_cache = NeuralNetwork.linear_forward(A_prev, W, b)
        A = NeuralNetwork.relu(Z)
        activation_cache = Z
        cache = (linear_cache, activation_cache)
        return A, cache
    
    @staticmethod
    def linear_activation_forward_sigmoid(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        Z, linear_cache = NeuralNetwork.linear_forward(A_prev, W, b)
        A = 1 / (1 + np.exp(-Z))

        activation_cache = Z
        cache = (linear_cache, activation_cache)
        return A, cache

    @staticmethod
    def compute_cost(AL: np.ndarray, Y: np.ndarray) -> float:
        m = Y.shape[1]

        logprobs = (np.multiply(np.log(AL),Y) + np.multiply(1-Y, np.log(1-AL)))
        cost = -1/m * np.sum(logprobs)

        return np.squeeze(cost)
    
    def model_forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        caches = []
        A = X
        L = len(self.Ws)                  

        for l in range(1, L):
            A_prev = A 
            A, cache = NeuralNetwork.linear_activation_forward_relu(A_prev, self.Ws[l-1], self.bs[l-1])
            caches.append(cache)
            
        # End in sigmoid activation for binary classification
        AL, cache = NeuralNetwork.linear_activation_forward_sigmoid(A, self.Ws[L-1], self.bs[L-1])
        caches.append(cache)
            
        return AL, caches
    
    @staticmethod
    def linear_backward(dZ : np.ndarray, cache : Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A_prev, W, b = cache
        m = A_prev.shape[1]
       
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db
    
    @staticmethod
    def linear_activation_backward_relu(dA: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        linear_cache, activation_cache = cache
        Z = activation_cache
        dZ = dA * (Z > 0)
        return NeuralNetwork.linear_backward(dZ, linear_cache)
    
    @staticmethod
    def linear_activation_backward_sigmoid(dA: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        linear_cache, activation_cache = cache
        Z = activation_cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return NeuralNetwork.linear_backward(dZ, linear_cache)
    
    def model_backward(self, AL: np.ndarray, Y: np.ndarray, caches: List[Tuple]) -> any:
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) 
        
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
        
        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = NeuralNetwork.linear_activation_backward_sigmoid(dAL, current_cache)
        grads['dA' + str(L-1)] = dA_prev_temp
        grads['dW' + str(L)] = dW_temp
        grads['db' + str(L)] = db_temp

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = NeuralNetwork.linear_activation_backward_relu(dA_prev_temp, current_cache)
            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l+1)] = dW_temp
            grads['db' + str(l+1)] = db_temp

        return grads
    
    def update_parameters(self, grads : any) -> None:
        # assert self.L == len(self.Ws) == len(self.bs), "Layer dimensions do not match the number of parameters."

        # print(grads.keys())

        for l in range(1, self.L):
            # print(l)
            self.Ws[l-1] -= self.learning_rate * grads['dW' + str(l)]
            self.bs[l-1] -= self.learning_rate * grads['db' + str(l)]

    def train(self, X : np.ndarray, Y : np.ndarray, num_iterations : int = 3000) -> List[float]:
        costs = []
        for i in range(0, num_iterations):

            AL, caches = self.model_forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.model_backward(AL, Y, caches)
            self.update_parameters(grads)

            if i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0:
                costs.append(cost)
        
        return costs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        AL, _ = self.model_forward(X)
        predictions = (AL > 0.5).astype(int)
        return predictions

    def save(self, filename: str) -> None:
        
        state = {"layer_dims": self.layer_dims}

        # Can't directly save Ws/bs as they are unequal length lists
        for i in range(len(self.layer_dims) - 1):
            state[f'W{i+1}'] = self.Ws[i].tolist()
            state[f'b{i+1}'] = self.bs[i].tolist()

        np.savez(filename, **state, allow_pickle=True)
        
        
    @staticmethod
    def load(filename: str) -> 'NeuralNetwork':
        data = np.load(filename, allow_pickle=True)
        layer_dims = data['layer_dims'].tolist()

        nn = NeuralNetwork(layer_dims)

        nn.Ws = []
        nn.bs = []

        for i in range(len(layer_dims) - 1):
            nn.Ws.append(np.array(data[f'W{i+1}']))
            nn.bs.append(np.array(data[f'b{i+1}']))

        return nn

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)