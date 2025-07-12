from cards import load_cards_from_csv
import numpy as np
from typing import List, Tuple
from nn import NeuralNetwork
import random 

print("Loading cards from CSV...")
cards = load_cards_from_csv(count=-1)
random.shuffle(cards)  
print(f"Loaded {len(cards)} cards.")

n_training = len(cards) // 2

X_train = np.array([card.image_data for card in cards[:n_training]]).T.squeeze()
Y_train = np.array([1 if card.isHeroism() else 0 for card in cards[:n_training]]).reshape(1, n_training)
X_test =  np.array([card.image_data for card in cards[n_training:]]).T.squeeze()
Y_test =  np.array([1 if card.isHeroism() else 0 for card in cards[n_training:]]).reshape(1, len(cards) - n_training)

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

layer_dims = [X_train.shape[0], 128, 32, 1]
print(f"{layer_dims=}")

nn = NeuralNetwork(layer_dims, learning_rate=0.0001)
costs = nn.train(X_train, Y_train, num_iterations=1000)
print(costs)

nn.save('heroism_model.npz')

predictions = nn.predict(X_test)
accuracy = np.mean(predictions == Y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

nn = NeuralNetwork.load('heroism_model.npz')
# print("Model loaded successfully.")

predictions = nn.predict(X_test)
accuracy = np.mean(predictions == Y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
