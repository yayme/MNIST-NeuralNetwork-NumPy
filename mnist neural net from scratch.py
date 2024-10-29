import numpy as np
import pandas as pd
from keras._tf_keras.keras.datasets import mnist
from statsmodels.stats.sandwich_covariance import weights_uniform

data=mnist.load_data();
(X_train,y_train),(X_test,y_test)=data;
# flattening each image
X_train=X_train.reshape(X_train.shape[0],-1)
X_test=X_test.reshape(X_test.shape[0],-1)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
# normalizing the pixels
X_train=X_train/255.0
X_test=X_test/255.0
# Neural Network part
class NeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size):
        # initializing weights
        self.weights_input_hidden=np.random.randn(input_size,hidden_size)*np.sqrt(2.0/input_size)
        self.weights_hidden_output=np.random.randn(hidden_size,output_size)*np.sqrt(2.0/input_size)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def sigmoid_derivative(self,z):
        return z*(1-z);
    def forward(self,X):
        self.hidden_layer_input=np.dot(X,self.weights_input_hidden)
        self.hidden_layer_output=self.sigmoid(self.hidden_layer_input)
        self.output_layer_input=np.dot(self.hidden_layer_output,self.weights_hidden_output)
        self.output=self.sigmoid(self.output_layer_input)
        return self.output

    def backward(self,X,y,output,learning_rate):
        output_error=y-output
        output_delta=output_error*self.sigmoid_derivative(output)
        hidden_layer_error=output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta=hidden_layer_error*self.sigmoid_derivative(self.hidden_layer_output)
        # now we update weights
        self.weights_hidden_output+=self.hidden_layer_output.T.dot(output_delta)*learning_rate
        self.weights_input_hidden+=X.T.dot(hidden_layer_delta)*learning_rate



def one_hot_encode(y,num_classes):
    return np.eye(num_classes)[y]

# now we train our neural net
input_size=784 #28*28 pixels
hidden_size=128
output_size=10
learning_rate=0.1
epochs=50
batch_size=32
nn=NeuralNetwork(input_size,hidden_size,output_size)
y_train_encoded=one_hot_encode(y_train,output_size)
for epoch in range(epochs):
    for i in range(0,X_train.shape[0],batch_size):
        X_batch=X_train[i:i+batch_size]
        y_batch=y_train_encoded[i:i+batch_size]

        output=nn.forward(X_batch)
        nn.backward(X_batch,y_batch,output,learning_rate)
    if epoch%10==0:
        loss=np.mean(np.square(y_train_encoded-nn.forward(X_train)))
        print(f"epoch {epoch}, loss: {loss}")

# now we do evaluation
def predict(X):
    output=nn.forward(X)
    return np.argmax(output,axis=1)
predictions=predict(X_test)
accuracy=np.mean(predictions==y_test)*100
print(f"accuracy: {accuracy: .2f}%")






