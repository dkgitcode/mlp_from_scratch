# Multi Layer Perceptron from Scratch
A multi layer perceptron trained on Cencus Income dataset using mini batch gradient descent.

## Dataset
The dataset used is the Cencus Income dataset from the UCI Machine Learning Repository. The dataset contains 14 features and 1 target variable. The target variable is binary and indicates whether the income of an individual is greater than $50,000 or not.

### Data Preparation

Before training on the data, we first shuffle, split, normalize, and one hot encode the data. We shuffle the data to prevent any bias in the training process. We split the data into training and testing sets to evaluate the model's performance. We normalize the data to ensure that the model converges faster. We one hot encode the target variable to convert it into a binary format.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def prep_data(X_train, X_test, y_train, y_test):
    # Combine train and test set to ensure same dummy variables
    X_combined = pd.concat([X_train, X_test])
    
    # Apply get_dummies to the combined dataset
    X_combined = pd.get_dummies(X_combined, drop_first=True)
    
    # Split the combined dataset back into train and test sets
    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]
    
    # Z-score normalization
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    # Handle potential NaN values resulting from normalization
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Convert categorical target variable y to binary (0/1)
    y_train = (y_train == '>50K').astype(int)
    y_test = (y_test == '>50K').astype(int)

    # Convert data to np arrays
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prep_data(X_train, X_test, y_train, y_test)
```

## Model Specification

Input (100, 64) $\rightarrow$ ReLU $\rightarrow$ Hidden Layer (64, 32) $\rightarrow$ ReLU $\rightarrow$ Output(32, 1) $\rightarrow$ Sigmoid


## Training

### Loss Function
We use binary cross entropy loss as the loss function.

$$
\begin{equation}
L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)
\end{equation}
$$

### Optimization

We use mini batch gradient descent to optimize the weights of the model. The model is trained for 100 epochs with a batch size of 32.

## Gradient Descent Calculus

### Forward Pass
Our forward pass looks as such,  where $X$ is the input, $W$ is the weight matrix, $b$ is the bias, $Z$ is the linear transformation, $A$ is the activation output, and $L$ is the loss function. $X$ is of shape (batch_size, input_size), $W$ is of shape (input_size, output_size), $b$ is of shape (1, output_size), $Z$ is of shape (batch_size, output_size), $A$ is of shape (batch_size, output_size), and $L$ is a scalar. 

### Stochastic Gradient Descent vs Mini Batch Gradient Descent

In stochastic gradient descent, we calculate the loss for each sample in the dataset and update the weights and biases after each sample. In standard gradient descent we compute the loss for the entire dataset and update the weights and biases after each epoch.

However, in mini batch gradient descent, we calculate the loss for a batch of samples and update the weights and biases after each batch.


$$
\begin{equation}
Z_1 = XW_1 + b_1
\end{equation}
$$

$$
\begin{equation}
A_1 = ReLU(Z_1)
\end{equation}
$$

$$
\begin{equation}
Z_2 = A_1W_2 + b_2
\end{equation}
$$


<!-- A2 -->
$$
\begin{equation}
A_2 = ReLU(Z_2)
\end{equation}
$$

<!-- z3 -->

$$
\begin{equation}
Z_3 = A_2W_3 + b_3
\end{equation}
$$



$$
\begin{equation}
A_3 = Sigmoid(Z_2)
\end{equation}
$$

$$
\begin{equation}
L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)
\end{equation}
$$


### Backward Pass

We now will calculate the gradients of the loss function with respect to the weights and biases of the model in order to update them using gradient descent.

$$
\begin{equation}
\frac{\partial L}{\partial W_3} = \frac{\partial L}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial W_3}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial L}{\partial b_3} = \frac{\partial L}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial b_3}
\end{equation}
$$

$$ ... $$


$$
\begin{equation}
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial W_1}
\end{equation}
$$

As you can see, the chain rule produces a "chain" of gradients. This equation begins to get increasingly complex as the number of layers in the network increases.

So, after we propagate backwards through the network, we pass back $$\frac{\partial L}{\partial A_i}$$ to the previous layer so we can keep track of this "chain" of gradients and propagate it back through the network.

### Code for Backward Pass

```python
def backward(self, dL_dA):
        """
        Calculates the gradient of the loss with respect to weights, biases, and the previous layer's activations.
        :param dL_dA: The gradient of the loss with respect to the layer's output.
        :return dL_dA_prev: The gradient of the loss with respect to the previous layer's activations.
        """
        if self.activation_function is None:
            self.dL_dz = dL_dA
        else:
            self.dL_dz = dL_dA * self.activation_prime(self.z)
        # Remember, z = w * a + b, so dz/dw = a
        self.dL_dW = np.dot(self.inputs.T, self.dL_dz) # dL/dw = dL/da * da/dz * dz/dw
        self.dL_dB = np.sum(self.dL_dz, axis=0, keepdims=True) # dL/db = dL/da * da/dz * dz/db
        
        self.dL_dA_prev = np.dot(self.dL_dz, self.weights.T) # dL/dz * dz/dA[L-1]
        
        return self.dL_dA_prev
```

### Update Weights and Biases

After calculating the gradients of the loss function with respect to the weights and biases, we update the weights and biases using the following equations:

$$
\begin{equation}
W = W - \alpha \cdot \frac{\partial L}{\partial W}
\end{equation}
$$

$$
\begin{equation}
b = b- \alpha \cdot \frac{\partial L}{\partial b}
\end{equation}
$$

Where $\alpha$ is the learning rate.

Note: We also decay the learning rate in this example to prevent the model from overshooting the minimum.

```python
    learning_rate = initial_learning_rate * (1 / (1 + 0.01 * epoch))
```
### Gradient Clipping

After backpropagation, we also clip the gradients of each layer to prevent exploding gradients. I added this after training as I ran into issues where the gradients were exploding.

```python
def clip_gradients(self, max_norm):
        """
        Clips the gradients to prevent exploding gradients using L2 norm clipping.
        :param max_norm: The maximum allowable norm for the gradients.
        """
        total_norm = np.linalg.norm(self.dL_dW) # Calculate the L2 norm of the gradients
        if total_norm > max_norm:
            self.dL_dW = self.dL_dW * (max_norm / total_norm) # Rescale the gradients
```

We calculate the total norm of the gradients and if it exceeds a certain threshold, we rescale the gradients to prevent them from exploding. 

The norm of the gradients is calculated using the L2 norm. The L2 norm of a vector is the square root of the sum of the squares of the vector's elements.

## Results

After training this model on 10 epochs, with a batch size of 64, and an initial learning rate of 0.01, we achieve around an 84% accuracy on the test set.
