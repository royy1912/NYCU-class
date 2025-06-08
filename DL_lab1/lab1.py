import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 強制使用 TkAgg 介面

def sigmoid(x):
    #return x
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    #return 1
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-12  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
def generate_linear(n=100):  
    pts = np.random.uniform(0 , 1 , (n , 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0] , pt[1]])
        distance = (pt[0]-pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i , 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i , 1-0.1*i])
        labels.append(1)

    return np.array(inputs) , np.array(labels).reshape(21 , 1)

def plot_data(inputs, labels, title, filename):
    inputs = np.array(inputs)
    labels = np.array(labels).flatten()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap='bwr', edgecolors='k')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.colorbar(label="Label")


    plt.savefig(filename, dpi=300)  
    plt.close()  
    print(f"Saved plot as {filename}")

def plot_comparison(y_true, y_pred):
    for i in range(len(y_true)):
        print(f"Iter{i}  Ground truth: {y_true[i][0]}  prediction: {y_pred[i][0]:.4f}")
    accuracy = np.mean((y_pred > 0.5).astype(int) == y_true) * 100
    epsilon = 1e-12  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    print(f"Loss: {loss:.4f}  Accuracy: {accuracy:.1f}%")

def test_neural_network(X_test, y_test, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X_test, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    plot_comparison(y_test, A3)

def plot_learning_curve(losses , index):
    plt.plot(range(len(losses)), losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Loss vs. Epoch)")
    plt.legend()
    plt.savefig(f"learning_curve_{index}.png", dpi=300)
    plt.close()
    print("Saved plot as learning_curve.png")

def train(X_train , y_train , X_test , y_test , index):    
    losses_1 = []
    
    input_size = 2
    hidden_size_1 = 4
    hidden_size_2 = 4
    output_size = 1
    learning_rate = 0.1
    epochs = 5000

    
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size_1)
    b1 = np.zeros((1, hidden_size_1))
    W2 = np.random.randn(hidden_size_1, hidden_size_2)
    b2 = np.zeros((1, hidden_size_2))
    W3 = np.random.randn(hidden_size_2, output_size)
    b3 = np.zeros((1, output_size))

    
    for epoch in range(epochs):
        # Forward Pass
        Z1 = np.dot(X_train, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = sigmoid(Z3)
        
        
        loss = -np.mean(y_train * np.log(A3 + 1e-12) + (1 - y_train) * np.log(1 - A3 + 1e-12))
        losses_1.append(loss)
        # Backpropagation
        dA3 = cross_entropy_derivative(y_train , A3)
        dZ3 = dA3 * sigmoid_derivative(A3)
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * sigmoid_derivative(A2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = np.dot(X_train.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    
    predictions = (A3 > 0.5).astype(int)
    plot_data(X_train, predictions, "Trained Neural Network Output", f"nn_output_{index}.png")


    print("Testing Neural Network...")
    test_neural_network(X_test, y_test, W1, b1, W2, b2, W3, b3)

    plot_learning_curve(losses_1 , index)

def train_1(X_train , y_train , X_test , y_test , index):    
    losses_1 = []
    
    input_size = 2
    hidden_size_1 = 64
    hidden_size_2 = 64
    output_size = 1
    learning_rate = 0.01
    epochs = 10000

    
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size_1)
    b1 = np.zeros((1, hidden_size_1))
    W2 = np.random.randn(hidden_size_1, hidden_size_2)
    b2 = np.zeros((1, hidden_size_2))
    W3 = np.random.randn(hidden_size_2, output_size)
    b3 = np.zeros((1, output_size))

    
    for epoch in range(epochs):
        # Forward Pass
        Z1 = np.dot(X_train, W1) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = tanh(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = sigmoid(Z3)
        
        
        loss = -np.mean(y_train * np.log(A3 + 1e-12) + (1 - y_train) * np.log(1 - A3 + 1e-12))
        losses_1.append(loss)
        # Backpropagation
        dA3 = cross_entropy_derivative(y_train , A3)
        dZ3 = dA3 * sigmoid_derivative(Z3)
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * tanh_derivative(A2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * tanh_derivative(A1)
        dW1 = np.dot(X_train.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    
    predictions = (A3 > 0.5).astype(int)
    plot_data(X_train, predictions, "Trained Neural Network Output", f"nn_output_{index}.png")


    print("Testing Neural Network...")
    test_neural_network(X_test, y_test, W1, b1, W2, b2, W3, b3)

    plot_learning_curve(losses_1 , index)


def main():
    
    X_train, y_train = generate_linear()
    X_test, y_test = generate_linear()
    train(X_train , y_train , X_test , y_test , 1)
    
    X_train, y_train = generate_XOR_easy()
    X_test, y_test = generate_XOR_easy()
    train(X_train , y_train , X_test , y_test , 2)
    


if __name__ == "__main__":
    main()