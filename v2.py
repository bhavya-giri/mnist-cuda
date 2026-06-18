import numpy as np
import time

#Load binary data
X_train = np.fromfile('data/X_train.bin', dtype=np.float32).reshape(60000,28*28)[:10000]
y_train = np.fromfile('data/y_train.bin', dtype=np.int32)[:10000]
X_test = np.fromfile('data/X_test.bin',dtype=np.float32).reshape(10000,28*28)
y_test = np.fromfile('data/y_test.bin', dtype=np.int32)

# Apply MNIST normalization
mean, std = 0.1307, 0.3081
X_train_np = (X_train - mean) / std
X_test_np = (X_test - mean) / std

# reshape to (N,1,28,28) format
X_train = X_train_np.reshape(-1,1,28,28)
X_test = X_test_np.reshape(-1,1,28,28)

# Activation functions
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

# Linear Layer
def initialize_weights(input_size,output_size):
    scale = np.sqrt(2.0/input_size)
    return (np.random.rand(input_size,output_size)*2.0 - 1.0) * scale

def initialize_bias(output_size):
    return np.zeros((1,output_size))

def linear_forward(x,w,b):
    return x @ w + b

def linear_backward(grad_output,x,w):
    grad_w= x.T @ grad_output
    grad_b = np.sum(grad_output,axis=0,keepdims=True)
    grad_input = grad_output @ w.T
    return grad_input, grad_w, grad_b

# softmax and cross entropy loss
def softmax(x):
    exp_x = np.exp(x-np.max(x,axis=1,keepdims=True))
    return exp_x / np.sum(exp_x,axis=1, keepdims=True)

def cross_entropy_loss(y_pred,y_true):
    batch_size = y_pred.shape[0]
    probabilities = softmax(y_pred)
    correct_log_probs = -np.log(probabilities[np.arange(batch_size),y_true])
    loss = np.sum(correct_log_probs)/batch_size
    return loss

class NeuralNetwork:
    def __init__(self, input_size, hidden_size,output_size):
        # initialize weights and biases
        self.W1 = initialize_weights(input_size, hidden_size)
        self.b1 = initialize_bias(hidden_size)
        self.W2 = initialize_weights(hidden_size, output_size)
        self.b2 = initialize_bias(output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size,-1)
        # forward pass
        z1 = linear_forward(x, self.W1, self.b1)
        a1 = relu(z1)
        z2 = linear_forward(a1, self.W2, self.b2)
        return z2, (x,z1,a1)
    
    def backward(self,grad_output,cache):
        x,z1,a1 = cache
        
        grad_a1, grad_W2, grad_b2 = linear_backward(grad_output,a1,self.W2)
        grad_z1 = grad_a1 * relu_derivative(z1)
        grad_x, grad_W1, grad_b1 = linear_backward(grad_z1,x,self.W1)
        return grad_W1, grad_b1, grad_W2, grad_b2
    
    def update_weights(self,grad_W1,grad_b1,grad_w2,grad_b2,learning_rate):
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        self.W2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
    

def train_timed(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate):
    # Initialize timing stats
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
        'total_time': 0.0
    }
    
    # Start total timing
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, len(X_train), batch_size):
            # Data loading timing
            data_start = time.time()
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            data_end = time.time()
            timing_stats['data_loading'] += data_end - data_start
            
            # Forward pass timing
            forward_start = time.time()
            y_pred, cache = model.forward(batch_X)
            forward_end = time.time()
            timing_stats['forward'] += forward_end - forward_start
            
            # Loss computation timing
            loss_start = time.time()
            loss = cross_entropy_loss(y_pred, batch_y)
            epoch_loss += loss

            softmax_probs = softmax(y_pred)
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[np.arange(len(batch_y)), batch_y] = 1
            grad_output = (softmax_probs - y_true_one_hot) / len(batch_y)
            loss_end = time.time()
            timing_stats['loss_computation'] += loss_end - loss_start

            # Backward pass timing
            backward_start = time.time()
            grad_weights1, grad_bias1, grad_weights2, grad_bias2 = model.backward(grad_output, cache)
            backward_end = time.time()
            timing_stats['backward'] += backward_end - backward_start
            
            # Weight updates timing
            update_start = time.time()
            model.update_weights(grad_weights1, grad_bias1, grad_weights2, grad_bias2, learning_rate)
            update_end = time.time()
            timing_stats['weight_updates'] += update_end - update_start
            num_batches += 1

        print(f"Epoch {epoch} loss: {epoch_loss / num_batches:.4f}")

    # End total timing
    total_end = time.time()
    timing_stats['total_time'] = total_end - total_start
    
    # Print detailed timing breakdown
    print("\n=== PYTHON NUMPY IMPLEMENTATION TIMING BREAKDOWN ===")
    print(f"Total training time: {timing_stats['total_time']:.1f} seconds\n")
    
    print("Detailed Breakdown:")
    print(f"  Data loading:     {timing_stats['data_loading']:6.3f}s ({100.0 * timing_stats['data_loading'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Forward pass:     {timing_stats['forward']:6.3f}s ({100.0 * timing_stats['forward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Loss computation: {timing_stats['loss_computation']:6.3f}s ({100.0 * timing_stats['loss_computation'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Backward pass:    {timing_stats['backward']:6.3f}s ({100.0 * timing_stats['backward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Weight updates:   {timing_stats['weight_updates']:6.3f}s ({100.0 * timing_stats['weight_updates'] / timing_stats['total_time']:5.1f}%)")
    
    print("Training completed!")

if __name__ == "__main__":
    # Use random seed for natural variance (no fixed seed)
    
    input_size = 784  # 28x28 pixels
    hidden_size = 1024
    output_size = 10  # 10 digits
    
    model = NeuralNetwork(input_size, hidden_size, output_size)
    
    batch_size = 32
    epochs = 10
    learning_rate = 0.01
    
    train_timed(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate)