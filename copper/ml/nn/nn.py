class NN_1HL(object):
    
    def __init__(self, reg_lambda=0, epsilon_init=0.12, hidden_layer_size=25, opti_method='TNC'):
        self.reg_lambda = reg_lambda
        self.epsilon_init = epsilon_init
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = self.sigmoid
        self.activation_func_prime = self.sigmoid_prime
        self.method = opti_method
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def sumsqr(self, a):
        return np.sum(a ** 2)
    
    def rand_init(self, l_in, l_out):
        return np.random.rand(l_out, l_in + 1) * 2 * self.epsilon_init - self.epsilon_init
    
    def pack_thetas(self, t1, t2):
        return np.concatenate((t1.reshape(-1), t2.reshape(-1)))
    
    def unpack_thetas(self, thetas, input_layer_size, hidden_layer_size, num_labels):
        t1_start = 0
        t1_end = hidden_layer_size * (input_layer_size + 1)
        t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, input_layer_size + 1))
        t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))
        return t1, t2
    
    def _forward(self, X, t1, t2):
        m = X.shape[0]
        
        ones = np.ones(m).reshape(m,1) if len(X.shape) > 1 else np.array([1])
        # Input layer
        a1 = np.append(ones, X, axis=1)  # [5000, 401]
        
        # Hidden layer
        z2 = np.dot(t1, a1.T)  # [25, 5000]
        a2 = self.activation_func(z2)  # [25, 5000]
        
        a2 = np.append(ones.T, a2, axis=0)  # [26, 5000]
        
        # Output layer
        z3 = np.dot(t2, a2)  # [10, 5000]
        a3 = self.activation_func(z3)  # [10, 5000]
        return a1, z2, a2, z3, a3
    
    def function(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)
        
        _, _, _, _, h = self._forward(X, t1, t2)
        
        m = X.shape[0]
        Y = np.eye(num_labels)[y]
        
        costPositive = -Y * np.log(h).T
        costNegative = (1 - Y) * np.log(1 - h).T
        cost = costPositive - costNegative
        J = np.sum(cost) / m
        
        t1f = t1[:, 1:]
        t2f = t2[:, 1:]
        reg = (self.reg_lambda / (2 * m)) * (self.sumsqr(t1f) + self.sumsqr(t2f))
        
        return J + reg
        
    def function_prime(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)
        
        m = X.shape[0]
        t1f = t1[:, 1:]
        t2f = t2[:, 1:]
        Y = np.eye(num_labels)[y]
        
        Delta1, Delta2 = 0, 0
        for i, row in enumerate(X):
            a1, 2, a2, z3, a3 = self._forward(row, t1, t2)
            
            # Backprop
            d3 = a3 - Y[i, :].T
            d2 = np.dot(t2f.T, d3) * self.activation_func_prime(z2)
            
            Delta2 += np.dot(d3[np.newaxis].T, a2[np.newaxis])  # [10, 26]
            Delta1 += np.dot(d2[np.newaxis].T, a1[np.newaxis])  # [25, 401]
            
        Theta1_grad = (1 / m) * Delta1  # [25, 401]
        Theta2_grad = (1 / m) * Delta2  # [10, 26]
        
        Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (reg_lambda / m) * t1f
        Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (reg_lambda / m) * t2f
        
        return self.pack_thetas(Theta1_grad, Theta2_grad)
    
    def fit(self, X, y):
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        hidden_layer_size = self.hidden_layer_size
        num_labels = len(set(y))
        
        theta1_0 = self.rand_init(input_layer_size, hidden_layer_size)
        theta2_0 = self.rand_init(hidden_layer_size, num_labels)
        thetas0 = np.concatenate((theta1_0.reshape(-1), theta2_0.reshape(-1)))
        
        options = {'maxiter': 100}
        _res = optimize.minimize(self.function, thetas0, jac=self.function_prime, method=self.method, 
                                 args=(input_layer_size, hidden_layer_size, num_labels, X, y, 0), options=options)
        
        self.t1, self.t2 = self.unpack_thetas(_res.x, input_layer_size, hidden_layer_size, num_labels)
    
    def predict(self, X):
        # _forward() contains the probalities, so take the max of each column
        return self.predict_proba(X).argmax(0)
    
    def predict_proba(self, X):
        _, _, _, _, h = self._forward(X, self.t1, self.t2)
        return h