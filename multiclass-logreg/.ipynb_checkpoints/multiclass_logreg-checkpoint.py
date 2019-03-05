import numpy as np

class multiclass_logreg:
    def __init__(self, learning_rate, n_iter, k_keep, Xtrain, Ytrain, Xtest, Ytest):
        self.k_keep = k_keep
        self.x = Xtrain
        self.y = Ytrain
        self.X = Xtest
        self.Y = Ytest
        self.x_samples, self.x_features = self.x.shape
        self.weights = np.zeros((self.x_features, self.y.shape[1]))
        self.loss = np.zeros((len(k_keep), self.x_samples))
        self.n_iter = n_iter
        self.s = 1e-2
        self.lr = learning_rate
        self.train_error = []
        self.test_error = []
        self.mu = 100
    
    def lorenz(self, u):
        inds = np.where(u <= 1)
        lorenz_ = np.zeros(u.shape)
        derivative_ = np.zeros(u.shape)
        u = np.asarray(u)
        lorenz_ = (u <= 1)*np.log(1 + (u-1)**2)
        derivative_ = (u <= 1) * (2 * (u-1))/(1 + (u-1)**2)
        return np.asarray(lorenz_), np.asarray(derivative_)
  
    def schedule(self, m, i ,k):
        mi = k + (m-k) * max(0, (500-2*i)/(2*i*100 + 500))
        return mi
    
    def vapnik_loss(self, xx, w, y):
        xx = np.asarray(xx)
        w = np.asarray(w)
        y = np.asarray(y)
        dxby = np.asarray(np.diag(np.dot(np.dot(xx, w), y.T)).reshape(1, self.x_samples))
        tdxby = np.asarray(np.dot(dxby.T, np.ones((1, y.shape[1]))))
        vl = 1/self.x_samples * np.sum(self.lorenz(tdxby - xx @ w)[0]) - np.log(2) + (self.s * (np.linalg.norm(w)))**2
        return vl
    
    def FSA(self, k, m, w, i):
        w = np.asarray(w)
        mi = k + (m-k)*max(0, (self.n_iter - 2 * i)/(2*i*self.mu+self.n_iter))
        inds = np.ravel(sorted(np.argsort(np.linalg.norm(w, axis=1))[-int(mi):]))
        return inds
    
    def softmax(self, x, w):
        x = np.asarray(x)
        w = np.asarray(w)
        top = np.exp(np.dot(x.T, w))
        bottom = np.sum(np.exp(np.dot(x.T, w)))
        return top/bottom
    
    def softmax_predict(self, x, w):
        return np.argmax(self.softmax(x, w))
    
    def gradient(self, xx, w, y):
        xx = np.asarray(xx)
        w = np.asarray(w)
        y = np.asarray(y)
        dxby = np.asarray(np.diag(np.dot(np.dot(xx, w), y.T)).reshape(1, self.x_samples))
        tdxby = np.asarray(np.dot(dxby.T, np.ones((1, y.shape[1]))))
        val1 = np.asarray(np.sum(self.lorenz(tdxby - np.dot(xx, w))[1], axis=1).reshape(self.x_samples, 1))
        val2 = np.asarray(np.add(-self.lorenz(tdxby - np.dot(xx, w))[1], np.multiply(np.dot(val1, np.ones((1, y.shape[1]))), y)))
        grad = np.asarray(1/self.x_samples * np.dot(xx.T, val2) + 2 * self.s * w)
        return grad
    
    def fit_predict(self):
        self.l_ = np.zeros((len(self.k_keep), self.n_iter))
        all_loss = []
        for k in range(len(self.k_keep)):
            weightstmp = self.weights
            Xtmp = self.X
            xtr = self.x
            ytr = self.y
            losses = [0] * self.n_iter
            tr_mis_count, test_mis_count = 0, 0
            for i in range(self.n_iter):
                losses[i] = self.vapnik_loss(xtr, weightstmp, ytr)
                weightstmp = weightstmp - self.lr * self.gradient(xtr, weightstmp, ytr)
                inds = self.FSA(k, self.x_features, weightstmp, i)
                weightstmp = weightstmp[inds]
                Xtmp = Xtmp[:, inds]
                xtr = xtr[:, inds]
                
                if self.softmax_predict(xtr[i], weightstmp) != np.argmax(self.y[i]):
                    tr_mis_count += 1
                    
                if self.softmax_predict(Xtmp[i], weightstmp) != np.argmax(self.Y[i]):
                    test_mis_count += 1
                
                if i % 100 == 0:
                    print('Training step:', i, '\tLoss:', losses[i])

            all_loss.append(losses)
            print('Training misclassification for k =', self.k_keep[k], ':', tr_mis_count/self.x_samples)
            print('Testing misclassification for k =', self.k_keep[k], ':', test_mis_count/self.x_samples)
            self.train_error.append(tr_mis_count/self.x_samples)
            self.test_error.append(test_mis_count/self.x_samples)
        
        return self.train_error, self.test_error, all_loss