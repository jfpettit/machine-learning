import numpy as np

class hidden_markov_model:
    def __init__(self, start_probs, transition_probs, emission_probs, state_set, observations):
        self.start = start_probs
        self.transition = transition_probs
        self.emission = emission_probs
        self.states = state_set
        self.obs = observations
        self.n_states = len(self.states)
        self.t_obs = len(observations)
        self.alpha = np.zeros((self.t_obs, self.n_states))
        self.beta = np.zeros((self.t_obs, self.n_states))
        self.underflow = np.ones(len(self.obs))
    
    def get_gammas(self, alphas, betas, T, N):
        gammas = np.zeros((len(self.obs)-1, self.n_states))
        xs = np.zeros((len(self.obs)-1, self.n_states, self.n_states))
        for t in range(self.t_obs-1):
            xinum = np.array([self.alpha[t]]*2).T * self.transition * np.array([self.beta[t+1]]*2)*np.array([self.emission[:, self.obs[t]-1]]*2)
            xs[t] = xinum/np.sum(xinum)
            gammas[t] = np.sum(xs[t], axis=1)
        
        return gammas, xs
        
    def EM(self, N, M, T, pi, A, B):
        alphas = self.forward()
        betas = self.backward()
        
        gammas, xs = self.get_gammas(alphas, betas, T, N)
        
        new_pi = gammas[0, :]
        
        new_a = np.sum(xs, axis=0)/(np.array([np.sum(gammas, axis=0)]*2).T)
       
        xinds = np.zeros((len(self.obs), len(self.emission[0])))
        xinds[np.arange(len(self.obs)), self.obs-1] = 1
        
        new_b = (gammas.T@xinds[:-1])/np.array([np.sum(gammas, axis=0)]*6).T
        
        return new_pi, new_a, new_b
        
    def baumwelch(self, tolerance, maxiter):
        T = self.t_obs
        N = self.n_states
        M = len(self.states)
        likely = []
        
        pi = np.log(self.start)
        A = np.log(self.transition)
        B = np.log(self.emission)
        
        iter_ = 0
        err = np.inf
        while iter_ < maxiter or err > tolerance:
            iter_ += 1
            if iter_ % 100 == 0:
                print('Iter:', iter_)
            new_pi, new_a, new_b = self.EM(N, M, T, pi, A, B)
            
            err = np.abs(np.linalg.norm(A - new_a) + np.linalg.norm(B - new_b))/2
            
            pi, A, B = new_pi, new_a, new_b
            
        return pi, A, B
    
    def forward(self):
        self.alpha[0] = self.start * self.emission[:, self.obs[0]-1]
        for t in range(1, self.t_obs):
            for k in self.states:
                self.alpha[t, k] = (self.alpha[t-1]@self.transition[k]) * self.emission[k, self.obs[t]-1]
            self.underflow[t] = np.sum(self.alpha[t])
            self.alpha[t] = self.alpha[t]/self.underflow[t]
        return self.alpha
    
    def backward(self):
        self.beta[:, -1:] = 1
        
        for t in range(2, len(self.obs)+1):
            for k in self.states:
                self.beta[-t, k] = np.sum(self.beta[-t+1] * self.transition[k] * self.emission[:, self.obs[-t+1]-1])
            self.beta[-t] = self.beta[-t]/self.underflow[-t]
        return self.beta
    
    def viterbi(self):
        N = self.transition.shape[0]
        delta = np.zeros((self.t_obs, N))
        psi = np.zeros((self.t_obs, N))
        delta[0] = self.start * self.emission[:, self.obs[0]-1]
        for tt in range(1, self.t_obs):
            for j in range(N):
                delta[tt, j] = np.max(delta[tt-1] * self.transition[:, j] * self.emission[j, self.obs[tt]-1])
                psi[tt, j] = np.argmax(delta[tt-1] * self.transition[:, j])
                
        states = np.zeros(self.t_obs)
        states[self.t_obs-1] = np.argmax(np.exp(delta[self.n_states-1]))
        states = states.astype('int')
        for t in range(self.t_obs-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states
    
    