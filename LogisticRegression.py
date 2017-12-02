import numpy as np

#numpy.arrayなら*,/は要素ごと
#sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x[0]))

"""
Compute variational parameters.
"""

def VI(Y, X, M, Sigma_w, alpha, max_iter):
    def rho2sig(rho):
        return np.log(1+np.exp(rho))
    
    def compute_df_dw(Y,X,Sigma_w,mu,rho,W):
        M,N=X.shape
        term1=(mu-W)/rho2sig(rho)**2
        term2=np.dot(np.linalg.inv(Sigma_w),W)
        term3=0
        for n in range(N):
            term3+=-(Y[n]-sigmoid(np.dot(W.T,(X[:, n].reshape(M,1)))))*X[:,n].reshape(M,1)
        return term1+term2+term3
        
    def compute_df_dmu(mu,rho,W):
        return (W-mu)/rho2sig(rho)**2
        
    def compute_df_drho(Y,X,Sigma_w,mu,rho,W):
        return -0.5*((W-mu)**2-rho2sig(rho)**2)*compute_dprec_drho(rho)
        
    def compute_dprec_drho(rho):
        return (2*rho2sig(rho)**(-3))*((1/(1+np.exp(rho)))**2)*(1/(1+np.exp(-rho)))
        
    # diag gaussian for approximate posterior
    mu=np.random.randn(M,1)
    rho=np.random.normal(0,1,(M,1))
    
    for i in range(max_iter):
        #sample epsilon
        ep=np.random.random((M,1))
        W_tmp=mu+np.log(1+np.exp(rho))*ep
        
        # calculate gradient
        df_dw = compute_df_dw(Y, X, Sigma_w, mu, rho, W_tmp)
        df_dmu = compute_df_dmu(mu, rho, W_tmp)
        df_drho = compute_df_drho(Y, X, Sigma_w, mu, rho, W_tmp)
        d_mu = df_dw + df_dmu
        d_rho = df_dw * (ep / (1+np.exp(-rho))) + df_drho
        
        # update variational parameters
        mu = mu - alpha * d_mu
        rho = rho - alpha * d_rho 
        
    return mu,rho
        
