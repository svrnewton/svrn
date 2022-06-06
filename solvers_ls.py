import numpy as np
import torch
from time import time

from sketches import gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores, lev_approx

SKETCH_FN = {'gaussian': gaussian, 'less': less, 'less_sparse': sparse_rademacher, 
             'srht': srht, 'rrs': rrs, 'rrs_lev_scores': rrs_lev_scores}


torch.set_default_dtype(torch.float64)

class RidgeRegression:
    
    def __init__(self, A, b, lambd, MAX_TIME, MIN_LOSS):
        self.A = A
        self.b = b
        if self.b.ndim == 1:
            self.b = self.b.reshape((-1,1))
        self.n, self.d = A.shape
        self.c = self.b.shape[1]
        self.lambd = lambd
        self.device = A.device
        self.MIN_LOSS = MIN_LOSS
        self.MAX_TIME = MAX_TIME
        self.stop_averaging = 0
        
    def loss(self, x):
        return 1./2 * ((self.H_opt @ (x - self.x_opt))**2).sum()
    
    def square_loss(self, x):
        return ((self.A @ x - self.b)**2).mean() + self.lambd/2 * (x**2).sum()

    def grad(self, x, indices=[]):
        batch_size = len(indices)
        if batch_size>1:
            return 1./batch_size*self.A[indices,::].T @ (self.A[indices,::] @ x - self.b[indices])+ self.lambd * x
        elif batch_size==0:
            return 1./self.n*self.A.T @ (self.A @ x - self.b)+ self.lambd * x
        else:
            index = indices[0]
            # index = np.random.choice(self.n)
            a_i = self.A[index,::].reshape((-1,1))
            b_i = self.b[index].squeeze()
            return a_i.reshape((-1,1)) * ((a_i*x).sum() - b_i)+ self.lambd * x
        
    def hessian(self, x):
        return 1./self.n * self.A.T @ self.A + self.lambd * torch.eye(self.d).to(self.device)

    def sqrt_hess(self, x):
        return 1./np.sqrt(self.n)

    def line_search(self, x, v, g, alpha=0.3, beta=0.8):
        delta = (v*g).sum()
        loss_x = self.square_loss(x)
        s = 1
        xs = x + s*v
        ls_passes = 1
        while self.square_loss(xs) > loss_x + alpha*s*delta and ls_passes<20:
            s = beta*s 
            xs = x + s*v
            ls_passes += 1
        print("line search: steps = " + str(ls_passes) + ", step size = " + str(s))        
        return s


    def uniform_weight(self, iter_num):
        return iter_num 
    
    def poly_weight(self, iter_num, p=2):
        return (iter_num) ** p

    def log_weight(self, iter_num):
        return (iter_num) ** np.log(iter_num+2)
    
    
    def solve_exactly(self):
        x = torch.zeros(self.d,self.c).to(self.device)
        
        g = self.grad(x)
        H = self.hessian(x)
        u = torch.linalg.cholesky(H)
        v = -torch.cholesky_solve(g, u)
        x = x + v
        
        self.x_opt = x
        self.H_opt = H
        
        _, sigma, _ = torch.svd(self.H_opt)
        
        de_ = torch.trace((H - self.lambd * torch.eye(self.d).to(self.device)) @ torch.pinverse(H))
        self.de = de_.cpu().numpy().item()
        
        return x
    
    def ihs_no_averaging(self, x, sketch_size, sketch, nnz):
        
        start = time()
        
        hsqrt = self.sqrt_hess(x)
        sa = SKETCH_FN[sketch](hsqrt * self.A, sketch_size, nnz=nnz)
        g = self.grad(x)
        
        if sketch_size >= self.d:
            hs = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
            u = torch.linalg.cholesky(hs)
            v = -torch.cholesky_solve(g, u)
        elif sketch_size < self.d:
            ws = sa @ sa.T + self.lambd * torch.eye(sketch_size).to(self.device)
            u = torch.linalg.cholesky(ws)
            sol_ = torch.cholesky_solve(sa @ g, u)
            v = -1./self.lambd * (g - sa.T @ sol_)
            
        s = self.line_search(x, v, g)
        x = x + s*v
        
        return x, time()-start


    def ihs_unweighted_(self, x, sketch_size, sketch, nnz, hs_old, iter):
        start = time()
        
        g = self.grad(x)

        if self.stop_averaging == 0 or iter < self.stop_averaging:
            hsqrt = self.sqrt_hess(x)
            sa = SKETCH_FN[sketch](hsqrt * self.A, sketch_size, nnz=nnz)

            w_old = self.uniform_weight(iter)
            w = self.uniform_weight(iter + 1)
            hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
            hs = (w_old / w) * hs_old + (1 - w_old / w) * hs_
        else:
            hs = hs_old
            
        u = torch.linalg.cholesky(hs)
        v = -torch.cholesky_solve(g, u)
        # v = -torch.pinverse(hs) @ g
        
        s = self.line_search(x, v, g)
        x = x + s*v
                
        return x, hs, time()-start

    
    def ihs(self, sketch_size, sketch='gaussian', nnz=1., n_iter=10, scheme='unweighted',stop_averaging=0):
        self.stop_averaging = stop_averaging
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        hs = torch.zeros(self.d,self.d).to(self.device)

        print("IHS "+scheme + "\n") 
        losses = [self.loss(x).cpu().numpy().item()]
        times = [0.]
        for i in range(n_iter):
            print("Pass "+str(i))            
            if scheme == 'unweighted':
                x, hs, time_ = self.ihs_unweighted_(x, sketch_size, sketch, nnz, hs, i)
            elif scheme == 'poly':
                x, hs, time_ = self.ihs_poly_(x, sketch_size, sketch, nnz, hs, i)
            elif scheme == 'log':
                x, hs, time_ = self.ihs_log_(x, sketch_size, sketch, nnz, hs, i)
            else:
                x, time_ = self.ihs_no_averaging(x, sketch_size, sketch, nnz)
            losses.append(self.loss(x).cpu().numpy().item())            
            times.append(time_)
            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (i + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (i + 1)))
                break
        
        losses = np.array(losses)
        times = np.array(times)        
        losses /= losses[0]
        
        return x, losses, np.cumsum(times)


    def ihs_svrn(self, sketch_size, sketch='rrs', nnz=.1, n_local = 0, n_iter=10, scheme='no averaging', sampling='per stage',with_vr=True,s=1,stop_averaging=0,permanent_switch=False):
                
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        hs = torch.zeros(self.d,self.d).to(self.device)

        A_full = self.A
        b_full = self.b
        n_full = self.n


        
        losses = [self.loss(x).cpu().numpy().item()]
        times = [0.]
        if n_local == 0:
            n_local = max(int(np.log(self.n/self.d)/np.log(2)),2)

        batch_size = int(self.n / n_local)
        print("SVRN with batch_size=" + str(batch_size) + " and n_local=" + str(n_local)+"\n")

        if sampling == 'once':
            batch_indices = np.random.choice(self.n, batch_size, replace=False) # n_full or self.n
            A_batch = A_full[batch_indices,::]
            b_batch = b_full[batch_indices]

        
        s_global = 0

        for i in range(n_iter):
            print("Pass "+str(i))            
            start = time()

            g0 = self.grad(x)

            if stop_averaging == 0 or i < stop_averaging:
                hsqrt = self.sqrt_hess(x)
                sa = SKETCH_FN[sketch](hsqrt * self.A, sketch_size, nnz=nnz)
                if scheme == 'unweighted':
                    w_old = self.uniform_weight(i)
                    w = self.uniform_weight(i + 1)
                    hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                    hs = (w_old / w) * hs + (1 - w_old / w) * hs_
                else:
                    hs = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                u = torch.linalg.cholesky(hs)          

            if s_global < 1:
                v = -torch.cholesky_solve(g0, u)
                s_global = self.line_search(x, v, g0)
                x = x + s_global*v
            else:                
                x0 = x.clone().detach()
                if sampling == 'once':
                    self.A = A_batch
                    self.b = b_batch
                    self.n = batch_size
                elif sampling == 'per stage':
                    batch_indices = np.random.choice(n_full, batch_size, replace=False)
                    self.A = A_full[batch_indices,::]
                    self.b = b_full[batch_indices]
                    self.n = batch_size
                for j in range(n_local):
                    if sampling == 'per step':
                        batch_indices = np.random.choice(n_full, batch_size, replace=False) # n_full or self.n
                        ghat = self.grad(x,indices=batch_indices)            
                        ghat0 = self.grad(x0,indices=batch_indices)
                    else:
                        ghat = self.grad(x)
                        ghat0 = self.grad(x0)
                    if with_vr:
                        g = ghat - ghat0 + g0
                    else:
                        g = ghat
                    v = -torch.cholesky_solve(g, u)
                    x = x + s*v
                    
                self.A = A_full
                self.b = b_full
                self.n = n_full
                v = x - x0                
                s_global = self.line_search(x0, v, g0)
                x = x0 + s_global*v
                if not with_vr or permanent_switch: s_global = 1

            times.append(time()-start)
            losses.append(self.loss(x).cpu().numpy().item())
            
            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (i + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (i + 1)))
                break
        
        losses = np.array(losses)
        times = np.array(times)        
        losses /= losses[0]
        
        return x, losses, np.cumsum(times)

    def svrg(self, m, n_iter=100, s=0.01, batch_size=10):
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)

        losses = [self.loss(x).cpu().numpy().item()]
        times = [0.]
        print("\nSVRG with s="+str(s)+" m="+str(m)+" b="+str(batch_size))
        for i in range(n_iter):
            print("Pass "+str(i))            
            start = time()

            g = self.grad(x)
            x0 = x.clone().detach()

            for j in range(m):
                batch_indices = np.random.choice(n_full, batch_size, replace=False)
                
                g_sto = self.grad(x, indices = batch_indices)
                g_sto0 = self.grad(x0, indices = batch_indices)
                x = x - s*(g_sto - g_sto0 + g)
            
            times.append(time()-start)
            losses.append(self.loss(x).cpu().numpy().item())
            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (i + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (i + 1)))
                break

            
        losses = np.array(losses)
        times = np.array(times)
        losses /= losses[0]
            
        return x, losses, np.cumsum(times)
    
    
