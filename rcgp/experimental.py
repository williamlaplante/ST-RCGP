import torch as tc
from torch import nn
import torch.nn.functional as F

from typing import Tuple
from .utils import eye
from .weight_functions import IMQ, partial_y_IMQ
from .kernels import Matern32Kernel
from .latent_process import MaternProcess
from .weight_functions import IMQ, partial_y_IMQ


class SpatioTemporalRCGP(nn.Module):
    """
    Spatio-Temporal Robust and Conjugate Gaussian Process.
    """
    def __init__(self, 
                 ts : tc.Tensor,
                 grid : tc.Tensor | None,
                 Ys : tc.Tensor, p : int = 1,
                 raw_var_y_init : float = 1.0,
                 raw_temporal_lengthscale_init : float = 1.0,
                 raw_temporal_magnitude_init : float = 1.0,
                 raw_spatial_lengthscale_init : float = 1.0,
                 raw_spatial_magnitude_init : float = 1.0,
                 ):
        """
        Initialization phase prior to hyperparameter optimization or inference/prediction. 

        Args:
            ts (tc.Tensor): time vector of size (n_t, 1). Must not contain nulls, inf, and must be strictly increasing.
            grid (tc.Tensor): Grid at which data is located, of size (n_r, d-1). Fixed across time. Must not contain nulls, inf.
            Ys (tc.Tensor): Data, of size (n_t, n_r, 1). If at a time step, there are any nulls, data will not be used for prediction.
            p (int, optional): Number of temporal derivatives modelled (>0). Greater p means more smoothness. Defaults to 1.
        """
        super().__init__()
        
        #Initial checks
        (self.n_r, self._n_t, self.d) = self.initialize(p=p, ts=ts, grid=grid, Ys=Ys)
        
        if grid is not None: self.grid = grid.to(tc.float32)
        self._ts = ts.to(tc.float32) #original ts (without padding)
        self._Ys = Ys.to(tc.float32) #original Ys (without padding)

        #Padding filtering time array and data array
        (self.n_t, self.Ys, self.ts) = self.padding(ts=self._ts, Ys=self._Ys)

        self.__prior_mean_funcs = ["constant", "local_constant", "m_pred", "spatial"]

        self.__fixed_params = {"p" : 1, #temporal Matern kernel (nu_{temporal} = p + 1/2)
                               "robust" : False,
                               "prior_mean" : "constant",
                               "beta" : None, #If None, will be sqrt(var_y / 2)
                               "is_beta_fixed" : False,
                               "c" : 1., #If None and is_c_fixed=True, will be c_factor * sqrt(var_y). 
                               "c_factor" : 1.,
                               "is_c_fixed": True, #When false, c is adaptive and = 2 * predicted std
                               }

        self._raw_var_y = nn.Parameter(tc.tensor(raw_var_y_init, dtype=tc.float32))
        self._raw_temporal_lengthscale = nn.Parameter(tc.tensor(raw_temporal_lengthscale_init, dtype=tc.float32))
        self._raw_temporal_magnitude = nn.Parameter(tc.tensor(raw_temporal_magnitude_init, dtype=tc.float32))
        self._raw_spatial_lengthscale = nn.Parameter(tc.tensor(raw_spatial_lengthscale_init, dtype=tc.float32))
        self._raw_spatial_magnitude = nn.Parameter(tc.tensor(raw_spatial_magnitude_init, dtype=tc.float32))

        if self.n_r == 0:
            self._raw_spatial_lengthscale.requires_grad_(False)
            self._raw_spatial_magnitude.requires_grad_(False)
        
        #Measurement matrices
        if self.n_r > 0: #spatio-temporal
            self.Id = eye(self.n_r).to(tc.float32)
            self.H0 = eye(1, (1+self.__fixed_params["p"]), k=0).to(tc.float32) #measurement matrix -> temporal only case
            self.H = tc.kron(self.Id, self.H0).to(tc.float32) #measurement matrix -> expanded to spatio-temporal
            self.latent_size = self.n_r * (p + 1)
            self.K_w = None #For prior mean
            self._K_w_lengthscale = 4.
        else: #temporal
            self.H = eye(1, (1+self.__fixed_params["p"]), k=0).to(tc.float32) #measurement matrix -> temporal only case
            self.latent_size = p + 1

        return
    

    @property
    def fixed_params(self):
        return self.__fixed_params

    @property
    def var_y(self):
        return F.softplus(self._raw_var_y)
    
    @var_y.setter
    def var_y(self, value):
        """
        Setter for the variance in y.
        """
        if value <= 0:
            raise ValueError("Parameter must be positive.")
        # Reverse transform: raw_parameter = log(exp(value) - 1)
        self._raw_var_y.data = tc.log(tc.exp(tc.tensor(value, dtype=tc.float32)) - 1)

    @property
    def beta(self):
        if self.__fixed_params["is_beta_fixed"]:

            if isinstance(self.__fixed_params["beta"], (float, int)):
                return tc.tensor(self.__fixed_params["beta"], dtype=tc.float32)
            
            elif isinstance(self.__fixed_params["beta"], tc.Tensor):
                return self.__fixed_params["beta"]
            else:
                raise ValueError(f"{self.__fixed_params["beta"]} is an invalid data type for beta.")
        else:
            return tc.sqrt(self.var_y / 2)#.clone().detach()
    
    @beta.setter
    def beta(self, value : float):
        if isinstance(value, tc.Tensor):
            self.__fixed_params["beta"] = value

        elif isinstance(value, (float, int)):
            self.__fixed_params["beta"] = tc.tensor(value, dtype=tc.float32)

        elif value is None: #If we set beta = None, we want to revert back to adaptive beta. 
            self.__fixed_params["beta"] = None
            self.__fixed_params["is_beta_fixed"] = False
            return
        else:
            raise ValueError(f"{value} is not a valid value for parameter beta.")
        
        self.__fixed_params["is_beta_fixed"] = True
        return
    
    @property
    def c(self):
        if isinstance(self.__fixed_params["c"], (float, int)):
            return tc.tensor(self.__fixed_params["c"], dtype=tc.float32)
        
        elif isinstance(self.__fixed_params["c"], tc.Tensor):
            return self.__fixed_params["c"]
        else:
            return self.__fixed_params["c_factor"] * tc.sqrt(self.var_y)#.clone().detach()
    
    @c.setter
    def c(self, value : float):
        if isinstance(value, tc.Tensor):
            self.__fixed_params["c"] = value
            
        elif isinstance(value, (float, int)):
            self.__fixed_params["c"] = tc.tensor(value, dtype=tc.float32)

        elif value is None:
            self.__fixed_params["c"] = None

        else:
            raise ValueError(f"{value} is not a valid value for parameter c.")
        
        self.__fixed_params["is_c_fixed"] = True
        return
    
    def adaptive_c(self):
        self.__fixed_params["is_c_fixed"] = False
        return

    @property
    def temporal_lengthscale(self):
        return F.softplus(self._raw_temporal_lengthscale)
    
    @temporal_lengthscale.setter
    def temporal_lengthscale(self, value):
        """
        Setter for the temporal lengthscale.
        """
        if value <= 0:
            raise ValueError("Temporal lengthscale must be positive.")
        # Reverse transform: raw_parameter = log(exp(value) - 1)
        if isinstance(value, tc.Tensor):
            self._raw_temporal_lengthscale.data = tc.log(tc.exp(value) - 1)
        else:
            self._raw_temporal_lengthscale.data = tc.log(tc.exp(tc.tensor(value, dtype=tc.float32)) - 1)

    @property
    def temporal_magnitude(self):
        return F.softplus(self._raw_temporal_magnitude)
    
    @temporal_magnitude.setter
    def temporal_magnitude(self, value):
        """
        Setter for the temporal magnitude.
        """
        if value <= 0:
            raise ValueError("Temporal magnitude must be positive.")
        # Reverse transform: raw_parameter = log(exp(value) - 1)
        if isinstance(value, tc.Tensor):
            self._raw_temporal_magnitude.data = tc.log(tc.exp(value) - 1)
        else:
            self._raw_temporal_magnitude.data = tc.log(tc.exp(tc.tensor(value, dtype=tc.float32)) - 1)

    @property
    def spatial_lengthscale(self):
        return F.softplus(self._raw_spatial_lengthscale)
    
    @spatial_lengthscale.setter
    def spatial_lengthscale(self, value):
        """
        Setter for the spatial lengthscale.
        """
        if value <= 0:
            raise ValueError("Spatial lengthscale must be positive.")
        # Reverse transform: raw_parameter = log(exp(value) - 1)
        if isinstance(value, tc.Tensor):
            self._raw_spatial_lengthscale.data = tc.log(tc.exp(value) - 1)
        else:
            self._raw_spatial_lengthscale.data = tc.log(tc.exp(tc.tensor(value, dtype=tc.float32)) - 1)

    @property
    def spatial_magnitude(self):
        return F.softplus(self._raw_spatial_magnitude)
    
    @spatial_magnitude.setter
    def spatial_magnitude(self, value):
        """
        Setter for the spatial magnitude.
        """
        if value <= 0:
            raise ValueError("Spatial magnitude must be positive.")
        # Reverse transform: raw_parameter = log(exp(value) - 1)
        if isinstance(value, tc.Tensor):
            self._raw_spatial_magnitude.data = tc.log(tc.exp(value) - 1)
        else:
            self._raw_spatial_magnitude.data = tc.log(tc.exp(tc.tensor(value, dtype=tc.float32)) - 1)

    
    def initialize(self, p, ts, grid, Ys) -> Tuple[int, int, int]:

        #checks on p
        assert type(p) == int and p > 0, "p must be an integer greater than 0."

        #checks on ts
        assert ts.ndim == 2 and ts.shape[1] == 1, "ts must be two-dimensional, with n_t rows and one column."
        if tc.isnan(ts).any(): raise ValueError("time values contain NaN values.")
        if tc.isinf(ts).any(): raise ValueError("time values contain Inf values.")
        if not tc.all(ts[1:] - ts[:-1] > 0): ValueError("time values are not strictly increasing.")

        if grid is None: #If no spatial grid (temporal only)
            #checks on Ys
            assert Ys.ndim == 2 and Ys.shape[0] == len(ts) and Ys.shape[1] == 1, "Temporal data must have shape (n_t, 1)."

            n_r = 0
            n_t = ts.shape[0]
            d = 1

        else:
            #checks on grid
            assert grid.ndim == 2, "grid must be two-dimensional, with n_r rows and d columns."
            if tc.isnan(grid).any(): raise ValueError("grid contains NaN values.")
            if tc.isinf(grid).any(): raise ValueError("grid contains Inf values.")

            #checks on Ys
            assert Ys.ndim == 3 and Ys.shape==(len(ts), len(grid), 1), "Ys must be of shape (n_t, n_r, 1), where n_t is the number of time steps, n_r the size of the grid. Also possible that grid has the wrong number of rows."
            if tc.isinf(Ys).any(): raise ValueError("Ys contains Inf values.")

            n_r = grid.shape[0]
            n_t = ts.shape[0]
            d = grid.shape[1] + 1 #number of spatial and temporal dimensions (hence the +1 for temporal)

        return n_r, n_t, d
 

    def padding(self, ts : tc.Tensor, Ys : tc.Tensor):

        t0 = (ts[0] - (ts[1] - ts[0])).unsqueeze(0)
        tf = (ts[-1] + (ts[-1] - ts[-2])).unsqueeze(0)

        ts_filtering = tc.concatenate([t0, ts, tf])
        n_t_filtering = ts_filtering.shape[0]

        if self.n_r > 0:
            nan_row = tc.full((1, self.n_r, 1), float('nan'))

            Ys_filtering = tc.cat([nan_row, Ys, nan_row], dim=0)

        else:
            nan_row = tc.full((1, 1), float('nan'))
            Ys_filtering = tc.cat([nan_row, Ys, nan_row], dim=0)

        return n_t_filtering, Ys_filtering.to(tc.float32), ts_filtering.to(tc.float32)
    

    def v(self, Y : tc.Tensor, weights : tc.Tensor, partial_weights : tc.Tensor):

        if not self.__fixed_params["robust"]:
            R_inv = self.R_inv(Y=Y, weights=weights, partial_weights=partial_weights)
            return R_inv @ Y
        else:
            return (2/self.var_y**2) * (weights**2 * Y - 2 * self.var_y * weights * partial_weights)

    def R_inv(self, Y : tc.Tensor, weights : tc.Tensor, partial_weights : tc.Tensor):

        if not self.__fixed_params["robust"]:
            return 1/self.var_y * tc.eye(Y.shape[1])
        else:
            #return self.var_y**2 / 2 * tc.diag(weights.squeeze(dim=-1)**(-2))
            return (2/self.var_y**2) * tc.diag(weights.flatten()**2)


    def get_predict_step_matrices(self, dt : float, temporal_kernel : MaternProcess, spatial_kernel : Matern32Kernel | None):
        
        At, Sigt = temporal_kernel.forward(dt)

        if self.n_r > 0:
            K_spatial = spatial_kernel.forward(self.grid, self.grid)
            A = tc.kron(self.Id, At) #IS THIS RIGHT? OR IS IT exp(kron(Id, F_t) * dt)?
            Sig = tc.kron(K_spatial, Sigt)
        else:
            A = At
            Sig = Sigt

        return A.to(tc.float32), Sig.to(tc.float32)
    
    def activate_robustness(self, func : str):
        if func in self.__prior_mean_funcs:
            self.__fixed_params["prior_mean"] = func
            self.__fixed_params["robust"] = True
        else:
            raise ValueError(f"{func} is not in {self.__prior_mean_funcs}.")

        return
    
    def deactivate_robustness(self):
        self.__fixed_params["robust"] = False
        return
    
    def set_prior_mean(self, func : str):
        self.__fixed_params["prior_mean"] = func

    def compute_K_w(self):
        k = Matern32Kernel(lengthscale=tc.tensor(self._K_w_lengthscale), magnitude=tc.tensor(1.))

        final_arr = tc.empty((self.n_r, self.n_r))

        for i in range(self.n_r):
            final_arr[i, :] = k.forward(self.grid[[i]], self.grid)
            final_arr[i, :] = final_arr[i, :] / final_arr[i, :].sum() #Normalizing, since these are weights

        return final_arr
        
    def prior_mean(self, Y : tc.Tensor, m_prior : tc.Tensor):

        if self.__fixed_params["prior_mean"] == 'constant':
            mean = self.Ys.nanmean() * tc.ones_like(Y) #nan mean because we might encounter NaNs in the data (we actually buffer with NaNs)
            return mean
        
        if self.__fixed_params["prior_mean"] == 'local_constant':
            return self.K_w @ Y
        
        elif self.__fixed_params["prior_mean"] == 'm_pred':
            return m_prior

        elif self.__fixed_params["prior_mean"] == "spatial":
            return self.K_w @ m_prior
        
        else:
            return m_prior
        
    def compute_weights(self, Y : tc.Tensor, m_prior : tc.Tensor, P_prior : tc.Tensor):

        if not self.__fixed_params["robust"]:
            return tc.sqrt(self.var_y / 2).clone().detach() * tc.ones_like(Y), tc.zeros_like(Y)
        else:
            m = self.prior_mean(Y=Y, m_prior=m_prior)

            if self.__fixed_params["is_c_fixed"]:
                c = self.c
            else:
                with tc.no_grad():
                    #c = 1.
                    c = tc.sqrt(tc.diagonal(P_prior) + self.var_y - self.var_y * tc.diagonal(P_prior))#.clone().detach()
                    if self.n_r > 0: 
                        c = c.reshape(1, -1, 1)


            weights = IMQ(Y=Y, m=m, beta=self.beta, c=c)

            partial_y_weights = partial_y_IMQ(Y=Y, m=m, beta=self.beta, c=c)


            return weights, partial_y_weights
    

    def predict_step(self, A : tc.Tensor, Sig : tc.Tensor, m : tc.Tensor, P : tc.Tensor):
        m_pred = A @ m  # Predict state
        P_pred = A @ P @ A.T + Sig  # Predict covariance
        return m_pred.to(tc.float32), P_pred.to(tc.float32)
    
    def update_step(self, m_pred : tc.Tensor, P_pred : tc.Tensor, Y : tc.Tensor, optim : bool, weighted_loss : bool = False):

        H_m_pred = self.H @ m_pred #Prediction of mean
        H_P_pred = self.H @ P_pred #Partial prediction of covariance

        weights, partial_weights = self.compute_weights(Y=Y, m_prior=H_m_pred, P_prior=H_P_pred @ self.H.T) #Weights Computation
        R_inv_k = self.R_inv(Y=Y, weights=weights, partial_weights=partial_weights)
        R_k = tc.linalg.inv(R_inv_k)
        v_k = self.v(Y=Y, weights=weights, partial_weights=partial_weights)

        P_updated = tc.linalg.inv( tc.linalg.inv(P_pred) + self.H.T @ R_inv_k @ self.H)
        K_k = P_updated @ self.H.T @ R_inv_k
        m_updated = m_pred + K_k @ (R_k @ v_k - H_m_pred)
        """
        if optim:
            y_hat_e = (Y - H_m_pred).squeeze(0) #shape is (1, n_s, 1). So we squeeze the first dimension, which is time. 
            S_e = H_P_pred @ self.H.T + self.var_y * tc.eye(Y.shape[1])

            if weighted_loss:
                #NOTE: .clone().detach() ????
                #S_e_w = tc.diag(self.beta * weights.flatten().clone().detach()**(-1)) @ S_e # diag(w^{-1}) @ S_k
                energy = (0.5 * tc.logdet(2 * tc.pi * S_e) + 0.5 * y_hat_e.T @ tc.linalg.inv(S_e) @ y_hat_e).squeeze()
            else:
                energy = (0.5 * tc.logdet(2 * tc.pi * S_e) + 0.5 * y_hat_e.T @ tc.linalg.inv(S_e) @ y_hat_e).squeeze()
            return m_updated, P_updated, energy, weights
        else:
            return m_updated, P_updated, weights
        """
        return m_updated, P_updated, weights
    
    def filtsmooth(self, temporal_kernel : MaternProcess, spatial_kernel : Matern32Kernel | None, m0 : tc.Tensor, P0 : tc.Tensor, energy0 : tc.Tensor = tc.tensor(0.0, requires_grad=True), optim=False, weighted_loss=False):

        #If we perform prediction (not optimization of energy function), generate containers for predictions and further analysis
        #if not optim:
            #Where we store the state, covariance estimates m_{k|k}, P_{k|k}
        ms = tc.empty(size=(self.n_t, self.latent_size, 1), dtype=tc.float32) #n_t time steps where n_t is len(ts) + 2 (padded). latent size depends on number of spatial dimensions and temporal derivatives
        Ps = tc.empty(size=(self.n_t, self.latent_size, self.latent_size), dtype=tc.float32) #same here. 

            #Where we store the one-step predictions, or predictive  p(z_k | y_{1:k-1})
            #m_preds = tc.empty_like(ms) 
            #P_preds = tc.empty_like(Ps)

            #We store the transition matrices A = exp(dt * F) for smoothing
        As = tc.empty_like(Ps)
        
        if self.n_r > 0: #if there is a spatial dimension
            #Container for the weights. Choosing lists because might have to autodiff through weights and inplace assignment is no good for autodiff.
            W0 = tc.zeros(size=(1, self.n_r, 1))
            Ws = [W0]
        else:
            W0 = tc.zeros(size=(1, 1))
            Ws = [W0]
        
        m_preds = [m0]
        P_preds = [P0]
        #energies = [energy0]

        #Initialize the 0th step. No data observed on 0th step. We always do look-ahead predict and update (i.e. from k, predict k+1 and update with data at k+1)
        m, P = m0, P0 #m, P are the "running" state and covariance matrix. They get overwritten. 

        if not optim:
            ms[0], Ps[0] = m0.clone().detach(), P0.clone().detach()
            #m_preds[0], P_preds[0] = m0.clone().detach(), P0.clone().detach()

        for k in range(self.n_t - 1):
            #Compute time difference from now to next step
            dt = (self.ts[k+1] - self.ts[k]).item()

            #--------------------Predict Step------------------------
            A, Sig = self.get_predict_step_matrices(dt=dt, temporal_kernel=temporal_kernel, spatial_kernel=spatial_kernel) #Get the transition matrix and the noise matrix
            m_pred, P_pred = self.predict_step(A=A, Sig=Sig, m=m, P=P) #Get state,covariance estimate of z_k for y_{k+1}
            
            m_preds.append(m_pred)
            P_preds.append(P_pred)

            #if not optim: #If we don't optimize energy, store the prediction estimates and transition matrices
            As[k] = A.clone().detach()
                #m_preds[k+1] = m_pred.clone().detach()
                #P_preds[k+1] = P_pred.clone().detach()

            #---------------------Update Step------------------------
            if tc.isnan(self.Ys[k+1]).any(): #If data is incomplete we don't perform update step. 
                #We assign to active m, P the predict step estimate when there is missing data.
                m = m_pred
                P = P_pred

                #energy is 0 because there is nothing to compare against for our predict.
                #energies.append(tc.tensor(0.0, requires_grad=True))
                Ws.append(W0.clone().detach()) #Weights are also zero because we don't want to include in calculation a value of 0.

                #if not optim: #when we're not optimizing
                ms[k+1] = m_pred.clone().detach() #We store the predict step mean m_{k+1|k} as the mean estimate m_{k+1|k+1}
                Ps[k+1] = P_pred.clone().detach() #We store the predict step cov P_{k+1|k} as the cov estimate P_{k+1|k+1}

            else: #Update with data at k+1 when there is no missing data
                m_updated, P_updated, weights = self.update_step(m_pred=m_pred, P_pred=P_pred, Y=self.Ys[[k+1]], optim=optim, weighted_loss=True)
                #if optim:
                #    m_updated, P_updated, energy, weights = self.update_step(m_pred=m_pred, P_pred=P_pred, Y=self.Ys[[k+1]], optim=optim, weighted_loss=True)
                #    Ws.append(weights) 
                #    energies.append(energy)
                #else:
                #    m_updated, P_updated, weights = self.update_step(m_pred=m_pred, P_pred=P_pred, Y=self.Ys[[k+1]], optim=optim, weighted_loss=False)
                ms[k+1] = m_updated.clone().detach()
                Ps[k+1] = P_updated.clone().detach()
                Ws.append(weights.clone().detach()) 
                    #energies.append(energy.clone().detach())

                #Running mean, cov are replaced with updated predict step mean, cov
                m = m_updated
                P = P_updated

        #Stacking into a tensor the weights and energies
        Ws = tc.concatenate(Ws, dim=0) #weights are of the form (1, n_r, 1) for n_r>0 and (1,1) for n_r = 0. So we concatenate on first dim.
        if self.n_r > 0:
            m_preds = tc.concatenate(m_preds, dim=0)
            P_preds = tc.stack(P_preds)
        else:
            m_preds = tc.stack(m_preds)
            P_preds = tc.stack(P_preds)

        #energies = tc.stack(energies) #we stack here since energies are squeezed to 0-dim tensors.


        #if optim: #If we're optimizing, skip smoothing and return the energies, weights
        #    return energies, Ws

        #If we're predicting, we need to condition on all data, i.e. p(z_k | y_{1:n_t})
        #else:

        #Smoother
        for k in range(self.n_t - 2, -1, -1):  
            #Using the m_{k+1|k}, P_{k+1|k} predict step estimates from filtering, we can perform RTS smoothing
            C = Ps[k] @ As[k].T @ tc.linalg.inv(P_preds[k+1].clone().detach())

            ms[k] = ms[k] + C @ (ms[k+1] - m_preds[k+1].clone().detach())
            Ps[k] = Ps[k] + C @ (Ps[k+1] - P_preds[k+1].clone().detach()) @ C.T
            
        return ms, Ps, m_preds, P_preds, Ws
    
    def forward(self, optim : bool = False, weighted_loss=False, predict=False, m0_data=False):
        
        #Instantiate the temporal kernel
        temporal_kernel = MaternProcess(p=self.__fixed_params["p"],
                                        lengthscale=self.temporal_lengthscale,
                                        magnitude=self.temporal_magnitude,
                                        )

        #If there is are spatial dimensions
        if self.n_r > 0:
            spatial_kernel = Matern32Kernel(lengthscale=self.spatial_lengthscale,
                                       magnitude=self.spatial_magnitude,
                                       )
            
            self.K_w = self.compute_K_w()

            #Initialize m0 as zeros, since the first step is always a "calibration step" and no data is observed at 0 (because of padding).
            #P0 is the kronecker between spatial kernel and temporal covariance at time 0 (\Sigma_\infty)
            
            m0=tc.zeros(size=(1, self.latent_size, 1), dtype=tc.float32)
            if m0_data: 
                m0[:, ::(1+self.__fixed_params["p"]), :] = self.Ys[[1]]
            P0=tc.kron(spatial_kernel.forward(self.grid, self.grid), temporal_kernel.Sig0).to(tc.float32)

        #If no spatial 
        else:
            spatial_kernel = None
            m0=tc.zeros(size=(self.latent_size, 1), dtype=tc.float32)
            P0=temporal_kernel.Sig0.to(tc.float32)

        ms, Ps, m_preds, P_preds, Ws = self.filtsmooth(m0=m0, P0=P0, spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel, optim=optim, weighted_loss=weighted_loss)

        preds_filt = (self.H @ m_preds)[1:-1]
        if self.n_r == 0:
            preds_filt = preds_filt.squeeze(-1)
            
        covs_filt = (self.H @ P_preds @ self.H.T)[1:-1]

        preds_smooth = (self.H @ ms)[1:-1].squeeze(-1)
        covs_smooth = (self.H @ Ps @ self.H.T)[1:-1]
        stds_smooth = tc.sqrt(tc.diagonal(covs_smooth, dim1=1, dim2=2)).squeeze()

        R = self.var_y * tc.eye(self.n_r) if self.n_r > 0 else self.var_y * tc.eye(1)
        Ws_norm = Ws[1:-1] / Ws.sum()
        eff = (Ws.clone().detach() / self.beta.clone().detach())[1:-1]

        return (preds_smooth, stds_smooth, eff), (preds_filt, covs_filt, R, Ws_norm), (ms, Ps)
        """
        if predict:
            preds = (self.H @ ms)[1:-1] #[1:-1] because of the padding
            stds = tc.sqrt(tc.diagonal((self.H @ Ps @ self.H.T), dim1=1, dim2=2))[1:-1]
            eff = Ws[1:-1].clone().detach() / self.beta.clone().detach()
            return preds, stds, eff, m_preds, P_preds

        else:
            return ms, Ps, m_preds, P_preds, Ws
        """
        #If we're optimizing
        """if optim: 
            energies, weights = self.filtsmooth(m0=m0, P0=P0, spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel, optim=optim, weighted_loss=weighted_loss)
            return energies, weights"""
        
        #If we're performing predictions
        """else:
            with tc.no_grad():
                ms, Ps, m_preds, P_preds, energies, Ws = self.filtsmooth(m0=m0, P0=P0, spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel, optim=optim, weighted_loss=weighted_loss)
                
                if predict:
                    preds = (self.H @ ms)[1:-1] #[1:-1] because of the padding
                    stds = tc.sqrt(tc.diagonal((self.H @ Ps @ self.H.T), dim1=1, dim2=2))[1:-1]
                    eff = Ws[1:-1].clone().detach() / self.beta.clone().detach()
                    return preds, stds, eff, m_preds, P_preds
                else:
                    return ms, Ps, m_preds, P_preds, energies, Ws"""