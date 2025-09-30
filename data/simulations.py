import numpy as np
import torch as tc
from scipy import stats
from rcgp.kernels import Matern32Kernel
from BayesNewton.bayesnewton.utils import create_spatiotemporal_grid



def spatio_temporal_half_contaminated(seed=2, outlier_proportion=0.1):
    np.random.seed(seed)

    london_box = [
        [51.279, 51.684],  # lat
        [-0.533, 0.208]  # lon
    ]

    n_lat = 25
    n_lon = 25
    n_t = 10
    n_s = n_lat * n_lon

    lat = np.linspace(*london_box[0], n_lat)
    lon = np.linspace(*london_box[1], n_lon)
    times = np.linspace(0.2, 0.8, n_t)


    lat_norm = (lat - lat.mean())/lat.std()
    lon_norm = (lon - lon.mean())/lon.std()

    def f(t, x, y):
        return np.sin(2 * np.pi * t)  * x**2 + np.cos(2 * np.pi * t) * y**2

    X = []
    Y = []
    Y_no_outliers = []
    true_Y = []
    true_scale = 0.2
    outliers_loc = []

    for t in times:
        for x in lat_norm:
            for y in lon_norm:
                X.append([t, x, y])
                true_Y.append([f(t, x, y)])

                regular_noise = stats.norm(loc=0, scale=true_scale).rvs()
                y_val = f(t, x, y) + regular_noise

                Y_no_outliers.append([y_val])

                if x < lat_norm.mean() and np.random.uniform() > (1-outlier_proportion):
                    p = stats.bernoulli(p=0.5).rvs()
                    extra_noise = p * np.random.uniform(-8,-6) + (1-p)*np.random.uniform(6,8)
                    y_val += extra_noise
                    outliers_loc.append([t, x, y, y_val])

                Y.append([y_val])
                

    X=np.array(X)
    outliers_loc = np.array(outliers_loc)

    Y=np.array(Y)
    true_Y = np.array(true_Y)
    Y_no_outliers = np.array(Y_no_outliers)

    """Y = Y/Y.max()
    true_Y = true_Y/Y.max()"""

    t, grids, Y = create_spatiotemporal_grid(X, Y)
    _, _, true_Y = create_spatiotemporal_grid(X, true_Y)
    _, _, Y_no_outliers = create_spatiotemporal_grid(X, Y_no_outliers)

    ts = tc.from_numpy(np.array(t)).type(tc.float32)
    grids = tc.from_numpy(np.array(grids)).type(tc.float32)
    Ys = tc.from_numpy(np.array(Y)).type(tc.float32)
    true_Ys = tc.from_numpy(np.array(true_Y)).type(tc.float32)
    Ys_no_outliers = tc.from_numpy(np.array(Y_no_outliers)).type(tc.float32)

    return ts, grids, Ys, true_Ys, Ys_no_outliers, outliers_loc, n_lon, n_lat, lon_norm, lat_norm




def generate_gp_temporal_data(lengthscale=0.4, magnitude=2., var_y = 0.02):
    tc.manual_seed(42)  # For reproducibility
    x = tc.linspace(0, 1, 100) 
    kernel = Matern32Kernel(lengthscale=tc.tensor(lengthscale), magnitude=tc.tensor(magnitude))

    cov_matrix, mean = kernel.forward(x.reshape(-1,1), x.reshape(-1,1)) + 1e-5 * tc.eye(len(x)), tc.zeros(x.shape)

    true_f = tc.distributions.MultivariateNormal(mean, cov_matrix).sample()
    true_f = true_f - true_f.mean()

    noise = tc.distributions.Normal(0, var_y).sample(true_f.shape)
    y = true_f + noise

    return x, y, true_f

