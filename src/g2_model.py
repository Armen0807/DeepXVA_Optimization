import numpy as np


class G2PlusPlus:
    def __init__(self, params):
        self.kx = params['kappa_x']
        self.ky = params['kappa_y']
        self.sx = params['sigma_x']
        self.sy = params['sigma_y']
        self.rho = params['rho']
        self.r0 = params['r0']

    def simulate_paths(self, T, dt, n_paths):
        n_steps = int(T / dt)
        cov = [[1.0, self.rho], [self.rho, 1.0]]
        L = np.linalg.cholesky(cov)

        dw = np.random.normal(0, np.sqrt(dt), (n_steps, n_paths, 2))
        dw_corr = dw @ L.T

        x = np.zeros((n_steps + 1, n_paths))
        y = np.zeros((n_steps + 1, n_paths))
        r = np.zeros((n_steps + 1, n_paths))

        r[0] = self.r0

        for t in range(n_steps):
            x[t + 1] = x[t] - self.kx * x[t] * dt + self.sx * dw_corr[t, :, 0]
            y[t + 1] = y[t] - self.ky * y[t] * dt + self.sy * dw_corr[t, :, 1]
            r[t + 1] = x[t + 1] + y[t + 1] + self.r0

        t_grid = np.linspace(0, T, n_steps + 1)
        return t_grid, r, x, y

    def price_zcb_analytic(self, t, T, x_t, y_t):
        tau = T - t
        Bx = (1 - np.exp(-self.kx * tau)) / self.kx
        By = (1 - np.exp(-self.ky * tau)) / self.ky
        A = np.exp(-self.r0 * tau)

        return A * np.exp(-Bx * x_t - By * y_t)