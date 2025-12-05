import numpy as np


def calculate_dim_nested(model, states_t, dt_mpor, n_nested, confidence=0.99):
    n_scenarios = len(states_t['x'])
    dims = np.zeros(n_scenarios)

    for i in range(n_scenarios):
        x0 = states_t['x'][i]
        y0 = states_t['y'][i]
        t0 = states_t['t'][i]

        v_t = model.price_zcb_analytic(0, 5.0, x0, y0)

        dw = np.random.normal(0, np.sqrt(dt_mpor), (n_nested, 2))
        dw[:, 1] = model.rho * dw[:, 0] + np.sqrt(1 - model.rho ** 2) * dw[:, 1]

        x_next = x0 - model.kx * x0 * dt_mpor + model.sx * dw[:, 0]
        y_next = y0 - model.ky * y0 * dt_mpor + model.sy * dw[:, 1]

        v_next = model.price_zcb_analytic(0, 5.0 - dt_mpor, x_next, y_next)

        pnl = v_next - v_t
        dims[i] = np.percentile(pnl, (1 - confidence) * 100) * -1

    return dims