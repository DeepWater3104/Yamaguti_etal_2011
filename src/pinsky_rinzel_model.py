import numpy as np

class PinskyRinzelModel:
    def __init__(self, params=None):
        self.params = self._set_default_params()
        if params:
            self.params.update(params)

    def _set_default_params(self):
        default_params = {
            'C_m': 1.0,
            'g_Na': 120.0,
        }
        return default_params

    def equations(self, t, y):
        dV_soma_dt = ...
        dV_dend_dt = ...

        return [dV_soma_dt, dV_dend_dt, ...]

    def simulate(self, t_span, y0, dt):
        from scipy.integrate import solve_ivp

        sol = solve_ivp(self.equations, t_span, y0, method='RK45', t_eval=np.arange(t_span[0], t_span[1], dt))
        return sol
