import numpy as np
from scipy.integrate import solve_ivp

class PinskyRinzelModel:
    def __init__(self, neuron_type="bursting", synapse_type="NMDA", dt=0.05):
        self.dt = dt
        self.params = self._set_neuron_params(neuron_type)
        self.params.update(self._set_synapse_params(synapse_type))
        self.V_th = 0.0

        self.initial_conditions = {
            'Vs': -65.0,
            'Vd': -65.0,
            'Ca':  0.0,
            'm':  self.alpha_m(-65.0) / (self.alpha_m(-65.0)+self.beta_m(-65.0)),
            'h':  self.alpha_h(-65.0) / (self.alpha_h(-65.0)+self.beta_h(-65.0)),
            'n':  self.alpha_n(-65.0) / (self.alpha_n(-65.0)+self.beta_n(-65.0)),
            's':  self.alpha_s(-65.0) / (self.alpha_s(-65.0)+self.beta_s(-65.0)),
            'c':  self.alpha_c(-65.0) / (self.alpha_c(-65.0)+self.beta_c(-65.0)),
            'q':  self.alpha_q(0.0) / (self.alpha_q(0.0)+self.beta_q(0.0)),
            'Ga': 0.0,
            'Gn': 0.0 
        }

    def _set_neuron_params(self, neuron_type):
        params = {
            'Cm': 3.0, # OK
            'gc': 2.1, # OK
            'p': 0.5,  # OK
            'E_L': -60.0, # OK
            'E_Na': 60.0, # OK
            'E_K': -75.0, # OK
            'E_Ca': 80.0, # OK
            'gL': 0.1, # not specified
            'gNa': 30.0, # OK
            'gKDR': 15.0, # OK
            'gCa': 10.0, # OK
            'gKC': 15.0, # OK
            'gKAHP': 0.8, # OK
            'Is': -0.5, # OK
            'Id': 0.0,  # OK
            'aCa': 0.13, # not specified
            'bCa': 0.075  # not specified
        }

        if neuron_type == "spiking":
            params['gCa'] = 2.5
            params['gKDR'] = 25.0
        elif neuron_type != "bursting":
            raise ValueError("neuron_type must be 'bursting' or 'spiking'")
        return params

    def _set_synapse_params(self, synapse_type):
        params = {
            'VAMPA': 0.0,
            'VNMDA': 0.0,
            'gAMPA': 0.0,
            'gNMDA': 0.0
        }

        if synapse_type == "BOTH":
            params['gAMPA'] = 0.004 # OK
            params['gNMDA'] = 0.01 # OK
        elif synapse_type == "AMPA":
            params['gAMPA'] = 0.01 # OK
            params['gNMDA'] = 0.0 # OK
        else:
            raise ValueError("synapse_type must be 'AMPA' or 'BOTH'")
        return params

    # --- ionic current ---
    def I_L(self, V, gL, E_L): # OK
        return gL * (V - E_L)

    def I_Na(self, Vs, m, h, gNa, E_Na): # OK
        m_inf = self.alpha_m(Vs) / (self.alpha_m(Vs) + self.beta_m(Vs))
        return gNa * m_inf * m_inf * h * (Vs - E_Na)

    def I_K_DR(self, Vs, n, gKDR, E_KDR): # OK
        return gKDR * n * (Vs - E_KDR)

    def I_Ca(self, Vd, s, gCa, E_Ca): # OK
        return gCa * s * s * (Vd - E_Ca)

    def I_K_C(self, Vd, Ca, c, gKC, VK): # OK
        chi_Ca = min(Ca / 250.0, 1.0)
        return gKC * c * chi_Ca * (Vd - VK)

    def I_K_AHP(self, Vd, q, gKAHP, E_KAHP): # OK
        return gKAHP * q * (Vd - E_KAHP)

    # -- open-close(alpha) or close-opne(beta) rate for gating varbiables ---
    def alpha_m(self, Vs): # OK
        return 0.32 * (-46.9 - Vs) / (np.exp((-46.9 - Vs) / 4.0) - 1)

    def beta_m(self, Vs): # OK
        return 0.28 * (Vs + 19.9) / (np.exp((Vs + 19.9) / 5.0) - 1)

    def alpha_h(self, Vs): # OK
        return 0.128 * np.exp((-43.0 - Vs) / 18.0)

    def beta_h(self, Vs): # OK
        return 4.0 / (1.0 + np.exp((-20.0 - Vs) / 5.0))

    def alpha_n(self, Vs): # OK # OK
        return 0.016 * (-24.9 - Vs) / (np.exp((-24.9 - Vs) / 5.0) - 1)

    def beta_n(self, Vs): # OK
        return 0.25 * np.exp((-40.0 - Vs) / 40.0)

    def alpha_s(self, Vd): # OK
        return 1.6 / (1.0 + np.exp(-0.072 * (Vd - 5.0)))

    def beta_s(self, Vd): # OK
        return 0.02 * (Vd + 8.9) / (np.exp((Vd + 8.9) / 5.0) - 1)
        #return 0.02 * (Vd + 8.9) / (np.exp((Vd + 8.9) / 5.0) - 1)

    def alpha_c(self, Vd): # OK
        if Vd < -10.0:
            return (np.exp((Vd + 50.0) / 11.0) - np.exp((Vd + 53.5) / 27.0)) / 18.975
        else:
            return 2.0 * np.exp((-53.5 - Vd) / 27.0)

    def beta_c(self, Vd): # OK
        if Vd < -10.0:
            return 2.0 * np.exp((-53.5 - Vd) / 27.0) - self.alpha_c(Vd)
        else:
            return 0.0

    def alpha_q(self, Ca): # OK
        return min(0.00002 * Ca, 0.01)

    def beta_q(self, Ca): # OK
        return 0.001

    # --- general formalism of time derivative in gating variables ---
    def dy_dt(self, y, alpha_y_func, beta_y_func, *args):
        alpha_y = alpha_y_func(*args)
        beta_y = beta_y_func(*args)
        return alpha_y * (1 - y) - beta_y * y

    # -- synaptic current ---
    def I_AMPA(self, Vd, Ga, VAMPA, gAMPA): # OK
        return gAMPA * Ga * (Vd - VAMPA)

    def I_NMDA(self, Vd, Gn, VNMDA, gNMDA): # OK
        Mg_block = 1.0 / (1.0 + 0.28 * np.exp(-0.062 * Vd))
        return gNMDA * Gn * Mg_block * (Vd - VNMDA)

    def dGa_dt(self, Ga, sa): # OK
        return sa - Ga / 2.0

    def dGn_dt(self, Gn, sn): # OK
        return sn - Gn / 150.0

    def equations(self, t, state_vars, input_signal_val=0.0, input_current_val=0.0):
        Vs, Vd, Ca, m, h, n, s, c, q, Ga, Gn = state_vars
        p = self.params['p']
        Cm = self.params['Cm']
        gc = self.params['gc']
        Is = self.params['Is'] + input_current_val
        Id = self.params['Id']

        sa = input_signal_val
        sn = input_signal_val

        iL_soma = self.I_L(Vs, self.params['gL'], self.params['E_L'])
        iNa = self.I_Na(Vs, m, h, self.params['gNa'], self.params['E_Na'])
        iKDR = self.I_K_DR(Vs, n, self.params['gKDR'], self.params['E_K'])

        iL_dend = self.I_L(Vd, self.params['gL'], self.params['E_L'])
        iCa = self.I_Ca(Vd, s, self.params['gCa'], self.params['E_Ca'])
        iKC = self.I_K_C(Vd, Ca, c, self.params['gKC'], self.params['E_K'])
        iKAHP = self.I_K_AHP(Vd, q, self.params['gKAHP'], self.params['E_K'])

        iAMPA = self.I_AMPA(Vd, Ga, self.params['VAMPA'], self.params['gAMPA'])
        iNMDA = self.I_NMDA(Vd, Gn, self.params['VNMDA'], self.params['gNMDA'])


        dVs_dt = (-iL_soma - iNa - iKDR + (gc / p) * (Vd - Vs) + Is / p) / Cm
        dVd_dt = (-iL_dend - iCa - iKAHP - iKC - (iAMPA + iNMDA) / (1 - p) + (gc / (1 - p)) * (Vs - Vd) + Id / (1 - p)) / Cm

        dm_dt = self.dy_dt(m, self.alpha_m, self.beta_m, Vs)
        dh_dt = self.dy_dt(h, self.alpha_h, self.beta_h, Vs)
        dn_dt = self.dy_dt(n, self.alpha_n, self.beta_n, Vs)
        ds_dt = self.dy_dt(s, self.alpha_s, self.beta_s, Vd)
        dc_dt = self.dy_dt(c, self.alpha_c, self.beta_c, Vd)
        dq_dt = self.dy_dt(q, self.alpha_q, self.beta_q, Ca)

        dCa_dt = -self.params['aCa'] * iCa - self.params['bCa'] * Ca

        dGa_dt_val = self.dGa_dt(Ga, sa)
        dGn_dt_val = self.dGn_dt(Gn, sn)

        return [dVs_dt, dVd_dt, dCa_dt, dm_dt, dh_dt, dn_dt, ds_dt, dc_dt, dq_dt, dGa_dt_val, dGn_dt_val]

    def simulate(self, t_span, y0=None, spike_input_function=None, current_input_function=None):
        if y0 is None:
            y0_list = [
                self.initial_conditions['Vs'],
                self.initial_conditions['Vd'],
                self.initial_conditions['Ca'],
                self.initial_conditions['m'],
                self.initial_conditions['h'],
                self.initial_conditions['n'],
                self.initial_conditions['s'],
                self.initial_conditions['c'],
                self.initial_conditions['q'],
                self.initial_conditions['Ga'],
                self.initial_conditions['Gn']
            ]
            y0 = np.array(y0_list)


        # 外部入力関数を equations に渡すためのラッパー関数
        def ode_func(t, state_vars):
            spike_input   = spike_input_function(t) if spike_input_function else 0.0
            current_input = current_input_function(t) if current_input_function else 0.0
            return self.equations(t, state_vars, spike_input, current_input)

        # solve_ivpは指定されたt_evalで結果を返すため、dtを使って生成
        t_eval = np.arange(t_span[0], t_span[1], self.dt)

        sol = solve_ivp(ode_func, t_span, y0, method='RK45', t_eval=t_eval, rtol=1e-5, atol=1e-8)
        # rtol, atolはデフォルト値か、論文のC++実装の精度に合わせるため適宜調整

        return sol


if __name__ == '__main__':
    # ---  for replication of figure 2 in the original paper ---
    import numpy as np
    import matplotlib.pyplot as plt

    t_span = (0, 6000)
    def zero_input_func(t):
        return 0.0

    def DC_input_func(t):
        if t < 5000:
            return 0.0
        else:
            return 0.5


    print("Start Simulating Bursting Type Neuron...")
    bursting_neuron = PinskyRinzelModel(neuron_type="bursting", synapse_type="AMPA", dt=0.05)
    sol_bursting = bursting_neuron.simulate(t_span, spike_input_function=zero_input_func, current_input_function=DC_input_func)
    print("End Simulating Bursting Type Neuron...")
    
    print("Start Simulating Spiking Type Neuron...")
    spiking_neuron = PinskyRinzelModel(neuron_type="spiking", synapse_type="AMPA", dt=0.05)
    sol_spiking  = spiking_neuron.simulate(t_span, spike_input_function=zero_input_func, current_input_function=DC_input_func)
    print("End Simulating Spiking Type Neuron...")
    
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    t_plot_bursting = sol_bursting.t[sol_bursting.t >= 4600]
    Vs_plot_bursting = sol_bursting.y[0, sol_bursting.t >= 4600]
    #t_plot_bursting = sol_bursting.t
    #Vs_plot_bursting = sol_bursting.y[0, :]
    plt.plot(t_plot_bursting, Vs_plot_bursting, color='blue')
    plt.title('Figure 2 (a) Bursting Type Neuron (Vs)')
    plt.xlabel('Time (msec)')
    plt.ylabel('Vs (mV)')
    plt.ylim([-80, 40])
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    t_plot_spiking = sol_spiking.t[sol_spiking.t >= 4600]
    Vs_plot_spiking = sol_spiking.y[0, sol_spiking.t >= 4600]
    #t_plot_spiking = sol_spiking.t
    #Vs_plot_spiking = sol_spiking.y[0, :]
    plt.plot(t_plot_spiking, Vs_plot_spiking, color='red')
    plt.title('Figure 2 (b) Spiking Type Neuron (Vs)')
    plt.xlabel('Time (msec)')
    plt.ylabel('Vs (mV)')
    plt.ylim([-80, 40])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("output.png")
