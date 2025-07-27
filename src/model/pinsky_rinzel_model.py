import numpy as np
from scipy.integrate import solve_ivp

class PinskyRinzelModel:
    def __init__(self, num_neurons=1, neuron_type="bursting", synapse_type="AMPA", dt=0.05):
        self.dt = dt
        self.num_neurons = num_neurons
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
        
        self.state_vars_key_to_idx = {
            'Vs': 0,
            'Vd': 1,
            'Ca': 2,
            'm':  3,
            'h':  4,
            'n':  5,
            's':  6,
            'c':  7,
            'q':  8,
            'Ga': 9,
            'Gn': 10,
        }
    def _set_neuron_params(self, neuron_type):
        params_dict = {
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

        params = {}
        for key in params_dict:
            params[key] = np.full(self.num_neurons, params_dict[key])

        for neuron_idx in range(self.num_neurons):
            if neuron_type[neuron_idx] == "spiking":
                params['gCa'][neuron_idx] = 2.5
                params['gKDR'][neuron_idx] = 25.0
            elif neuron_type[neuron_idx] != "bursting":
                raise ValueError("neuron_type must be 'bursting' or 'spiking'")
        return params

    def _set_synapse_params(self, synapse_type):
        params_dict = {
            'VAMPA': 0.0,
            'VNMDA': 0.0,
            'gAMPA': 0.0,
            'gNMDA': 0.0
        }

        params = {}
        for key in params_dict:
            params[key] = np.full(self.num_neurons, params_dict[key])

        for neuron_idx in range(self.num_neurons):
            if synapse_type[neuron_idx] == "BOTH":
                params['gAMPA'][neuron_idx] = 0.004 # OK
                params['gNMDA'][neuron_idx] = 0.01 # OK
            elif synapse_type[neuron_idx] == "AMPA":
                params['gAMPA'][neuron_idx] = 0.01 # OK
                params['gNMDA'][neuron_idx] = 0.0 # OK
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
        chi_Ca = np.minimum(Ca / 250.0, 1.0)
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
        #if Vd < -10.0:
        #    return (np.exp((Vd + 50.0) / 11.0) - np.exp((Vd + 53.5) / 27.0)) / 18.975
        #else:
        #    return 2.0 * np.exp((-53.5 - Vd) / 27.0)
        return np.where(Vd < -10.0, (np.exp((Vd + 50.0) / 11.0) - np.exp((Vd + 53.5) / 27.0)) / 18.975, 2.0 * np.exp((-53.5 - Vd) / 27.0))

    def beta_c(self, Vd): # OK
        #if Vd < -10.0:
        #    return 2.0 * np.exp((-53.5 - Vd) / 27.0) - self.alpha_c(Vd)
        #else:
        #    return 0.0
        return np.where(Vd < -10.0, 2.0 * np.exp((-53.5 - Vd) / 27.0) - self.alpha_c(Vd), 0.0)


    def alpha_q(self, Ca): # OK
        return np.minimum(0.00002 * Ca, 0.01)

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

    def equations(self, t, state_vars, input_signal_val, input_current_val):
        #Vs, Vd, Ca, m, h, n, s, c, q, Ga, Gn = state_vars
        #p = self.params['p']
        #Cm = self.params['Cm']
        #gc = self.params['gc']
        Is = self.params['Is'] + input_current_val
        #Id = self.params['Id']

        sa = input_signal_val
        sn = input_signal_val
        #iL_soma = self.I_L(Vs, self.params['gL'], self.params['E_L'])
        iL_soma = self.I_L(state_vars[:, self.state_vars_key_to_idx['Vs']], self.params['gL'], self.params['E_L'])
        #iNa = self.I_Na(Vs, m, h, self.params['gNa'], self.params['E_Na'])
        iNa = self.I_Na(state_vars[:, self.state_vars_key_to_idx['Vs']], state_vars[:, self.state_vars_key_to_idx['m']], state_vars[:, self.state_vars_key_to_idx['h']], self.params['gNa'], self.params['E_Na'])
        #iKDR = self.I_K_DR(Vs, n, self.params['gKDR'], self.params['E_K'])
        iKDR = self.I_K_DR(state_vars[:, self.state_vars_key_to_idx['Vs']], state_vars[:, self.state_vars_key_to_idx['n']], self.params['gKDR'], self.params['E_K'])

        #iL_dend = self.I_L(Vd, self.params['gL'], self.params['E_L'])
        iL_dend = self.I_L(state_vars[:, self.state_vars_key_to_idx['Vd']], self.params['gL'], self.params['E_L'])
        #iCa = self.I_Ca(Vd, s, self.params['gCa'], self.params['E_Ca'])
        iCa = self.I_Ca(state_vars[:, self.state_vars_key_to_idx['Vd']], state_vars[:, self.state_vars_key_to_idx['s']], self.params['gCa'], self.params['E_Ca'])
        #iKC = self.I_K_C(Vd, Ca, c, self.params['gKC'], self.params['E_K'])
        iKC = self.I_K_C(state_vars[:, self.state_vars_key_to_idx['Vd']], state_vars[:, self.state_vars_key_to_idx['Ca']], state_vars[:, self.state_vars_key_to_idx['c']], self.params['gKC'], self.params['E_K'])
        #iKAHP = self.I_K_AHP(Vd, q, self.params['gKAHP'], self.params['E_K'])
        iKAHP = self.I_K_AHP(state_vars[:, self.state_vars_key_to_idx['Vd']], state_vars[:, self.state_vars_key_to_idx['q']], self.params['gKAHP'], self.params['E_K'])

        #iAMPA = self.I_AMPA(Vd, Ga, self.params['VAMPA'], self.params['gAMPA'])
        iAMPA = self.I_AMPA(state_vars[:, self.state_vars_key_to_idx['Vd']], state_vars[:, self.state_vars_key_to_idx['Ga']], self.params['VAMPA'], self.params['gAMPA'])
        #iNMDA = self.I_NMDA(Vd, Gn, self.params['VNMDA'], self.params['gNMDA'])
        iNMDA = self.I_NMDA(state_vars[:, self.state_vars_key_to_idx['Vd']], state_vars[:, self.state_vars_key_to_idx['Gn']], self.params['VNMDA'], self.params['gNMDA'])


        #dVs_dt = (-iL_soma - iNa - iKDR + (gc / p) * (Vd - Vs) + Is / p) / Cm
        dVs_dt = (-iL_soma - iNa - iKDR + (self.params['gc'] / self.params['p']) * (state_vars[:, self.state_vars_key_to_idx['Vd']] - state_vars[:, self.state_vars_key_to_idx['Vs']]) + Is / self.params['p']) / self.params['Cm']
        #dVd_dt = (-iL_dend - iCa - iKAHP - iKC - (iAMPA + iNMDA) / (1 - p) + (gc / (1 - p)) * (Vs - Vd) + Id / (1 - p)) / Cm
        dVd_dt = (-iL_dend - iCa - iKAHP - iKC - (iAMPA + iNMDA) / (1 - self.params['p']) + (self.params['gc'] / (1 - self.params['p'])) * (state_vars[:, self.state_vars_key_to_idx['Vs']] - state_vars[:, self.state_vars_key_to_idx['Vd']]) + self.params['Id'] / (1 - self.params['p'])) / self.params['Cm']

        dm_dt = self.dy_dt(state_vars[:, self.state_vars_key_to_idx['m']], self.alpha_m, self.beta_m, state_vars[:, self.state_vars_key_to_idx['Vs']])
        dh_dt = self.dy_dt(state_vars[:, self.state_vars_key_to_idx['h']], self.alpha_h, self.beta_h, state_vars[:, self.state_vars_key_to_idx['Vs']])
        dn_dt = self.dy_dt(state_vars[:, self.state_vars_key_to_idx['n']], self.alpha_n, self.beta_n, state_vars[:, self.state_vars_key_to_idx['Vs']])
        ds_dt = self.dy_dt(state_vars[:, self.state_vars_key_to_idx['s']], self.alpha_s, self.beta_s, state_vars[:, self.state_vars_key_to_idx['Vd']])
        dc_dt = self.dy_dt(state_vars[:, self.state_vars_key_to_idx['c']], self.alpha_c, self.beta_c, state_vars[:, self.state_vars_key_to_idx['Vd']])
        dq_dt = self.dy_dt(state_vars[:, self.state_vars_key_to_idx['q']], self.alpha_q, self.beta_q, state_vars[:, self.state_vars_key_to_idx['Ca']])

        dCa_dt = -self.params['aCa'] * iCa - self.params['bCa'] * state_vars[:, self.state_vars_key_to_idx['Ca']]

        #dGa_dt_val = self.dGa_dt(Ga, sa)
        #dGn_dt_val = self.dGn_dt(Gn, sn)
        dGa_dt_val = self.dGa_dt(state_vars[:, self.state_vars_key_to_idx['Ga']], sa)
        dGn_dt_val = self.dGn_dt(state_vars[:, self.state_vars_key_to_idx['Gn']], sn)

        #return np.array([dVs_dt, dVd_dt, dCa_dt, dm_dt, dh_dt, dn_dt, ds_dt, dc_dt, dq_dt, dGa_dt_val, dGn_dt_val])
        return np.transpose(np.stack([dVs_dt, dVd_dt, dCa_dt, dm_dt, dh_dt, dn_dt, ds_dt, dc_dt, dq_dt, dGa_dt_val, dGn_dt_val]))

    def runge_kutta4(self, func, t_span, y0, t_eval, spike_input_function, current_input_function):
        y_history = np.zeros((np.size(t_eval), self.num_neurons, len(self.state_vars_key_to_idx)))
        y_history[0, :, :] = y0
        current_y       = y0

        for t_idx in range(np.size(t_eval)-1):
            dt = t_eval[t_idx+1] - t_eval[t_idx]
            k1 = np.array(func(t_eval[t_idx],        current_y            ,  spike_input_function, current_input_function))
            k2 = np.array(func(t_eval[t_idx] + dt/2, current_y + dt/2 * k1,  spike_input_function, current_input_function))
            k3 = np.array(func(t_eval[t_idx] + dt/2, current_y + dt/2 * k2,  spike_input_function, current_input_function))
            k4 = np.array(func(t_eval[t_idx] + dt,   current_y + dt   * k3,  spike_input_function, current_input_function))
            current_y = current_y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            y_history[t_idx+1, :, :] = current_y

        class OdeResultMimic:
            def __init__(self, t, y):
                self.t = t
                self.y = y_history

        sol = OdeResultMimic(t_eval, y_history)
        return sol

    
    def simulate(self, t_span, y0_dict=None, spike_input_function=None, current_input_function=None):
        # set intitial value(need further modification for heterogeneous initial value)
        if y0_dict is None:
            y0_dict = {
                'Vs': self.initial_conditions['Vs'],
                'Vd': self.initial_conditions['Vd'],
                'Ca': self.initial_conditions['Ca'],
                'm': self.initial_conditions['m'],
                'h': self.initial_conditions['h'],
                'n': self.initial_conditions['n'],
                's': self.initial_conditions['s'],
                'c': self.initial_conditions['c'],
                'q': self.initial_conditions['q'],
                'Ga': self.initial_conditions['Ga'],
                'Gn': self.initial_conditions['Gn']
            }
        
        y0 = np.zeros((self.num_neurons, len(y0_dict)))
        for key in y0_dict:
            y0[:, self.state_vars_key_to_idx[key]] = np.full(self.num_neurons, y0_dict[key])

        # translate spike input and current input into ODE
        def ode_func(t, state_vars, spike_input_function, current_input_function):
            spike_input   = spike_input_function(t) if spike_input_function else np.zeros(self.num_neurons)
            current_input = current_input_function(t) if current_input_function else np.zeros(self.num_neurons)
            return self.equations(t, state_vars, spike_input, current_input)

        t_eval = np.arange(t_span[0], t_span[1], self.dt)

        #sol = solve_ivp(ode_func, t_span, y0, args=(spike_input_function, current_input_function), method='RK45', t_eval=t_eval, rtol=1e-5, atol=1e-8) # for scipy.integrate.solve_ivp
        sol = self.runge_kutta4(ode_func, t_span, y0, t_eval, spike_input_function, current_input_function)

        return sol

    def count_spikes_in_trace(self, Vs_trace):
        spikes = 0
        above_threshold = Vs_trace >= self.V_th
        spike_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0]
        spikes = len(spike_indices)
        return spikes

if __name__ == '__main__':
    # ---  for replication of figure 2 in the original paper ---
    import numpy as np
    import matplotlib.pyplot as plt

    num_neurons = 2

    t_span = (0, 6000)
    def zero_input_func(t):
        return np.zeros(num_neurons)

    def DC_input_func(t):
        if t < 5000:
            return np.zeros(num_neurons)
        else:
            return np.full(num_neurons, 0.5)

    print("Start Simulating Neuron...")
    neuron = PinskyRinzelModel(num_neurons, neuron_type=["bursting", "spiking"], synapse_type=["AMPA", "AMPA"], dt=0.05)
    sol = neuron.simulate(t_span, spike_input_function=zero_input_func, current_input_function=DC_input_func)
    print("End Simulating Neuron...")
    
    #print("Start Simulating Spiking Type Neuron...")
    #spiking_neuron = PinskyRinzelModel(neuron_type=["spiking"], synapse_type=["AMPA"], dt=0.05)
    #sol_spiking  = spiking_neuron.simulate(t_span, spike_input_function=zero_input_func, current_input_function=DC_input_func)
    #print("End Simulating Spiking Type Neuron...")
    
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    t_plot_bursting = sol.t[sol.t >= 4600]
    Vs_plot_bursting = sol.y[sol.t >= 4600, 0, 0]
    #t_plot_bursting = sol_bursting.t
    #Vs_plot_bursting = sol_bursting.y[0, :]
    plt.plot(t_plot_bursting, Vs_plot_bursting, color='blue')
    plt.title('Figure 2 (a) Bursting Type Neuron (Vs)')
    plt.xlabel('Time (msec)')
    plt.ylabel('Vs (mV)')
    plt.ylim([-80, 40])
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    #print(np.shape(sol_spiking.y))
    t_plot_spiking = sol.t[sol.t >= 4600]
    Vs_plot_spiking = sol.y[sol.t >= 4600, 1, 0]
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
