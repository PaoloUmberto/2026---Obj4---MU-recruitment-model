import numpy as np
import scipy.linalg as la
import control

#Function to simulate coupled spiking control  #############################################################################################
def simulate_coupled_spiking_control(nT, dt, t_fut, M, N, A, B, C, z, x, x0, lam, a, mu, ref_period_lenght, delay=True, kill=False, **kwargs):
    """Run the full simulation over nT timesteps with spike-based control on coupled oscillator systems (arbitrary dimensions).
    
        Parameters:
        ----------
        nT : int
            Number of timesteps in the simulation.
        dt : float
            Simulation timestep (time increment per step).
        t_fut : float
            Prediction time horizon for the control.
        M : int
            Number of total controllable entities downstream (for instance 10 --> 2 dimension muscle + 8 MUs).
        N : int
            Number of neurons in the network.
        A : array
            System dynamics matrix (state transition matrix).
        B : array
            Control input matrix (maps control signals to state changes).
        C : array
            Cost matrix (introduces a spike cost on velocity).
        x0 : array
            Initial state of the system.
        lam : float
            Decay factor for the firing rates (controls how quickly firing rates decay).
        a : float
            Adjusts the adaptive thresholds based on past spiking activity of each neuron (only if delay is True).
        mu : float
            Scaling factor for the threshold.
        ref_period_lenght : float
            Length of the refractory period for neurons (time during which neurons cannot spike again, usually set to zero).
        delay : bool
            If True, introduces adaptive thresholds for each neuron (helps the network deal with delayed signals).
        Zstep : bool
            If True, uses a step-based target trajectory; otherwise, uses a sigmoid-based trajectory (default is False).
        **kwargs : dict
        Additional parameters for the target:
        - z_base : array, required if Zstep=True
            Base trajectory for the target when using step-based control.
        - leak_z : float, required if Zstep=True
            Leak factor for the target trajectory when using step-based control.
        - z_target : float, required if Zstep=False
            Final value reached by the target trajectory when using sigmoid-based control.
        - theta : float, required if Zstep=False
            Decay factor for the sigmoid-based target trajectory.

        Returns:
        -------
        results : dict
        A dictionary containing the simulation results, including states, control signals, and spiking activity.
    """
    #Pre-compute matrices and initial values

    #A matrix exponential
    Af = la.expm(A*t_fut)
    #Af = Af * np.array([1, 0])[:, None]

    # Initialize matrices
    x[:,0] = x0
    x_pred = np.zeros((2*M, nT)) #Predicted state (only used for plotting)
    V = np.zeros((N, nT))
    r = np.zeros((N, nT))
    s = np.zeros((N, nT))
    Th = np.zeros((N, nT))
    #Errors 
    error_spiking = np.zeros((len(x[:, 0]), nT)) #(only used for plotting)
    pred_error_spiking = np.zeros((len(x[:, 0]), nT)) #(only used for plotting)

    #Time settings
    ref_period = t_fut * ref_period_lenght
    can_spike = np.ones(N)
    timer = np.ones(N) * ref_period

    # Set up target based on the parameters provided. z_pre is one pre-defined smooth increase.
    z_base = kwargs['z_base']
    leak_z = kwargs['leak_z']

    # Run simulation
    for i in range(1, nT):

        odd_idx = np.arange(1, 2*M, 2)
        z_base[[1, 3], 5000:] = 5
        z[odd_idx, i] = z[odd_idx, i-1] + dt * leak_z * (z_base[odd_idx, i-1] - z[odd_idx, i-1])

        # Update system states
        x[:, i] = x[:, i-1] + dt * (A @ x[:, i-1]) + B @ s[:, i-1]

        # Voltage Update
        V[:, i] = B.T @ Af.T @ C @ (z[:, i] - Af @ x[:, i])
                  
        # Update rates
        r[:, i] = r[:, i-1] + dt * (-lam * r[:, i-1]) + s[:, i-1]

        #Neuron killing
        if kill == True: 
            killed_neurons = np.zeros((N, nT), dtype=int)
            k_per_step = kwargs['k_per_step']
            cell_death_timings = kwargs['cell_death_timings']
        
            if i in cell_death_timings:
                #print(f"Timestep {i}: Killing neurons.")
                # Obtain indices of alive neurons
                alive_neurons = np.where(killed_neurons[:, i - 1] == 0)[0]
                # Identify the k_per_step neurons
                neurons_to_kill = alive_neurons[-k_per_step:]
                # Update the kill_neruons to mark these neurons as killed for the rest of the simulation
                killed_neurons[neurons_to_kill, i:] = 1
                #print(f"Timestep {i}: Neurons killed: {neurons_to_kill}")
            # Apply the kill mask to zero out voltages and spikes for killed neurons
            V[killed_neurons[:,i] == 1, i]  = 0

        # Spike determination
        if delay == True:
            # Update Adaptive Threshold (scales for each neuron based on its past activity)
            Th[:, i] = (np.diag(B.T @ Af.T @ C @ Af @ B)+mu) / 2 + a * (r[:, i] + 0.5)
            # Update Predicted State (plotting stuff)
            x_pred[:, i] = Af @ x[:, i]
            #Register Spikes
            abovethreshold = np.where(np.logical_and(V[:, i] > Th[:, i], can_spike))[0]
            if len(abovethreshold) > 0:
                s[abovethreshold, i] = 1
                #Apply Refractory Period
                can_spike[abovethreshold] = 0
                timer[abovethreshold] = ref_period
            timer[can_spike == 0] -= dt
            can_spike[timer <= 0] = 1

        if delay == False:
            # Update Non-Adaptive Threshold
            Th = np.diag((B.T @ Af.T @ C @ Af @ B) + mu)/2 
            x_pred[:, i] = Af @ x[:, i]
            #One Neuron Spikes at a time
            abovethreshold = np.where(np.logical_and(V[:, i] > Th, can_spike))[0]  
            if len(abovethreshold) > 0:
                maxid = np.argmax(V[abovethreshold, i])
                #Choose one spiking neuron, can also use maxid to choose the highest voltage one
                tospike = abovethreshold[np.random.randint(len(abovethreshold))] 
                s[tospike, i] = 1
                #Apply Refractory Period
                can_spike[tospike] = 0
                timer[tospike] = ref_period
            timer[can_spike == 0] -= dt
            can_spike[timer <= 0] = 1

        # Compute error (plotting stuff)
        even_dims = range(0, x.shape[0], 2)
        for j in even_dims:
            error_spiking[j, i] = (x[j, i] - z[j, i])
            pred_error_spiking[j, i] = (x_pred[j, i] - z[j, i])
        
        # error_spiking[:, i] = (x[:, i] - z[0, i])
        # pred_error_spiking[:, i] = (x_pred[:, i] - z[0, i])

    return z, s, x, V, Th, x_pred, error_spiking, pred_error_spiking

#Function to simulate spiking control of a single SMD (just in case)  #############################################################################################
def simulate_spiking_control(nT, dt, t_fut, N, A, B, C, x0, lam, a, mu, ref_period_lenght, delay=True, Zstep=False, perturb=False, noise=False, **kwargs):
    """Run the full simulation over nT timesteps with spike-based control.
    
        Parameters:
        ----------
        nT : int
            Number of timesteps in the simulation.
        dt : float
            Simulation timestep (time increment per step).
        t_fut : float
            Prediction time horizon for the control.
        N : int
            Number of neurons in the network.
        A : array
            System dynamics matrix (state transition matrix).
        B : array
            Control input matrix (maps control signals to state changes).
        C : array
            Cost matrix (introduces a spike cost on velocity).
        x0 : array
            Initial state of the system.
        lam : float
            Decay factor for the firing rates (controls how quickly firing rates decay).
        a : float
            Adjusts the adaptive thresholds based on past spiking activity of each neuron (only if delay is True).
        mu : float
            Scaling factor for the threshold.
        ref_period_lenght : float
            Length of the refractory period for neurons (time during which neurons cannot spike again).
        delay : bool
            If True, introduces adaptive thresholds for each neuron (helps the network deal with delayed signals).
        Zstep : bool
            If True, uses a step-based target trajectory; otherwise, uses a sigmoid-based trajectory (default is False).
        **kwargs : dict
        Additional parameters for the target:
        - z_base : array, required if Zstep=True
            Base trajectory for the target when using step-based control.
        - leak_z : float, required if Zstep=True
            Leak factor for the target trajectory when using step-based control.
        - z_target : float, required if Zstep=False
            Final value reached by the target trajectory when using sigmoid-based control.
        - theta : float, required if Zstep=False
            Decay factor for the sigmoid-based target trajectory.

        Returns:
        -------
        results : dict
        A dictionary containing the simulation results, including states, control signals, and spiking activity.
    """
    #Pre-compute matrices and initial values

    #A matrix exponential
    Af = la.expm(A*t_fut)
    #Af = Af * np.array([1, 0])[:, None]

    # Initialize matrices
    z = np.zeros((2, nT))
    x = np.zeros((2, nT))
    x[:,0] = x0
    x_pred = np.zeros((2, nT)) #Predicted state
    V = np.zeros((N, nT))
    r = np.zeros((N, nT))
    s = np.zeros((N, nT))
    Th = np.zeros((N, nT))
    #Errors 
    error_spiking = np.zeros((len(x[:, 0]), nT))
    pred_error_spiking = np.zeros((len(x[:, 0]), nT))

    #Time settings
    t0 = nT / 2.5
    ref_period = t_fut * ref_period_lenght
    can_spike = np.ones(N)
    timer = np.ones(N) * ref_period
   

    # Set up target based on the parameters provided
    if Zstep:
        if 'z_base' not in kwargs or 'leak_z' not in kwargs:
            raise ValueError("When Zstep is True, 'z_base' and 'leak_z' must be provided.")
        z_base = kwargs['z_base']
        leak_z = kwargs['leak_z']
    else:
        if 'z_target' not in kwargs or 'theta' not in kwargs:
            raise ValueError("When Zstep is False, 'z_target' and 'theta' must be provided.")
        z_target = kwargs['z_target']
        theta = kwargs['theta']

    # Run simulation
    for i in range(1, nT):

        # Update target 
        if Zstep:
            z[0, i] = z[0, i-1] + dt * (leak_z * (z_base[0, i-1] - z[0, i-1]))
        else:
            z[0, i] = z_target / (1 + np.exp(-theta * (i - t0)))

        # Update system states
        x[:, i] = x[:, i-1] + dt * (A @ x[:, i-1]) + B @ s[:, i-1]

        if perturb:
            x[1, 1500:1600] = 6
            x[1, 3000:3100] = -6

        # Voltage Update
        V[:, i] = B.T @ Af.T @ C @ (z[:, i] - Af @ x[:, i])
        if noise == True and i > nT/2.5:  # add noise after some time steps 
            sigma = kwargs['sigma'] #noise level for voltage update
            vnoise = sigma * np.random.randn(V.shape[0])
            V[:, i] += vnoise

        # Update rates
        r[:, i] = r[:, i-1] + dt * (-lam * r[:, i-1]) + s[:, i-1]

        # Old Threshold Update
        # Th[:, i] = (np.diag(D.T @ Af.T @ Af @ D) / 2) + a * (2 * r[:, i] + 0.5)

        #Update Predicted State
        x_pred[:, i] = Af @ x[:, i]

        # Spike determination
        if delay == True:
            #Update Adaptive Threshold
            Th[:, i] = ((np.diag(B.T @ Af.T @ C @ Af @ B)+mu) + a * (r[:, i] + 0.5)) /2
            #Register Spikes
            abovethreshold = np.where(np.logical_and(V[:, i] > Th[:, i], can_spike))[0]
            if len(abovethreshold) > 0:
                s[abovethreshold, i] = 1
                #Apply Refractory Period
                can_spike[abovethreshold] = 0
                timer[abovethreshold] = ref_period
            timer[can_spike == 0] -= dt
            can_spike[timer <= 0] = 1

        if delay == False:
            # Update Non-Adaptive Threshold
            Th = np.diag((B.T @ Af.T @ C @ Af @ B) + mu)/2 
            x_pred[:, i] = Af @ x[:, i]
            #One Neuron Spikes at a time
            abovethreshold = np.where(np.logical_and(V[:, i] > Th, can_spike))[0]  
            if len(abovethreshold) > 0:
                maxid = np.argmax(V[abovethreshold, i])
                #Choose one random spiking neuron
                #tospike = abovethreshold[np.random.randint(len(abovethreshold))]
                #Use maxid to choose the highest voltage one
                tospike = abovethreshold[maxid] 
                s[tospike, i] = 1
                #Apply Refractory Period
                #can_spike[tospike] = 0
                #timer[tospike] = ref_period
            #timer[can_spike == 0] -= dt
            #can_spike[timer <= 0] = 1
        
        # Compute error
        error_spiking[:, i] = (x[:, i] - z[0, i])
        pred_error_spiking[:, i] = (x_pred[:, i] - z[0, i])


    return z, s, x, V, Th, x_pred, error_spiking, pred_error_spiking

