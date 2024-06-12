import numpy as np
import matplotlib.pyplot as plt
from HawkesPyLib.simulation import ApproxPowerlawHawkesProcessSimulation

def simulate_hawkes_process(mu_initial, eta, alpha, tau0, m, M, steps, T=500, mu_increment=0.02):
    """
    This Function simulates a univariate Hawkes process with gradually changing parameters.
    
    Parameters:
        mu_initial (float): Initial base rate.
        eta (float): Excitability parameter.
        alpha (float): Scaling factor for the power-law decay.
        tau0 (float): Scale parameter for the decay function.
        m (float): Lower bound of integral for power law.
        M (int): Upper bound of integral for power law.
        steps (int): Number of steps to simulate with increasing base rate.
        T (int): Total time for each simulation.
        mu_increment (float): Increment to apply to the base rate at each step.
    Returns:
        A list containing the timestamps of events for each simulation step.
    """
    mu = mu_initial
    simulations = []
    
    for i in range(steps):
        # Create the simulator with current parameters
        simulator = ApproxPowerlawHawkesProcessSimulation("powlaw", mu, eta, alpha, tau0, m, M)
        timestamps = simulator.simulate(T=T, seed=-1)
        simulations.append(timestamps)
        
        # Plot the current simulation
        plt.figure(figsize=(10, 2))
        plt.eventplot(timestamps, lineoffsets=1, colors='blue')
        plt.title(f'Simulation with Base Intensity: {mu:.2f}')
        plt.xlabel('Time')
        plt.show()
        
        # Increment the base rate for the next simulation
        mu += mu_increment
    
        return timestamps

# Parameters
mu_initial = 0.1
eta = 0.5
alpha = 0.4
tau0 = 0.05
m = 5.0
M = 5
steps = 1

# Simulate the Hawkes process with gradual changes in the base intensity
simulated_data = simulate_hawkes_process(mu_initial, eta, alpha, tau0, m, M, steps)
f = open("two.txt", "w")
f.write(f"Generated data with mu_initial = 0.1, eta = 0.5, alpha = 0.4, tau0 = 0.05, m = 5.0, M = 5\n")
for i in simulated_data : 
    f.write(f"{i}\n")
f.close


# import numpy as np
# import pickle

# def prepare_data_for_thp(timestamps):
#     event_type = 1  # Assuming all events are of the same type for simplicity
#     data_for_thp = []
#     for sequence in timestamps:
#         if len(sequence) == 0:
#             continue  # Skip empty sequences
        
#         time_since_start = np.array(sequence)
#         # Ensure time_since_last_event handles single-event sequences correctly
#         if len(time_since_start) > 1:
#             time_since_last_event = np.diff(time_since_start, prepend=time_since_start[0])
#         else:
#             time_since_last_event = np.array([0])  # No time gap if only one event

#         types = np.full(len(time_since_start), event_type)

#         events = [{
#             'time_since_start': float(ts),
#             'time_since_last_event': float(tsl),
#             'type_event': int(typ)
#         } for ts, tsl, typ in zip(time_since_start, time_since_last_event, types)]
        
#         data_for_thp.append(events)

#     return data_for_thp

# # Simulate and process data
# simulated_data = [simulate_hawkes_process(mu_initial, eta, alpha, tau0, m, M, steps)]
# processed_data = prepare_data_for_thp(simulated_data)

# # Save the processed data
# with open("THP_ready_data.pkl", "wb") as f:
#     pickle.dump(processed_data, f)
