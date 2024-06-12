import numpy as np
import matplotlib.pyplot as plt
from HawkesPyLib.simulation import ApproxPowerlawHawkesProcessSimulation

def simulate_gradual_hawkes_process(mu_initial, eta_initial, alpha, tau0, m, M, total_time, segments, amplitude, frequency):
    """
    Simulate a univariate Hawkes process with parameters changing according to a sinusoidal function.
    
    Parameters:
        mu_initial (float): Initial base rate.
        eta_initial (float): Initial excitability parameter.
        alpha (float): Scaling factor for the power-law decay.
        tau0 (float): Scale parameter for the decay function.
        m (float): Lower bound of integral for power law.
        M (int): Upper bound of integral for power law.
        total_time (int): Total simulation time.
        segments (int): Number of segments to divide the total time.
        amplitude (float): Amplitude of the sinusoidal change.
        frequency (float): Frequency of the sinusoidal change.
    
    Returns:
        A list containing all timestamps from the simulated process.
    """
    timestamps_global = []
    current_start_time = 0
    segment_length = total_time / segments

    for i in range(segments):
        # Calculate parameters for current segment
        t = (i / segments) * total_time
        mu = mu_initial + amplitude * np.sin(2 * np.pi * frequency * t / total_time)
        eta = eta_initial + amplitude * np.sin(2 * np.pi * frequency * t / total_time)

        # Create the simulator with current parameters
        simulator = ApproxPowerlawHawkesProcessSimulation("powlaw", mu, eta, alpha, tau0, m, M)
        timestamps = simulator.simulate(T=segment_length, seed=np.random.randint(10000))

        # Adjust timestamps to continue from the end of the last segment
        timestamps = [t + current_start_time for t in timestamps]
        timestamps_global.extend(timestamps)
        current_start_time += segment_length

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(timestamps_global)), timestamps_global, marker='o', s=5, alpha=0.6)
    plt.title('Gradual Hawkes Process Simulation with Sinusoidal Parameter Changes')
    plt.xlabel('Event Index')
    plt.ylabel('Timestamp')
    plt.grid(True)
    plt.show()

    return timestamps_global

# Example usage
mu_initial = 0.08
eta_initial = 0.2
alpha = 0.5
tau0 = 0.05
m = 2
M = 1
total_time = 5000
segments = 500
amplitude = 0.01  # Amplitude of the sinusoidal parameter modulation
frequency = 1   # Frequency of the sinusoidal changes (number of full cycles over the total time)

timestamps = simulate_gradual_hawkes_process(mu_initial, eta_initial, alpha, tau0, m, M, total_time, segments, amplitude, frequency)
