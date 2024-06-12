# import numpy as np
# import matplotlib.pyplot as plt
# from HawkesPyLib.simulation import ApproxPowerlawHawkesProcessSimulation
# import pickle

# train_length = 0
# test_length = 0


# def simulate_hawkes_process(mu_initial, eta, alpha, tau0, m, M, steps, T=5000, mu_increment=0.02):
#     mu = mu_initial
#     simulations = []
    
#     for i in range(steps):
#         simulator = ApproxPowerlawHawkesProcessSimulation("powlaw", mu, eta, alpha, tau0, m, M)
#         timestamps = simulator.simulate(T=T, seed=-1)
#         if timestamps.size > 0 and i == steps -1 :
#             simulations.append(timestamps)
#         mu += mu_increment

#     return simulations

# def prepare_data_for_thp(simulations):
#     data_for_thp = []
#     minimum_length = 10000000

#     for sequence in simulations:
#         minimum_length = min(minimum_length, len(sequence))
#     print(minimum_length)
#     for sequence in simulations:
#         if sequence.size == 0:  # Check if the sequence is empty
#             continue

#         sequence_data = []
#         for i in range(minimum_length): 
#             sequence_data.append({'time_since_start': sequence[i], 'time_since_last_event': 0, 'type_event': 1})
#         data_for_thp.append(sequence_data)
    
#     return data_for_thp



# # def save_data(data, filename):
# #     with open(filename, 'wb') as f:
# #         pickle.dump(data, f)

# def save_data(data, filename):
#     with open(filename, 'wb') as f:
#         # Saving as a dictionary with necessary metadata
#         pickle.dump({'data': data, 'dim_process': 1}, f)

# num_sequences = 100
# train_split = 0.8
# all_simulated_data = []
# for _ in range(num_sequences):
#     # Each call generates multiple sequences
#     sequences = simulate_hawkes_process(0.1, 0.5, 0.4, 0.05, 5.0, 5, np.random.randint(1, 10))
#     all_simulated_data.extend(sequences)  # Flatten the list

# formatted_data = prepare_data_for_thp(all_simulated_data)

# split_index = int(len(formatted_data) * train_split)
# train_data, test_data = formatted_data[:split_index], formatted_data[split_index:]

# # save_data(train_data, 'train_data.pkl')
# # save_data(test_data, 'test_data.pkl')



import numpy as np
import matplotlib.pyplot as plt
from HawkesPyLib.simulation import ApproxPowerlawHawkesProcessSimulation
import pickle


def simulate_hawkes_process():
    # Randomize parameters within reasonable ranges
    mu_initial = np.random.uniform(0.1, 0.5)  # Base rate
    eta = np.random.uniform(0.2, 1.0)  # Excitability
    alpha = np.random.uniform(0.2, 0.8)  # Scaling factor for decay
    tau0 = np.random.uniform(0.05, 0.2)  # Decay time scale
    m = 3.0  # Lower bound (fixed for simplicity)
    M = 10  # Upper bound (fixed for simplicity)
    T = 5000  # Total simulation time
    
    simulator = ApproxPowerlawHawkesProcessSimulation("powlaw", mu_initial, eta, alpha, tau0, m, M)
    timestamps = simulator.simulate(T=T, seed=-1)
    
    return timestamps if timestamps.size > 0 else None

def prepare_data_for_thp(timestamps):
    if timestamps.size == 0:
        return None
    sequence_data = [{'time_since_start': ts, 'time_since_last_event': ts - timestamps[i - 1] if i > 0 else 0, 'type_event': int(1)}
                     for i, ts in enumerate(timestamps)]
    return sequence_data

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

num_sequences = 100

train_data = [simulate_hawkes_process() for _ in range(num_sequences)]
train_data = [data for data in train_data if data is not None] 
min_train_length = 10000000
for data in train_data : 
    min_train_length = min(min_train_length, len(data))
train_data = [data[:min_train_length] for data in train_data] 
train_pickle = {"train" : [prepare_data_for_thp(data) for data in train_data], "test": []}
# print(min_train_length)
print(len(train_pickle['train']))
print(len(train_pickle['train'][0]))
print(len(train_pickle['train'][0][0]))
print((train_pickle['train'][0][0]))

test_data = [simulate_hawkes_process() for _ in range(num_sequences)]
test_data = [data for data in test_data if data is not None] 
min_test_length = 10000000
for data in test_data : 
    min_test_length = min(min_test_length, len(data))
test_data = [data[:min_test_length] for data in test_data] 
test_pickle = {"train" : [], "test": [prepare_data_for_thp(data) for data in test_data]}
# print(min_test_length)

# save_data(train_pickle, './Test_data/train.pkl')
# save_data(test_pickle, './Test_data/test.pkl')
