import numpy as np
import os

#TJs = np.array([1.0, 1.8, 2.0, 2.2, 2.25, 2.3, 2.4, 2.6, 3.4])
#TJs = np.array([1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.26, 2.27, 2.3, 2.4, 2.5, 2.6, 3.0, 3.4])
TJs = np.array([2.28, 2.29, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39])

path = os.path.join(os.path.dirname(__file__), "128")

for TJ in TJs:
    #-----------------
    print("loadtxt states TJ:", TJ)
    file_path = os.path.join(path, "simulation_states_TJ_{TJ}".format(TJ=TJ))
    
    states = np.loadtxt(file_path+".txt", skiprows=1, dtype=np.short)    
    np.save(file_path+".npy", states)
       
    #-----------------
    print("loadtxt observ TJ:", TJ)
    file_path = os.path.join(path, "simulation_observ_TJ_{TJ}".format(TJ=TJ))
    
    observ = np.loadtxt(file_path+".txt", skiprows=1, dtype=np.float32)    
    np.save(file_path+".npy", observ)