import numpy as np

TJs = np.array([1.0, 1.8, 2.0, 2.2, 2.25, 2.3, 2.4, 2.6, 3.4])

for TJ in TJs:
    #-----------------
    print("loadtxt states TJ:", TJ)
    file_path = "simulation_states_TJ_{TJ}".format(TJ=TJ)
    
    states = np.loadtxt(file_path+".txt", skiprows=1, dtype=np.short)    
    np.save(file_path+".npy", states)
       
    #-----------------
    print("loadtxt observ TJ:", TJ)
    file_path = "simulation_observ_TJ_{TJ}".format(TJ=TJ)
    
    observ = np.loadtxt(file_path+".txt", skiprows=1, dtype=np.float32)    
    np.save(file_path+".npy", observ)