#pragma once
#include <iostream>
#include <chrono>

#include "Simulator.h"

//------------------

int main()
{
    srand(std::time(nullptr));

    PRECISION T = 2.2;
    PRECISION J = 1;

    //------------------

    SimulationParameter para = { N_TERM, N_BINS, N_SWEEPS };

    Simulator sim(para);
    sim.init_simulation(T, J);
    std::cout << "[SimulationParameter] " << sim.parameter_string() << std::endl;

    //sim.run_monte_carlo(J);
    sim.run_wolff_cluster(J);

    //------------------
    //Performance test

    #if 0
        int count = 10;
        double mean = 0;
        for (int i = 0; i < count; i++)
        {
            const auto t0 = std::chrono::steady_clock::now();
            //sim.run_monte_carlo(J);
            sim.run_wolff_cluster(J);
            mean += (std::chrono::steady_clock::now() - t0).count() * 1e-6;
        }
        mean /= count;
        std::cout << "mean [ms]: " << mean << std::endl << std::endl;
    #endif 
  
    //------------------

    std::cout << "Sample m2:" << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << sim.m_m2[i] << std::endl;  
    
    std::cout << std::endl << "Sample energy:" << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << sim.m_energy[i] << std::endl;

    //------------------

    sim.store_data();
    return 0;
}
