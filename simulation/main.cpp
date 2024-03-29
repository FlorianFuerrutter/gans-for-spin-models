#pragma once
#include <iostream>
#include <chrono>

#include "Simulator.h"

//------------------

int main()
{
    srand(std::time(nullptr));
    //srand(1);

    PRECISION J = 1;
    //std::array<PRECISION, 9> Ts = { 1.0, 1.8, 2.0, 2.2, 2.25, 2.3, 2.4, 2.6, 3.4 };

    // for corr
    //std::array<PRECISION, 15> Ts = { 1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.26, 2.27, 2.3, 2.4, 2.5, 2.6, 3.0, 3.4 };
    //std::array<PRECISION, 11> Ts = { 2.28, 2.29, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39 };

    //THE real 15
    std::array<PRECISION, 15> Ts = { 1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.3, 2.35, 2.4, 2.5, 2.6, 2.8, 3.0, 3.4 };

    //std::array<PRECISION, 6> Ts = {1.5, 2.1, 2.35, 2.5, 2.8, 3.0};

    //std::array<PRECISION, 1> Ts = { 3.4 };

    SimulationParameter para = { N_TERM, N_BINS, N_SWEEPS };

    //------------------

    Simulator sim(para);

    for (int i = 0; i < Ts.size(); i++)
    {
        PRECISION T = Ts[i];
        if (T <= 0) continue;

        sim.init_simulation(T, J);
        std::cout << "[SimulationParameter] " << sim.parameter_string() << std::endl;

        //sim.run_monte_carlo();
        sim.run_wolff_cluster();

        sim.store_data();
    }

    //------------------

    std::cout << std::endl << "Sample energy:" << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << sim.m_energy[i] << std::endl;

    std::cout << "Sample mAbs:" << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << sim.m_mAbs[i] << std::endl;
    
    //------------------
    //Performance test

    #if 0
        int count = 10;
        double mean = 0;
        for (int i = 0; i < count; i++)
        {
            const auto t0 = std::chrono::steady_clock::now();
            //sim.run_monte_carlo();
            sim.run_wolff_cluster();
            mean += (std::chrono::steady_clock::now() - t0).count() * 1e-6;
        }
        mean /= count;
        std::cout << "mean [ms]: " << mean << std::endl << std::endl;
    #endif 
  
    //------------------

    return 0;
}
