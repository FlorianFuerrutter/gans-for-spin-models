#pragma once
#include <iostream>

#include "Simulator.h"

//------------------

int main()
{
    srand(std::time(nullptr));

    PRECISION T = 1;
    PRECISION J = 1;


    //------------------

    SimulationParameter para = { N_TERM, N_BINS, N_SWEEPS };

    Simulator sim(para);
    sim.init_monte_carlo(T, J);
    sim.run_monte_carlo(J);

    for (int i = 0; i < 10; i++)
        std::cout << sim.m_m2[i] << std::endl;  
    
    std::cout << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << sim.m_energy[i] << std::endl;

    //sim.store_monte_carlo();

    //------------------


    return 0;
}
