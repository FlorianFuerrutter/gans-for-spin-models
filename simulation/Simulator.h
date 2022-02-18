#pragma once
#include <vector>
#include <array>
#include "defines.h"

//------------------

struct SimulationParameter
{
    int nTherm  = 1000;      //updates until termalization
    int nBins   = 1000;        //actual data series size
    int nSweeps = 1000;      //sweeps between bins
};

class Simulator
{
public:
    Simulator(const SimulationParameter& para) : m_para(para) { }

    //setup
    void init_monte_carlo(PRECISION T, PRECISION J);

    //run simulation
    void run_monte_carlo(PRECISION J);
   
    //store generated data to file
    void store_monte_carlo();

    //------------------
    
    //recieved data
    std::vector<std::array<int8_t, N>> m_states; //spin states series
    std::vector<PRECISION>             m_m2;     //magnetization squared series
    std::vector<PRECISION>             m_energy;  //energy series

    SimulationParameter m_para;

private:
    //--- HAMILTON USED!! ---
    void precalc_monte_carlo(PRECISION* pFlips, PRECISION T, PRECISION J);
 
    //--- HAMILTON USED!! ---
    PRECISION calcStateEnergy(const int8_t* state, const uint16_t* nnList, PRECISION J);
    
    void update_monte_carlo(int8_t* state, const uint16_t* nnList, const PRECISION* pFlips, int n);

    //------------------


};

