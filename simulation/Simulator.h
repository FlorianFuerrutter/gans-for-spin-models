#pragma once
#include <vector>
#include <array>
#include <format>
#include "defines.h"

//------------------

struct SimulationParameter
{
    int nTherm  = 1000;      //updates until termalization
    int nBins   = 1000;      //actual data series size
    int nSweeps = 1000;      //sweeps between bins

    std::string to_string() {return std::format("nTherm: {}, nBins: {}, nSweeps: {}", nTherm, nBins, nSweeps);}
};

class Simulator
{
public:
    Simulator(const SimulationParameter& para) : m_para(para) { }
    std::string parameter_string();

    //setup
    void init_simulation(PRECISION T, PRECISION J);    

    //Monte Carlo     
    void run_monte_carlo(PRECISION J);          
    
    //Wolff Cluster
    void run_wolff_cluster(PRECISION J);

    //store generated data to file
    void store_data();
    
    //------------------
    
    //recieved data
    std::vector<std::array<int8_t, N>> m_states;    //spin states series
    std::vector<PRECISION>             m_energy;    //energy series
    std::vector<PRECISION>             m_m2;        //magnetization squared series

    SimulationParameter m_para;

private:

    //------------------
    //Monte Carlo
     
    //--- HAMILTON USED!! ---
    void precalc_monte_carlo(PRECISION* pFlips, PRECISION T, PRECISION J);                  //--- HAMILTON USED!! ---
    PRECISION calcStateEnergy(const int8_t* state, const uint16_t* nnList, PRECISION J);    //--- HAMILTON USED!! ---      
    void update_monte_carlo(int8_t* state, const uint16_t* nnList, const PRECISION* pFlips, int n); //perform updates

    //------------------
    //Wolff Cluster
    void update_wolff_cluster(int8_t* state, const uint16_t* nnList, int n); //perform updates

    //------------------
    PRECISION m_pAccept = 0;
    PRECISION m_TJ = -1;
};

