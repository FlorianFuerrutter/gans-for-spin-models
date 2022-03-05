#include <numeric>
#include <random>

#include "Simulator.h"
#include "helper.h"
#include "storage.h"

//------------------

namespace
{
    //compile time for max performance
    static   int8_t  state[N];        //actual system state, spins +-1
    static uint16_t  nnList[N * 4];   //nn list
    static PRECISION pFlips[9];       //precalculated monte carlo values
}

//------------------

std::string Simulator::parameter_string()
{
    std::string tj = std::format("T/J: {}, (LX, LY): ({}, {}), ", m_TJ, LX, LY);
    return tj + m_para.to_string();
}

void Simulator::init_simulation(PRECISION T, PRECISION J)
{
    m_TJ = T / J;
    m_J = J;
    m_T = T;

    //calc expensive values before
    precalc_monte_carlo(pFlips, T, J);   

    //--- HAMILTON USED!! ---
    m_pAccept = 1.0 - exp(-2.0 * J / T); //NOTE: only static if no external fields, else create array!!

    //set random spins +-1
    randomize_state(state, N);

    //gen the next neighbor list
    generate_2D_NNList(LX, LY, nnList);
}

void Simulator::run_monte_carlo()
{
    //resize to requested bins count
    m_states.resize(m_para.nBins);
    m_energy.resize(m_para.nBins);
    m_m.resize(m_para.nBins);
    m_mAbs.resize(m_para.nBins);
    m_m2.resize(m_para.nBins);
    m_m4.resize(m_para.nBins);

    //termalization
    for (int i = 0; i < m_para.nTherm; i++)
        update_monte_carlo(state, nnList, pFlips, N);

    //simulation
    PRECISION scale_N  =  1.0 / (m_para.nSweeps * N);
    PRECISION scale_N2 =  1.0 / (m_para.nSweeps * N * N);
    PRECISION scale_N4 = (1.0 / (m_para.nSweeps * N * N)) * (1.0 / (N * N));

    size_t    state_size = sizeof(int8_t) * N;

    for (int bin = 0; bin < m_para.nBins; bin++)
    {
        PRECISION energy = 0;
        PRECISION mAbs = 0;
        PRECISION m2 = 0;
        PRECISION m4 = 0;

        //sweeps per bin (prebinning)
        for (int sweep = 0; sweep < m_para.nSweeps; sweep++)
        {
            update_monte_carlo(state, nnList, pFlips, N);

            //take measurements
            int mag_sweep = std::reduce(std::begin(state), std::end(state), 0);
            PRECISION energy_sweep = calcStateEnergy(state, nnList, m_J);

            //prebinning
            energy += energy_sweep;
            mAbs   += abs(mag_sweep);
            m2     += (mag_sweep * mag_sweep);
            m4     += (mag_sweep * mag_sweep * mag_sweep * mag_sweep);
        }

        //evaluate prebinning     
        energy *= scale_N;
        mAbs   *= scale_N;
        m2     *= scale_N2;
        m4     *= scale_N4;

        //store bin
        memcpy(m_states[bin].data(), state, state_size);

        m_energy[bin] = energy;
        m_m[bin]    = std::reduce(std::begin(state), std::end(state), 0);
        m_mAbs[bin] = mAbs;
        m_m2[bin]   = m2;
        m_m4[bin]   = m4;
    }
}
void Simulator::run_wolff_cluster()
{
    //resize to requested bins count
    m_states.resize(m_para.nBins);
    m_energy.resize(m_para.nBins);
    m_m.resize(m_para.nBins);
    m_mAbs.resize(m_para.nBins);
    m_m2.resize(m_para.nBins);
    m_m4.resize(m_para.nBins);

    //termalization
    for (int i = 0; i < m_para.nTherm; i++)
        update_wolff_cluster(state, nnList, N);

    //simulation
    PRECISION scale_N  =  1.0 / (m_para.nSweeps * N);
    PRECISION scale_N2 =  1.0 / (m_para.nSweeps * N * N);
    PRECISION scale_N4 = (1.0 / (m_para.nSweeps * N * N)) * (1.0 / (N * N));

    size_t    state_size = sizeof(int8_t) * N;

    for (int bin = 0; bin < m_para.nBins; bin++)
    {
        PRECISION energy = 0;
        PRECISION mAbs = 0;
        PRECISION m2 = 0;
        PRECISION m4 = 0;

        //sweeps per bin (prebinning)
        for (int sweep = 0; sweep < m_para.nSweeps; sweep++)
        {
            update_wolff_cluster(state, nnList, N);

            //take measurements
            int mag_sweep = std::reduce(std::begin(state), std::end(state), 0);
            PRECISION energy_sweep = calcStateEnergy(state, nnList, m_J);

            //prebinning
            energy += energy_sweep;
            mAbs   += abs(mag_sweep);
            m2     += (mag_sweep * mag_sweep);
            m4     += (mag_sweep * mag_sweep * mag_sweep * mag_sweep);
        }

        //evaluate prebinning     
        energy *= scale_N;
        mAbs   *= scale_N;
        m2     *= scale_N2;
        m4     *= scale_N4;

        //store bin
        memcpy(m_states[bin].data(), state, state_size);

        m_energy[bin] = energy;
        m_m[bin]    = std::reduce(std::begin(state), std::end(state), 0);
        m_mAbs[bin] = mAbs;
        m_m2[bin]   = m2;
        m_m4[bin]   = m4;
    }
}

void Simulator::store_data()
{
    std::vector<std::vector<PRECISION>> data;
    for (int i = 0; i < m_para.nBins; i++)
        data.push_back({ m_energy.at(i), m_m.at(i), m_mAbs.at(i), m_m2.at(i), m_m4.at(i) });

    std::string tj = std::format("TJ_{}", m_TJ);
    std::string sPara = parameter_string();

    storeFloatData(data    , "simulation_observ_" + tj + ".txt", "[Prebinned Observables: energy, m (not prebinned), mAbs, m2, m4] " + sPara);
    storeStateData(m_states, "simulation_states_" + tj + ".txt", "[Spin states] "             + sPara);
}

PRECISION Simulator::calcStateEnergy(const int8_t* state, const uint16_t* nnList, PRECISION J)
{
    PRECISION enery = 0;

    for (int i = 0; i < N; i++)
    {
        int8_t local = 0;
        local += state[nnList[i]];
        local += state[nnList[i + 1]];
        local += state[nnList[i + 2]];
        local += state[nnList[i + 3]];

        enery += local * state[i];
    }

    return -0.5 * J * enery;
}

void Simulator::precalc_monte_carlo(PRECISION* pFlips, PRECISION T, PRECISION J)
{
    PRECISION preFactor = -2.0 * (J / T);

    for (int8_t m = -4; m < 5; m += 2)
        pFlips[m + 4] = exp(preFactor * m);
}
void Simulator::update_monte_carlo(int8_t* state, const uint16_t* nnList, const PRECISION* pFlips, int n)
{
    for (int i = 0; i < n; i++)
    {
        int pos = rand() % n;
       
        //--------    

        int nn_index = pos * 4;
        uint8_t m = 0;

        m += state[nnList[nn_index]];
        m += state[nnList[nn_index + 1]];
        m += state[nnList[nn_index + 2]];
        m += state[nnList[nn_index + 3]];

        uint8_t index = 4 + m * state[pos];
        PRECISION pAccept = pFlips[index];

        //-----

        PRECISION rnd = PRECISION(rand()) / RAND_MAX;
        if (rnd < pAccept)
            state[pos] = -state[pos];
    }
}

void Simulator::update_wolff_cluster(int8_t* state, const uint16_t* nnList, int n)
{
    //starting position
    int init_pos      = rand() % n;
    int8_t init_spin  = state[init_pos];
    state[init_pos]   = -init_spin;

    //add init pos
    std::vector<uint16_t> cluster;
    cluster.push_back(init_pos);

    //loop over cluster
    while (cluster.size())
    {
        uint16_t pos = cluster.back();
        cluster.pop_back();

        //check the nn of pos
        for (uint16_t i = 0; i < 4; i++)
        {
            uint16_t n_pos = nnList[pos * 4 + i];

            //check spin alignment
            if (state[n_pos] != init_spin)
                continue;
                
            //m_pAccept probability for n_pos!!
            PRECISION rnd = PRECISION(rand()) / RAND_MAX;           
            if (rnd < m_pAccept)
            {
                cluster.push_back(n_pos);
                state[n_pos] = -init_spin;
            }
        }
    }
}
