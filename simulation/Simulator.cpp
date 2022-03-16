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
    m_mAbs3.resize(m_para.nBins);
    m_m4.resize(m_para.nBins);

    //termalization
    for (int i = 0; i < m_para.nTherm; i++)
        update_monte_carlo(state, nnList, pFlips, N);

    //simulation
    #if PREBINNING_OBSERVABLES
        PRECISION scale_N = 1.0 / (m_para.nSweeps * N);
        PRECISION scale_N2 = 1.0 / (m_para.nSweeps * N * N);
        PRECISION scale_N3 = (1.0 / (m_para.nSweeps * N * N)) * (1.0 / (N));
        PRECISION scale_N4 = (1.0 / (m_para.nSweeps * N * N)) * (1.0 / (N * N));
    #else
        PRECISION scale_N = 1.0 / (N);
        PRECISION scale_N2 = 1.0 / (N * N);
        PRECISION scale_N3 = (1.0 / (N * N)) * (1.0 / (N));
        PRECISION scale_N4 = (1.0 / (N * N)) * (1.0 / (N * N));
    #endif // PREBINNING_OBSERVABLES

    size_t state_size = sizeof(int8_t) * N;

    for (int bin = 0; bin < m_para.nBins; bin++)
    {
        #if PREBINNING_OBSERVABLES
                PRECISION energy = 0;
                PRECISION mAbs = 0;
                PRECISION m2 = 0;
                PRECISION mAbs3 = 0;
                PRECISION m4 = 0;
        #endif // PREBINNING_OBSERVABLES

        //sweeps per bin (prebinning)
        for (int sweep = 0; sweep < m_para.nSweeps; sweep++)
        {
            update_monte_carlo(state, nnList, pFlips, N);

            #if PREBINNING_OBSERVABLES
                //take measurements
                PRECISION mag_sweep = std::reduce(std::begin(state), std::end(state), 0);
                PRECISION energy_sweep = calcStateEnergy(state, nnList, m_J);

                //prebinning
                energy += energy_sweep;
                mAbs += abs(mag_sweep);
                m2 += (mag_sweep * mag_sweep);
                mAbs3 += (mAbs * mAbs * mAbs);
                m4 += (mag_sweep * mag_sweep * mag_sweep * mag_sweep);
            #endif // PREBINNING_OBSERVABLES
        }

        #if PREBINNING_OBSERVABLES
            //evaluate prebinning     
            energy *= scale_N;
            mAbs *= scale_N;
            m2 *= scale_N2;
            mAbs3 *= scale_N3;
            m4 *= scale_N4;

            m_energy[bin] = energy;
            m_m[bin] = std::reduce(std::begin(state), std::end(state), 0) / PRECISION(N);
            m_mAbs[bin] = mAbs;
            m_m2[bin] = m2;
            m_mAbs3[bin] = mAbs3;
            m_m4[bin] = m4;
        #else
            PRECISION m = std::reduce(std::begin(state), std::end(state), 0);

            m_energy[bin] = scale_N * calcStateEnergy(state, nnList, m_J);
            m_m[bin]      = scale_N * m;
            m_mAbs[bin]   = scale_N * abs(m);
            m_m2[bin]     = scale_N2 * m * m;
            m_mAbs3[bin]  = scale_N3 * abs(m * m * m);
            m_m4[bin]     = scale_N4 * m * m * m * m;
        #endif // PREBINNING_OBSERVABLES

        //store bin
        memcpy(m_states[bin].data(), state, state_size);
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
    m_mAbs3.resize(m_para.nBins);
    m_m4.resize(m_para.nBins);

    //modify updates respecting m_TJ in [1, 3.4], this is bcs the clusters will get smaller with m_TJ
    int nTherm  = m_para.nTherm  * m_TJ;
    int nSweeps = m_para.nSweeps * m_TJ * m_TJ;

    //termalization
    for (int i = 0; i < nTherm; i++)
        update_wolff_cluster(state, nnList, N);

    //simulation
    #if PREBINNING_OBSERVABLES
        PRECISION scale_N = 1.0 / (m_para.nSweeps * N);
        PRECISION scale_N2 = 1.0 / (m_para.nSweeps * N * N);
        PRECISION scale_N3 = (1.0 / (m_para.nSweeps * N * N)) * (1.0 / (N));
        PRECISION scale_N4 = (1.0 / (m_para.nSweeps * N * N)) * (1.0 / (N * N));
    #else
        PRECISION scale_N = 1.0 / (N);
        PRECISION scale_N2 = 1.0 / (N * N);
        PRECISION scale_N3 = (1.0 / (N * N)) * (1.0 / (N));
        PRECISION scale_N4 = (1.0 / (N * N)) * (1.0 / (N * N));
    #endif // PREBINNING_OBSERVABLES

    size_t state_size = sizeof(int8_t) * N;

    for (int bin = 0; bin < m_para.nBins; bin++)
    {
        if (bin % (2<<9) == 0)
            std::cout << int((float(bin) / m_para.nBins) * 100.f) << " %" << std::endl;

        #if PREBINNING_OBSERVABLES
            PRECISION energy = 0;
            PRECISION mAbs = 0;
            PRECISION m2 = 0;
            PRECISION mAbs3 = 0;
            PRECISION m4 = 0;
        #endif // PREBINNING_OBSERVABLES

        //sweeps per bin (prebinning)
        for (int sweep = 0; sweep < nSweeps; sweep++)
        {
            update_wolff_cluster(state, nnList, N);

            #if PREBINNING_OBSERVABLES
                //take measurements
                PRECISION mag_sweep = std::reduce(std::begin(state), std::end(state), 0);
                PRECISION energy_sweep = calcStateEnergy(state, nnList, m_J);

                //prebinning
                energy += energy_sweep;
                mAbs   += abs(mag_sweep);
                m2     += (mag_sweep * mag_sweep);
                mAbs3  += (mAbs * mAbs * mAbs);
                m4     += (mag_sweep * mag_sweep * mag_sweep * mag_sweep);
            #endif // PREBINNING_OBSERVABLES
        }

        #if PREBINNING_OBSERVABLES
            //evaluate prebinning     
            energy *= scale_N;
            mAbs *= scale_N;
            m2 *= scale_N2;
            mAbs3 *= scale_N3;
            m4 *= scale_N4;

            m_energy[bin] = energy;
            m_m[bin] = std::reduce(std::begin(state), std::end(state), 0) / PRECISION(N);
            m_mAbs[bin] = mAbs;
            m_m2[bin] = m2;
            m_mAbs3[bin] = mAbs3;
            m_m4[bin] = m4;
        #else
            PRECISION m = std::reduce(std::begin(state), std::end(state), 0);

            m_energy[bin] = scale_N  * calcStateEnergy(state, nnList, m_J);
            m_m[bin]      = scale_N  * m;
            m_mAbs[bin]   = scale_N  * abs(m);
            m_m2[bin]     = scale_N2 * m * m;
            m_mAbs3[bin]  = scale_N3 * abs(m * m * m);
            m_m4[bin]     = scale_N4 * m * m * m * m;
        #endif // PREBINNING_OBSERVABLES

        //store bin
        memcpy(m_states[bin].data(), state, state_size);
    }
}

void Simulator::store_data()
{
    std::vector<std::vector<PRECISION>> data;
    for (int i = 0; i < m_para.nBins; i++)
        data.push_back({ m_energy.at(i), m_m.at(i), m_mAbs.at(i), m_m2.at(i), m_mAbs3.at(i), m_m4.at(i) });

    std::string tj = std::format("TJ_{}", m_TJ); //:.1f
    std::string sPara = parameter_string();
   
    #if PREBINNING_OBSERVABLES
        std::string header = "[Prebinned Observables: energy, m (not prebinned), mAbs, m2, mAbs3, m4] ";
    #else
        std::string header = "[(not prebinned) Observables: energy, m , mAbs, m2, mAbs3, m4] ";
    #endif // PREBINNING_OBSERVABLES

    storeFloatData(data    , "simulation_observ_" + tj + ".txt", header + sPara);
    storeStateData(m_states, "simulation_states_" + tj + ".txt", "[Spin states] " + sPara);
}

PRECISION Simulator::calcStateEnergy(const int8_t* state, const uint16_t* nnList, PRECISION J)
{
    PRECISION energy = 0;

    for (int i = 0; i < N; i++)
    {
        int8_t local = 0;
        local += state[nnList[i * 4]];
        local += state[nnList[i * 4 + 1]];
        local += state[nnList[i * 4 + 2]];
        local += state[nnList[i * 4 + 3]];

        energy += local * state[i];
    }

    return -0.5 * J * energy;
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
