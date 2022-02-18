#include <numeric>
#include "Simulator.h"
#include "helper.h"

//------------------

namespace
{
    //compile time for max performance
    static   int8_t  state[N];        //actual system state, spins +-1
    static uint16_t  nnList[N * 4];   //nn list
    static PRECISION pFlips[9];       //precalculated monte carlo values
}

//------------------

void Simulator::init_monte_carlo(PRECISION T, PRECISION J)
{
    //calc expensive values before
    precalc_monte_carlo(pFlips, T, J);

    //set random spins +-1
    randomize_state(state, N);

    //gen the next neighbor list
    generate_2D_NNList(LX, LY, nnList);
}
void Simulator::run_monte_carlo(PRECISION J)
{
    m_states.resize(m_para.nBins);
    m_m2.resize(m_para.nBins);
    m_energy.resize(m_para.nBins);

    //termalization
    for (int i = 0; i < m_para.nTherm; i++)
        update_monte_carlo(state, nnList, pFlips, N);

    //simulation
    PRECISION    scale_energy = 1.0 / (m_para.nSweeps * N);
    PRECISION    scale_m2     = 1.0 / (m_para.nSweeps * N * N);
    size_t state_size = sizeof(int8_t) * N;

    for (int bin = 0; bin < m_para.nBins; bin++)
    {
        PRECISION energy = 0;
        PRECISION m2     = 0;

        //sweeps per bin (prebinning)
        for (int sweep = 0; sweep < m_para.nSweeps; sweep++)
        {
            update_monte_carlo(state, nnList, pFlips, N);

            //take measurements
            int mag_sweep = std::reduce(std::begin(state), std::end(state), 0);
            PRECISION energy_sweep = calcStateEnergy(state, nnList, J);
            
            //prebinning
            energy += energy_sweep;
            m2     += mag_sweep * mag_sweep;
        }

        //evaluate prebinning     
        energy *= scale_energy;
        m2     *= scale_m2;

        //store bin
        memcpy(m_states[bin].data(), state, state_size);
        m_energy[bin] = energy;
        m_m2[bin]     = m2;        
    }
}
void Simulator::store_monte_carlo()
{
    //storeFloatData(data, "test.txt", "header");
}

void Simulator::precalc_monte_carlo(PRECISION* pFlips, PRECISION T, PRECISION J)
{
    PRECISION preFactor = -2.0 * (J / T);

    for (int8_t m = -4; m < 5; m += 2)
        pFlips[m + 4] = exp(preFactor * m);
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
void Simulator::update_monte_carlo(int8_t* state, const uint16_t* nnList, const PRECISION* pFlips, int n)
{
    for (int i = 0; i < n; i++)
    {
        int pos = int(PRECISION(rand()) / RAND_MAX * n);

        //--------

        uint8_t m = 0;
        m += state[nnList[pos]];
        m += state[nnList[pos + 1]];
        m += state[nnList[pos + 2]];
        m += state[nnList[pos + 3]];

        uint8_t index = 4 + m * state[pos];
        PRECISION pAccept = pFlips[index];

        //-----

        PRECISION rnd = PRECISION(rand()) / RAND_MAX;
        if (rnd < pAccept)
            state[pos] = -state[pos];
    }
}
