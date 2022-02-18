#pragma once
#include <stdexcept>

namespace
{
    inline int xy_to_index(int x, int y, int yL) { return y * yL + x; }

    inline int positive_modulo(int i, int n) { return (i % n + n) % n; }
}

void generate_2D_NNList(int xL, int yL, uint16_t* nnList)
{ 
    if (xy_to_index(xL, yL, yL) > UINT16_MAX)
        throw std::runtime_error("xL * yL > UINT8_MAX");

    for (int y = 0; y < yL; y++)
    {
        for (int x = 0; x < xL; x++)
        {
            size_t index = xy_to_index(x, y, yL) * 4;

            nnList[index + 0] = xy_to_index(positive_modulo(x + 1, xL),                          y, yL);  //right
            nnList[index + 1] = xy_to_index(positive_modulo(x - 1, xL),                          y, yL);  //left
            nnList[index + 2] = xy_to_index(x                         , positive_modulo(y + 1, yL), yL);  //bottom
            nnList[index + 3] = xy_to_index(x                         , positive_modulo(y - 1, yL), yL);  //top
        }
    }  
}

void randomize_state(int8_t* state, int n)
{
    for (int i = 0; i < n; i++)
    {
        float rnd = float(rand()) / RAND_MAX;

        uint8_t spin = 1;
        if (rnd < 0.5f)
            spin = -1;

        state[i] = spin;
    }
}