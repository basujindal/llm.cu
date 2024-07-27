// c function for exp function
//     # - return the result
//     return np.exp(-x) / (1 + np.exp(-x)) ** 2


#include <stdio.h>
#include <math.h>


int main() {
    for(float x = 0.0; x > -100.0; x -= 1)
        printf("exp(%f) = %f\n", x, expf(x));

    return 0;
}
