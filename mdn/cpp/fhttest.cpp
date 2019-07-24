#include "fht.h"
int main() {
    std::vector<double> zomg(1 << 16);
    for(auto &e: zomg) e = std::rand() / double(RAND_MAX);
    std::vector<double> z2 = zomg;
    dumbfht(z2.data(), 16);
    assert(z2 == zomg);
}
