#include "fht.h"
#include <random>
using namespace fht;

int main(int argc, char *argv[]) {
    int l2 = argc == 1 ? 16: std::atoi(argv[1]), subl2 = 4;
    std::vector<double> zomg(1 << l2);
    std::normal_distribution<float> gen;
    std::mt19937_64 mt;
    for(auto &e: zomg) e = gen(mt);
    std::vector<double> z2 = zomg;
    assert(z2 == zomg);
    show(z2);
#if 0
    for(int i = 1; i <= l2; ++i) {
        printf("bhefore:\n");
        show(z2);
        dumbfht(z2.data(), i);
        printf("bafter:\n");
        show(z2);
        z2 = zomg;
    }
#endif
    dumbfht(z2.data(), l2);
    show(z2);
    show(zomg);
    call_dumbfht(zomg.data(), l2, subl2);
    show(zomg);
    double sum = 0.;
    for(size_t i = 0; i < z2.size(); ++i)
        sum += std::abs(z2[i] - zomg[i]);
    assert(z2 == zomg || std::fprintf(stderr, "Failed: %lf", sum / z2.size()));
}
