#include "fht.h"
#include <random>
#include "omp.h"
using namespace fht;

int main(int argc, char *argv[]) {
    int l2 = argc == 1 ? 16: std::atoi(argv[1]), subl2 = 8;
    int n = 256;
    std::vector<double> zomg(n << l2);
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
    size_t s = 0;
    omp_set_num_threads(48);
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel reduction(:+s)
    for(int i = 0; i < n; ++i) {
        dumbfht(z2.data() + (i << l2), l2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "cpu took %zu\n", (end - start).count());
    show(z2);
    show(zomg);
    call_dumbfht(zomg.data(), l2, subl2, n);
    show(zomg);
    double sum = 0.;
    for(size_t i = 0; i < z2.size(); ++i)
        sum += std::abs(z2[i] - zomg[i]);
    double md = 0.;
    if(z2 != zomg) {
        for(size_t i = 0; i < z2.size(); ++i)
            if(md < std::abs(z2[i] - zomg[i]))
                md = std::abs(z2[i] - zomg[i]), std::fprintf(stderr, "Max difference: %f/%f:%f\n", z2[i], zomg[i], abs(z2[i] - zomg[i]));
    }
    assert(z2 == zomg || (sum / z2.size() < 10.) || !std::fprintf(stderr, "Failed: %lf", sum / z2.size()));
}
