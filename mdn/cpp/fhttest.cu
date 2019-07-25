#include "fht.h"
using namespace fht;

template<typename T>void show(const T&x) {
    size_t k = 0;
    for(const auto i: x) {
        std::fprintf(stderr, "%lf,", double(i)); std::fputc('\n', stderr);
        if(++k == 10) return;
    }
}

int main() {
    int l2 = 16, subl2 = 8;
    std::vector<double> zomg(1 << l2);
    for(auto &e: zomg) e = std::rand() / double(RAND_MAX);
    std::vector<double> z2 = zomg;
    assert(z2 == zomg);
    show(z2);
    show(zomg);
    dumbfht(z2.data(), l2);
    call_dumbfht(zomg.data(), l2, subl2);
    show(z2);
    show(zomg);
    double sum = 0.;
    for(size_t i = 0; i < z2.size(); ++i)
        sum += std::abs(z2[i] - zomg[i]);
    assert(z2 == zomg || std::fprintf(stderr, "Failed: %lf", sum / z2.size()));
}
