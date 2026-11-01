#include <tbb/global_control.h>
#include <iostream>

int main() {
    auto n = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    std::cout << std::to_string(n) << std::endl;
}
