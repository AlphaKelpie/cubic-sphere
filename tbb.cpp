/*
    Check number of thread available and how limit works for tbb library
*/

#include <tbb/global_control.h>

#include <iostream>

int main() {
  auto n = tbb::global_control::active_value(
      tbb::global_control::max_allowed_parallelism);
  std::cout << "Total numer of thread for TBB: " << std::to_string(n)
            << std::endl;
  tbb::global_control limit(tbb::global_control::max_allowed_parallelism, 4);
  n = tbb::global_control::active_value(
      tbb::global_control::max_allowed_parallelism);
  std::cout << "After applying limit: " << std::to_string(n) << std::endl;
}
