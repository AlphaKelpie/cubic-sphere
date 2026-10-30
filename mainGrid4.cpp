#include "params.hpp"
#include "grid4.h"

#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <tbb/global_control.h>

typedef float Tf;

void evolving(Grid4<Tf>& g, std::string const& path) {
  const double deltaT = Params<Tf>::get().deltaT;
  std::cout << "\n\033[1;34mEvolving with deltaT = " << deltaT << "\033[0m\n";
  for (int step = Params<Tf>::get().step; step < Params<Tf>::get().total;
      step+=Params<Tf>::get().step) {
    std::cout << "Step " << step << ": " << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    for (int sim = 0; sim < Params<Tf>::get().step; ++sim) {
      g.evolveCN(deltaT);
    }
    auto t1 = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << dur << '\t' << std::flush;
    g.saveRho(path + "_evolving_" + std::format("{:04}", step));
  }

  g.project();
  g.saveRho(path + "_evolved");

  std::cout << "\033[1;32mEnd\033[0m\n";
}

int main(int argc, char* argv[]) {
  auto n = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
  int const max_thread = (n > 64) ? n - 10 : n - 2;
  tbb::global_control limit(tbb::global_control::max_allowed_parallelism, max_thread);

  if (argc == 2) {
    std::cout << "\033[1;36mParams from " << argv[1] << "\033[0m\n";
    Params<Tf>::load(argv[1]);
  } else {
    std::cout << "\033[1;36mParams from params.dat\033[0m\n";
    Params<Tf>::load("params.dat");
  }

  try
  {
    std::string const path = Params<Tf>::get().path;
    if (!std::filesystem::exists(std::filesystem::path{Params<Tf>::get().pathdata + "_s.dat"})) {
      std::cout << "\033[35mCreating\033[0m\n";
      Grid4<Tf> g(
        Params<Tf>::get().w,
        Params<Tf>::get().x,
        Params<Tf>::get().y,
        Params<Tf>::get().z,
        Params<Tf>::get().T,
        Params<Tf>::get().h
      );
      g.fillFirstRho();

      std::cout << "\033[35mSaving\033[0m\n";
      g.saveProjection(path);
      g.saveNeighbour(path);
      g.saveRho(path + "_init");
      g.saveSurface(path);

      evolving(g, path);
    }
    
    else {
      std::string const pathdata = Params<Tf>::get().pathdata;
      std::cout << "\033[35mLoading " << pathdata << "\033[0m\n";
      Grid4<Tf> g(pathdata, pathdata, Params<Tf>::get().T);
      g.saveRho(path + "_init");

      evolving(g, path);
    }

    return 0;
  }

  catch(const std::exception& e)
  {
    std::cerr << "\033[31m" << e.what() << "\033[0m\n";
    return 1;
  }
}
