#include "params.hpp"
#include "grid4.h"

#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>

void evolving(Grid4& g, std::string const& path) {
  const double deltaT = Params::get().deltaT;
  std::cout << "\n\033[1;34mEvolving with deltaT = " << deltaT << "\033[0m\n";
  for (int step = Params::get().step; step < Params::get().total;
      step+=Params::get().step) {
    std::cout << "Step " << step << ": " << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    for (int sim = 0; sim < Params::get().step; ++sim) {
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
  if (argc == 2) {
    std::cout << "\033[1;36mParams from " << argv[1] << "\033[0m\n";
    Params::load(argv[1]);
  } else {
    std::cout << "\033[1;36mParams from params.dat\033[0m\n";
    Params::load("params.dat");
  }

  try
  {
    std::string const path = Params::get().path;
    if (!std::filesystem::exists(std::filesystem::path{Params::get().pathdata + "_s.dat"})) {
      std::cout << "\033[35mCreating\033[0m\n";
      Grid4 g(
        Params::get().w,
        Params::get().x,
        Params::get().y,
        Params::get().z,
        Params::get().T,
        Params::get().h
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
      std::string const pathdata = Params::get().pathdata;
      std::cout << "\033[35mLoading " << pathdata << "\033[0m\n";
      Grid4 g(pathdata, pathdata, Params::get().T);
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
