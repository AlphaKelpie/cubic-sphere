#include "params.hpp"
#include "grid4.h"

#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>

void evolving(Grid4& g, std::string const& path) {
  std::cout << "\nEvolving\n";
  for (int step = Params::get().step; step < Params::get().total;
      step+=Params::get().step) {
    std::cout << "Step " << step << ": " << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int sim = 0; sim < Params::get().step; ++sim) {
      g.evolveCN();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> dur = t1 - t0;
    std::cout << dur.count() << "ms\t" << std::flush;
    g.saveRho(path + "_evolving_" + std::format("{:04}", step));
  }

  g.project();
  g.saveRho(path + "_evolved");

  std::cout << "End\n";
}

int main(int argc, char* argv[]) {
  if (argc == 2) {
    Params::load(argv[1]);
  } else {
    Params::load("params.dat");
  }

  try
  {
    std::string const path = Params::get().path;
    if (!std::filesystem::exists(std::filesystem::path{Params::get().pathdata + "_s.dat"})) {
      std::cout << "Creating\n";
      Grid4 g(
        Params::get().w,
        Params::get().x,
        Params::get().y,
        Params::get().z,
        Params::get().T,
        Params::get().h
      );
      g.fillFirstRho();

      std::cout << "Saving\n";
      g.saveProjection(path);
      g.saveNeighbour(path);
      g.saveRho(path + "_init");
      g.saveSurface(path);

      evolving(g, path);
    }
    
    else {
      std::string const pathdata = Params::get().pathdata;
      std::cout << "Loading " << pathdata << '\n';
      Grid4 g(pathdata, pathdata, Params::get().T);
      g.saveRho(path + "_init");

      evolving(g, path);
    }

    return 0;
  }

  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
