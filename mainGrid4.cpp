#include "params.hpp"
#include "grid4.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  if (argc == 2) {
    Params::load(argv[1]);
  } else {
    Params::load("params.dat");
  }

  try
  {
    std::string const path = Params::get().path;
    {
      std::cout << "Creating\n";
      Grid4 g(
        Params::get().w,
        Params::get().x,
        Params::get().y,
        Params::get().z,
        Params::get().T,
        Params::get().h
      );

      std::cout << "Saving\n";
      g.saveRelations(path);
      g.saveRho(path + "init_");
      g.saveSurface(path);
    }

    {
      std::cout << "Loading\n";
      Grid4 g(path, path, Params::get().T);

      std::cout << "Evolving\n";
      for (int step = 100; step < 2002; step+=100) {
        std::cout << "Step " << step << ": " << std::flush;
        for (int sim = 0; sim < 100; ++sim) {
          g.evolve();
          break;
        }
        g.saveRho(path + "evolving_" + std::to_string(step));
        break;
      }

      g.project();
      g.saveRho(path + "evolved_");
    }

    std::cout << "End\n";
    return 0;
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
