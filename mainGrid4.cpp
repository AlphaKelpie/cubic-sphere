#include "grid4.h"

#include <iostream>
#include <string>

int main() {
  try
  {
    std::string const path = "./data/test4_";
    {
      std::cout << "Creating\n";
      Grid4 g({0.,1.}, {0.,1.}, {0.,1.}, {0.,1.}, 40., .1);

      std::cout << "Saving\n";
      g.saveRelations(path);
      g.saveRho(path + "init_");
      g.saveSurface(path);
    }

    {
      std::cout << "Loading\n";
      Grid4 g(path, path, 40.);

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
      g.saveRho(path + "evolved");
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
