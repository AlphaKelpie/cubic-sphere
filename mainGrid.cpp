#include "grid3.h"

#include <iostream>

int main() {
  try
  {
    std::cout << "Creating\n";
    {
      Grid3 g({0.,1.}, {0.,1.}, {0.,1.}, 40, .1);

      std::cout << "Saving\n";
      g.saveRelations("test");
      g.saveRho("init");
      g.saveSurface("test");
    }

    std::cout << "Loading\n";
    Grid3 g("test", "test", 40.);

    std::cout << "Evolving\n";
    for (int step = 1; step < 2; step+=100) {
      std::cout << "Step " << step << ": " << std::flush;
      for (int sim = 0; sim < 1; ++sim) {
        std::cout << sim << ' ' << std::flush;
        g.evolve();
      }
      g.saveRho("evolving");
    }

    g.project();
    g.saveRho("evolved");

    std::cout << "End\n";
    return 0;
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
