#ifndef GRID4_H
#define GRID4_H

#include <vector>
#include "surface.h"
#include "function.h"

template<typename Tf>
class Grid4 {
  private:
  // Neighbours structure
  struct Neighbours {
    int point[33];

    Neighbours() {
      for (int i = 0; i < 33; ++i) {
        point[i] = -1;
      }
    }
  };

  // num of points
  int _nPoints;
  // center point
  int _centerPoint;

  // points in the volume
  Surface<Quaternion<Tf>> _volume;
  
  // 16 grid points around projection of each point:
  // first: array of index in _volume of the points
  // second: array of weights
  std::pair<int[16], Tf[16]>* _projection;
  
  // 33 grid points around each point and itself
  Neighbours* _neighbour;

  // density
  Function<Tf> _rho;

  Tf _t;
  Tf _h;

  // grid dimensions and origins (needed for index arithmetic in createProjection)
  int _wPoints;
  int _xPoints;
  int _yPoints;
  int _zPoints;
  Tf _wMin;
  Tf _xMin;
  Tf _yMin;
  Tf _zMin;

  // Used in std::for_each
  std::vector<int> _indices;
  std::vector<float> _diff;

  public:
  Grid4(Interval<Tf> w = {0,10},
    Interval<Tf> x = {0,10},
    Interval<Tf> y = {0,10},
    Interval<Tf> z = {0,10},
    Tf T = 40.,
    Tf h = 0.1);
  Grid4(std::string surfFile, std::string realFile, Tf T);
  ~Grid4();

  //set the first values of _rho
  void fillFirstRho();
  //set _rho provinding a _volume index point
  void fillIdxRho(int idx = 1412880);

  void saveProjection(std::string filename = "simulation");
  void saveNeighbour(std::string filename = "simulation");
  void saveSurface(std::string filename = "simulation");
  void saveRho(std::string filename = "simulation");

  void evolve(Tf dt = 0);
  void evolveCN(Tf dt = 0, int maxIter = 1000, Tf tol = 1e-3);
  void evolveCNDiff(Tf dt = 0, int maxIter = 1000, Tf tol = 1e-3);
  void project();

  private:
  //create volume and neighbour: hypercube case
  void createVolNeigh(Interval<Tf> w, Interval<Tf> x, Interval<Tf> y, Interval<Tf> z);
  
  //create volume: spherical crown case
  void createVol(Interval<Tf> w, Interval<Tf> x, Interval<Tf> y, Interval<Tf> z);
  
  //create projection: hypercube case
  void createProjection();
  
  //create neighbour and projection: spherical crown case
  void createMaps();
  
  //create the first rho value
  void createRho();
  
  //signed distance from the sphere
  Tf phi(Tf w, Tf x, Tf y, Tf z);

  //return the point on the surface which is closest to given point
  Quaternion<Tf> closest(Quaternion<Tf> const& p);

  //load the map between each surface point 'p' and the interpolation(closest(neighbour(p))).
  //Returns false if file is not found.
  bool loadSurface(std::string filename = "simulation");
  bool loadProjection(std::string filename = "simulation");
  bool loadNeighbour(std::string filename = "simulation");
  bool loadRho(std::string filename = "simulation");

  //first derivative along one direction
  Tf der1(int pointIndex, int direction) const;

  //second derivative along the same direction
  Tf der2(int pointIndex, int direction, Quaternion<Tf> const& pos = {-1., -1., -1., -1.}) const;

  //second mix derivative along two different directions
  Tf derij(int pointIndex, int dir1, int dir2, Quaternion<Tf> const& pos = {-1., -1., -1., -1.}) const;

  // overloads operating on an arbitrary Function (used by evolveCN)
  Tf der1(int pointIndex, int direction, Function<Tf> const& f) const;
  Tf der2(int pointIndex, int direction, Function<Tf> const& f, Quaternion<Tf> const& pos = {-1., -1., -1., -1.}) const;
  Tf derij(int pointIndex, int dir1, int dir2, Function<Tf> const& f, Quaternion<Tf> const& pos = {-1., -1., -1., -1.}) const;
};

#endif