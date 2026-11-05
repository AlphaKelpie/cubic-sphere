#ifndef GRID4_H
#define GRID4_H

#include "function.h"
#include "surface.h"
#include <vector>

template <typename Tf> class Grid4 {
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
  std::pair<int[16], Tf[16]> *_projection;

  // 33 grid points around each point and itself
  Neighbours *_neighbour;

  // density
  Function<Tf> _rho;

  double _t;
  double _h;

  // grid dimensions and origins (needed for index arithmetic in
  // createProjection)
  int _wPoints;
  int _xPoints;
  int _yPoints;
  int _zPoints;

  // Used in std::for_each
  std::vector<int> _indices;
  std::vector<float> _diff;

public:
  Grid4(Interval<double> w = {0, 10}, Interval<double> x = {0, 10},
        Interval<double> y = {0, 10}, Interval<double> z = {0, 10},
        double T = 40., double h = 0.1);
  Grid4(std::string surfFile, std::string realFile, Tf T);
  ~Grid4();

  // set the first values of _rho
  void fillFirstRho();
  // set _rho provinding a _volume index point
  void fillIdxRho(int idx = 1412880);

  void saveProjection(std::string filename = "simulation");
  void saveNeighbour(std::string filename = "simulation");
  void saveSurface(std::string filename = "simulation");
  void saveRho(std::string filename = "simulation");

  void evolve(double dt = 0);
  void evolveCN(double dt = 0, int maxIter = 1000, Tf tol = 1e-3);
  void evolveCNDiff(double dt = 0, int maxIter = 1000, Tf tol = 1e-3);
  void project();

private:
  // create volume and neighbour: hypercube case
  void createVolNeigh(Interval<double> w, Interval<double> x,
                      Interval<double> y, Interval<double> z);

  // create volume: spherical crown case
  void createVol(Interval<double> w, Interval<double> x, Interval<double> y,
                 Interval<double> z);

  // create projection: hypercube case
  void createProjection();

  // create neighbour and projection: spherical crown case
  void createMaps();

  // create the first rho value
  void createRho();

  // signed distance from the sphere
  double phi(double w, double x, double y, double z);

  // return the point on the surface which is closest to given point
  Quaternion<double> closest(Quaternion<double> const &p);

  // load the map between each surface point 'p' and the
  // interpolation(closest(neighbour(p))). Returns false if file is not found.
  bool loadSurface(std::string filename = "simulation");
  bool loadProjection(std::string filename = "simulation");
  bool loadNeighbour(std::string filename = "simulation");
  bool loadRho(std::string filename = "simulation");

  // first derivative along one direction
  double der1(int pointIndex, int direction) const;

  // second derivative along the same direction
  double der2(int pointIndex, int direction,
              Quaternion<double> const &pos = {-1., -1., -1., -1.}) const;

  // second mix derivative along two different directions
  double derij(int pointIndex, int dir1, int dir2,
               Quaternion<double> const &pos = {-1., -1., -1., -1.}) const;

  // overloads operating on an arbitrary Function (used by evolveCN)
  double der1(int pointIndex, int direction, Function<Tf> const &f) const;
  double der2(int pointIndex, int direction, Function<Tf> const &f,
              Quaternion<double> const &pos = {-1., -1., -1., -1.}) const;
  double derij(int pointIndex, int dir1, int dir2, Function<Tf> const &f,
               Quaternion<double> const &pos = {-1., -1., -1., -1.}) const;
};

#endif