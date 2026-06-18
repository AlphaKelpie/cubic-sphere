#ifndef GRID3_H
#define GRID3_H

#include "surface.h"
#include "function.h"

class Grid3 {
  private:
  // Neighbours structure
  struct Neighbours {
      int point[19];
  };

  // num of points
  int _nPoints;

  // points in the volume
  Surface<Point> _volume;
  
  // 8 grid points around projection of each point:
  // first: array of index in _s of the points
  // second: array of weights
  std::pair<int[8], double[8]>* _projection;
  
  // 19 grid points around each point and itself
  Neighbours* _neighbour;

  // density
  Function _rho;

  double _t;
  double _h;

  // grid dimensions and origins (needed for index arithmetic in createProjection)
  int _xPoints;
  int _yPoints;
  int _zPoints;
  double _xMin;
  double _yMin;
  double _zMin;

  public:
  Grid3(Interval x = {0,10},
    Interval y = {0,10},
    Interval z = {0,10},
    double T = 40.,
    double h = 0.1);
  Grid3(std::string surfFile, std::string realFile, double T);
  ~Grid3();

  void saveProjection(std::string filename = "simulation");
  void saveNeighbour(std::string filename = "simulation");
  void saveSurface(std::string filename = "simulation");
  void saveRho(std::string filename = "simulation");

  void evolve(double dt = 0);
  void project();

  private:
  //signed distance from the sphere
  double phi(double x, double y, double z);

  //create the first rho value
  void createRho();

  //set the first values of _rho
  void fillFirstRho();

  //calculate points for projection
  void createProjection();

  //return the point on the surface which is closest to given point
  Point closest(Point const& p);

  //load the map between each surface point 'p' and the interpolation(closest(neighbour(p))).
  //Returns false if file is not found.
  bool loadSurface(std::string filename = "simulation");
  bool loadProjection(std::string filename = "simulation");
  bool loadNeighbour(std::string filename = "simulation");
  bool loadRho(std::string filename = "simulation");

  //first derivative along one direction
  double der1(int pointIndex, int direction) const;

  //second derivative along the same direction
  double der2(int pointIndex, int direction) const;

  //second mix derivative along two different directions
  double derij(int pointIndex, int dir1, int dir2) const;
};

#endif