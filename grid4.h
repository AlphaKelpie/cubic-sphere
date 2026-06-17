#ifndef GRID4_H
#define GRID4_H

#include "volume.h"
#include "function.h"

class Grid4 {
  private:
  // Neighbours structure
  struct Neighbours {
      int point[33];
  };

  // num of points
  int _nPoints;

  // points in the volume
  Volume _volume;
  
  // 16 grid points around projection of each point:
  // first: array of index in _volume of the points
  // second: array of weights
  std::pair<int[16], float[16]>* _projection;
  
  // 33 grid points around each point and itself
  Neighbours* _neighbour;

  // density
  Function _rho;

  float _t;
  float _h;

  // grid dimensions and origins (needed for index arithmetic in createProjection)
  int _wPoints;
  int _xPoints;
  int _yPoints;
  int _zPoints;
  float _wMin;
  float _xMin;
  float _yMin;
  float _zMin;

  public:
  // ATTENZIONE
  Grid4(Interval w = {0,10},
    Interval x = {0,10},
    Interval y = {0,10},
    Interval z = {0,10},
    float T = 40.,
    float h = 0.1);
  Grid4(std::string surfFile, std::string realFile, float T);
  ~Grid4();

  void saveRelations(std::string filename = "simulation");
  void saveSurface(std::string filename = "simulation");
  void saveRho(std::string filename = "simulation");

  void evolve(float dt = 0);
  void project();

  private:
  //signed distance from the sphere
  float phi(float w, float x, float y, float z);

  //create the first rho value
  void createRho();

  //set the first values of _rho
  void fillFirstRho();

  //calculate points for projection
  void createProjection();

  //return the point on the surface which is closest to given point
  Quaternion closest(Quaternion const& p);

  //load the map between each surface point 'p' and the interpolation(closest(neighbour(p))).
  //Returns false if file is not found.
  bool loadSurface(std::string filename = "simulation");
  bool loadRelations(std::string filename = "simulation");
  bool loadRho(std::string filename = "simulation");

  //first derivative along one direction
  float der1(int pointIndex, int direction) const;

  //second derivative along the same direction
  float der2(int pointIndex, int direction) const;

  //second mix derivative along two different directions
  float derij(int pointIndex, int dir1, int dir2) const;

  //logic end of int between -1 and +1
  int sgn(int val1, int val2);
};

#endif