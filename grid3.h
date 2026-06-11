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
    Surface _volume;
    // points on the surface
    // Surface _sphere;
    // density
    Function _rho;

    // 8 grid points around projection of each point:
    // first: array of index in _s of the points
    // second: array of weights
    std::pair<int[8], double[8]>* _projection;

    // 19 grid points around each point and itself
    Neighbours* _neighbour;

    double _h;
    double _delta;

    public:
    Grid3(Interval x = {0,10}, Interval y = {0,10}, Interval z = {0,10});
    Grid3(std::string mapFile);
    ~Grid3();

    void saveRelations(std::string filename = "relation.dat");
    void saveSurface(std::string filename = "surface.dat");
    void saveRho(std::string filename = "rho.dat");

    void evolve(double dt = 0);
    void project();

    private:
    double phi(double x, double y, double z);

    //create the first rho value
    void createRho();

    //set the first values of _rho
    void fillFirstRho();

    void createProjection();

    //return the point on the surface which is closest to given point
    Point closest(Point const& p);

    //load the map between each surface point 'p' and the interpolation(closest(neighbour(p))).
    //Returns false if file is not found.
    bool loadMap(std::string filename = "map.dat");

    double der1(int pointIndex, char direction);

    double der2(int pointIndex, char direction);

    double derij(int pointIndex, std::string directions);

    int Grid3::sgn(int val1, int val2);
};

#endif