#include "grid3.h"
#include "params.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

Grid3::Grid3(Interval x, Interval y, Interval z, double T, double h)
	: _nPoints{0},
		_projection{nullptr},
		_neighbour{nullptr},
    _t{T/4},
		_h{h}
{
	int xPoints = (x.max - x.min) / _h + 1;
	int yPoints = (y.max - y.min) / _h + 1;
	int zPoints = (z.max - z.min) / _h + 1;
	_nPoints = xPoints * yPoints * zPoints;

	// Create surface and get _nPoints
	Point* temp = new Point[_nPoints];

	_neighbour = new Neighbours[_nPoints];

  std::cout << "Construct volume and neighbour...\n";
	//for every point of the descrete domain..
	bool down;
	bool top;
	bool behind;
	bool front;
	bool right;
	bool left;
  int index = 0;
  double i;
  double j;
  double k;
	for (short ii = 0; ii < _xPoints; ++ii) {
    i = x.min + _h * ii;
    left = (ii <= 0) ? false : true;
    right = (ii >= _xPoints - 1) ? false : true;
		for (short ji = 0; ji < _yPoints; ++ji) {
      j = y.min + _h * ji;
      front = (ji <= 0) ? false : true;
      behind = (ji >= _yPoints - 1) ? false : true;
			for (short ki = 0; ki < _zPoints; ++ki) {
        k = z.min + _h * ki;
        down = (ki <= 0) ? false : true;
        top = (ki >= _zPoints - 1) ? false : true;
        std::cout << '\r' << index << '/' << _nPoints << std::flush;
				//..add it to the volume and..
				temp[index] = {i, j, k};
				//..create corresponding neighbour
				_neighbour[index].point[0] = index;   // itself
				_neighbour[index].point[1] = (left) ? (index - yPoints * zPoints) : -1;
				_neighbour[index].point[2] = (right) ? (index + yPoints * zPoints) : -1;
				_neighbour[index].point[3] = (front) ? (index - zPoints) : -1;
				_neighbour[index].point[4] = (behind) ? (index + zPoints) : -1;
				_neighbour[index].point[5] = (down) ? (index - 1) : -1;
				_neighbour[index].point[6] = (top) ? (index + 1) : -1;

				_neighbour[index].point[7] = (left && front) ? (index - (yPoints + 1) * zPoints) : -1;
				_neighbour[index].point[8] = (left && behind) ? (index - (yPoints - 1) * zPoints) : -1;
				_neighbour[index].point[9] = (left && down) ? (index - (yPoints * zPoints + 1)) : -1;
				_neighbour[index].point[10] = (left && top) ? (index - (yPoints * zPoints - 1)) : -1;
				_neighbour[index].point[11] = (front && down) ? (index - (zPoints + 1)) : -1;
				_neighbour[index].point[12] = (front && top) ? (index - (zPoints - 1)) : -1;
				_neighbour[index].point[13] = (behind && down) ? (index + (zPoints - 1)) : -1;
				_neighbour[index].point[14] = (behind && top) ? (index + (zPoints + 1)) : -1;
				_neighbour[index].point[15] = (right && front) ? (index + (yPoints - 1) * zPoints) : -1;
				_neighbour[index].point[16] = (right && behind) ? (index + (yPoints + 1) * zPoints) : -1;
				_neighbour[index].point[17] = (right && down) ? (index + (yPoints * zPoints - 1)) : -1;
				_neighbour[index].point[18] = (right && top) ? (index + (yPoints * zPoints + 1)) : -1;
				++index;
			}
		}
	}
  if (index != _nPoints) {
    throw std::runtime_error("Error: _neighbour index different than _nPoints");
  }

	_volume = Surface(_nPoints, temp);
	delete[] temp;
  std::cout << "\tDone!\n";

	createProjection();
	createRho();
}

Grid3::Grid3(std::string surfFile, std::string relFile, double T)
	: _nPoints{0},
    _t{T/4},
    _xPoints{0}, _yPoints{0}, _zPoints{0},
    _xMin{0.}, _yMin{0.}, _zMin{0.}
{
  if (!loadSurface(surfFile)) {
		throw std::runtime_error{"loadSurface() cannot open file."};
  }
	if (!loadProjection(relFile)) {
		throw std::runtime_error{"loadProjection() cannot open file."};
	}
  if (!loadNeighbour(relFile)) {
		throw std::runtime_error{"loadNeighbour() cannot open file."};
	}
  if (!loadRho(relFile+"init_")) {
    std::cerr << "Warning: unable to load initial rho. Recreating";
  	createRho();
  }
}

Grid3::~Grid3() {
	delete[] _projection;
	delete[] _neighbour;
}

void Grid3::saveProjection(std::string filename) {
	// format:
  // first row: _nPoints
	// 1 row per point to _projection: index and weight for the 16 grid
	// points around

	std::cout << "Saving projection... " << std::flush;
	std::fstream fout(filename + "p.dat", std::ios::out);

	fout << _nPoints << "\n\n";

	for (int i = 0; i < _nPoints; ++i) {
		for (int j = 0; j < 8; ++j) {
			fout << _projection[i].first[j] << ' ';
			fout << _projection[i].second[j] << ' ';
		}
		fout << "\n";
	}

	fout.close();
	std::cout << "Done!\n";
}

void Grid3::saveNeighbour(std::string filename) {
  // format:
  // first row: _nPoints
	// 1 row per point to _neighbour indexes (33 int)

	std::cout << "Saving neighbour... " << std::flush;
	std::fstream fout(filename + "n.dat", std::ios::out);

	fout << _nPoints << "\n\n";

	for (int i = 0; i < _nPoints; ++i) {
		for (int j = 0; j < 19; ++j) {
			fout << _neighbour[i].point[j] << ' ';
		}
		fout << "\n";
	}

	fout.close();
	std::cout << "Done!\n";
}

void Grid3::saveSurface(std::string filename) {
  // format:
  // first row: _nPoints _h
  // 1 row per point in _volume

  std::cout << "Saving surface... " << std::flush;
  if (_volume.nPoints() == 0) {
    std::cerr << "Error: cannot save surface (probably empty)\n";
    return;
  }

  std::fstream fout(filename + "s.dat", std::ios::out);

  fout << _nPoints << ' ' << _h << "\n\n";

  fout << _volume;

  fout.close();
  std::cout << "Done!\n";
}

void Grid3::saveRho(std::string filename) {
  // format:
  // first row: _nPoints
  // 1 row per _rho value

  std::cout << "Saving rho... " << std::flush;

  std::fstream fout(filename + "r.dat", std::ios::out);

  fout << _nPoints << "\n\n";

  for (int i = 0; i < _nPoints; ++i) {
    fout << _rho[i] << '\n';
  }

  fout.close();
  std::cout << "Done!\n";
}

void Grid3::evolve(double dt) {
  if (dt == 0) {
    dt = 0.1 * _h * _h * _h;
  }

  Function nextRho = Function(_nPoints);

  double deltaRho = 0.;
  for (int idx = 0; idx < _nPoints; ++idx) {
    Point p = _volume[idx];

    for (int j = 0; j < 3; ++j) {
      deltaRho = p[j] * (1 - 3) * der1(idx, j);
      for (int i = 0; i < 3; ++i) {
        if (i == j) {
          deltaRho += p.D(i) * der2(idx, i);
        } else {
          deltaRho += p.D(i, j) * derij(idx, i, j);
        }
      }
    }

    deltaRho *= _t * dt;
    nextRho[idx] = _rho[idx] + deltaRho;
  }

  _rho = nextRho;
}

void Grid3::project() {
  Function sphere_rho(_nPoints);

  for (int i = 0; i < _nPoints; ++i) {
    double rho = _rho[i];
    if (rho == 0.) { continue; }
    std::pair<int[8], double[8]>& cubic = _projection[i];
    for (int j = 0; j < 8; ++j) {
      sphere_rho[cubic.first[j]] += rho * cubic.second[j];
    }
  }

  _rho = sphere_rho;
}


// private


double Grid3::phi(double x, double y, double z) {
  Quaternion& c = Params::get().center;
  return std::sqrt((x-c.x)*(x-c.x) + (y-c.y)*(y-c.y) + (z-c.z)*(z-c.z)
    ) - Params::get().radius;
}

void Grid3::createRho() {
  if (_rho.f() != nullptr) {
    std::cerr << "Warning: rho already created. Deleting it before recreating\n";
    _rho.del();
  }

  _rho = Function(_nPoints);

  fillFirstRho();
}

void Grid3::fillFirstRho() {
  std::random_device seed;
  // std::cout << "Debug: seed = " << seed() << '\n';
  std::mt19937 gen(seed());
  std::uniform_int_distribution<int> init(0, _nPoints);
  int start_point = init(gen);

  std::pair<int[8], double[8]>& cubic = _projection[start_point];
  for (int j = 0; j < 8; ++j) {
    _rho[cubic.first[j]] = 1. * cubic.second[j];
  }
}

void Grid3::createProjection() {
  std::cout << "Constructing projection...\n";
  _projection = new std::pair<int[8], double[8]>[_nPoints];

  // Precompute strides for the 4D grid (w is slowest index, z is fastest)
  const int strideX = _yPoints * _zPoints;
  const int strideY = _zPoints;
  const int strideZ = 1;

  // Helper: given a grid index along one axis and a +1 offset, returns the
  // clamped neighbour index and flags whether it exists (i.e. is inside the grid).
  // If the point projected onto the surface lands exactly on the max boundary,
  // the cube vertex at coord+_h would be out of domain: we clamp to the boundary
  // itself and assign weight 0 (handled via xd/yd/zd being 0 in that case).

  for (int i = 0; i < _nPoints; ++i) {
    std::cout << '\r' << i << '/' << _nPoints << std::flush;
  
    Point c = closest(_volume[i]);

    // Convert c's coordinates to floating-point grid indices
    // Use round() to snap to the nearest integer — this is robust to the
    // accumulated floating-point error in c (which is a float computed from
    // operations on _h, itself not exactly representable in binary).
    // We take the floor manually: grid index of the lower-left corner.
    double fx = (c.x - _xMin) / _h;
    double fy = (c.y - _yMin) / _h;
    double fz = (c.z - _zMin) / _h;

    // Lower corner indices (clamp to valid range)
    int ix = std::clamp((int)std::floor(fx), 0, _xPoints - 2);
    int iy = std::clamp((int)std::floor(fy), 0, _yPoints - 2);
    int iz = std::clamp((int)std::floor(fz), 0, _zPoints - 2);

    // Interpolation fractions (distance from lower corner, in [0,1])
    double xd = std::clamp(fx - (double)ix, 0., 1.);
    double yd = std::clamp(fy - (double)iy, 0., 1.);
    double zd = std::clamp(fz - (double)iz, 0., 1.);

    // Base linear index of the lower-left-front-bottom corner
    int base = ix * strideX + iy * strideY + iz * strideZ;

    // The 16 vertices of the 4D unit hypercube (w,x,y,z) in {0,1}^4,
    // ordered to match the weight layout used everywhere else in the code.
    _projection[i].first[0]  = base;
    _projection[i].first[1]  = base                          + strideZ;
    _projection[i].first[2]  = base             + strideY;
    _projection[i].first[3]  = base             + strideY    + strideZ;
    _projection[i].first[4]  = base + strideX;
    _projection[i].first[5]  = base + strideX                + strideZ;
    _projection[i].first[6]  = base + strideX   + strideY;
    _projection[i].first[7]  = base + strideX   + strideY    + strideZ;
    
    // Trilinear interpolation weights
    _projection[i].second[0]  = (1-xd) * (1-yd) * (1-zd);
    _projection[i].second[1]  = (1-xd) * (1-yd) *    zd;
    _projection[i].second[2]  = (1-xd) *    yd  * (1-zd);
    _projection[i].second[3]  = (1-xd) *    yd  *    zd;
    _projection[i].second[4]  =    xd  * (1-yd) * (1-zd);
    _projection[i].second[5]  =    xd  * (1-yd) *    zd;
    _projection[i].second[6]  =    xd  *    yd  * (1-zd);
    _projection[i].second[7]  =    xd  *    yd  *    zd;
  }

  std::cout << "\tDone!\n";
}

Point Grid3::closest(Point const& p) {
  double x = p.x;
  double y = p.y;
  double z = p.z;

  double gradient[3];   //nabla_h (phi)
  gradient[0] = (phi(x+_h, y   , z   ) - phi(x-_h, y,    z   )) /(2*_h);
  gradient[1] = (phi(x,    y+_h, z   ) - phi(x,    y-_h, z   )) /(2*_h);
  gradient[2] = (phi(x,    y,    z+_h) - phi(x,    y,    z-_h)) /(2*_h);

  //if gradient = 0, the point is already on the surface
  if (gradient[0] == 0 && gradient[1] == 0 && gradient[2] == 0) {
    return p;
  }

  // | nabla_h (phi) |
  double mod = std::sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1] +
    gradient[2]*gradient[2]);

  double dist_xyz = phi(x,y,z);

  Point r;
  r.x = x - gradient[0] * dist_xyz / mod;
  r.y = y - gradient[1] * dist_xyz / mod;
  r.z = z - gradient[2] * dist_xyz / mod;

  return r;
}

bool Grid3::loadSurface(std::string filename) {
  std::cout << "Loading surface... " << std::flush;
  std::fstream fin(filename + "s.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error: surface file '"<< filename
              << "' couldn't be opened.\n";
    return false;
  }

  fin >> _nPoints;
  fin >> _h;

  Point* temp = new Point[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    fin >> temp[i].x;
    fin >> temp[i].y;
    fin >> temp[i].z;
  }

  _volume = Surface(_nPoints, temp);
  delete[] temp;

  fin.close();
  std::cout << "Done!\n";
  return true;
}

bool Grid3::loadProjection(std::string filename) {
  std::cout << "Loading projection... " << std::flush;
  std::fstream fin(filename + "p.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error: projection file '" << filename
              << "' couldn't be opened.\n";
    return false;
  }

  int nPoints;
  fin >> nPoints;
  if (_nPoints != nPoints) {
    std::cerr << "Error: projection file has uncorrect number of points\n";
    return false;
  }

  _projection = new std::pair<int[8], double[8]>[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    for (int j = 0; j < 8; ++j) {
      fin >> _projection[i].first[j];
      fin >> _projection[i].second[j];
    }
  }

  fin.close();
  std::cout << "Done!\n";
  return true;
}

bool Grid3::loadNeighbour(std::string filename) {
  std::cout << "Loading neighbour... " << std::flush;
  std::fstream fin(filename + "n.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error: neighbour file '"<< filename
              << "' couldn't be opened.\n";
    return false;
  }

  int nPoints;
  fin >> nPoints;
  if (_nPoints != nPoints) {
    std::cerr << "Error: neighbour file has uncorrect number of points\n";
    return false;
  }

  _neighbour = new Neighbours[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    for (int j = 0; j < 19; ++j) {
      fin >> _neighbour[i].point[j];
    }
  }

  fin.close();
  std::cout << "Done!\n";
  return true;
}

bool Grid3::loadRho(std::string filename) {
  std::cout << "Loading rho... " << std::flush;
  std::fstream fin(filename + "r.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error: rho file '"<< filename << "' couldn't be opened.\n";
    return false;
  }

  int nPoints;
  fin >> nPoints;
  if (_nPoints != nPoints) {
    std::cerr << "Error: rho file has different number of points of surface.\n";
    return false;
  }

  _rho = Function(_nPoints);
  for (int i = 0; i < _nPoints; ++i) {
    fin >> _rho[i];
  }

  fin.close();
  std::cout << "Done!\n";
  return true;
}

double Grid3::der1(int pointIndex, int direction) const {
  Neighbours& near = _neighbour[pointIndex];

  double result = 0.;

  switch (direction + 1)
  {
  case 1: //x
    result = (_rho[near.point[2]] - _rho[near.point[1]]);
    break;
  case 2: //y
    result = (_rho[near.point[4]] - _rho[near.point[3]]);
    break;
  case 3: //z
    result = (_rho[near.point[6]] - _rho[near.point[5]]);
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in der1 function: " + direction);
    break;
  }

  return result / (_h * 2);

}

double Grid3::der2(int pointIndex, int direction) const {
  Neighbours& near = _neighbour[pointIndex];

  double result = 0.;

  switch (direction + 1)
  {
  case 1: //xx
    result = _rho[near.point[2]] + _rho[near.point[1]];
    break;
  case 2: //yy
    result = _rho[near.point[4]] + _rho[near.point[3]];
    break;
  case 3: //zz
    result = _rho[near.point[6]] + _rho[near.point[5]];
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in der2 function: " + direction);
    break;
  }

  result -= _rho[near.point[0]] * 2;
  return result / (_h * _h);
}

double Grid3::derij(int pointIndex, int dir1, int dir2) const {
  Neighbours near = _neighbour[pointIndex];

  double result = 0.;

  switch ((dir1 + 1) * (dir2 + 1))
  {
  case 2: //xy || yx
    result = _rho[near.point[16]] - _rho[near.point[15]] -
      _rho[near.point[8]] + _rho[near.point[7]];
    break;
  case 3: //xz || zx
    result = _rho[near.point[18]] - _rho[near.point[17]] -
      _rho[near.point[10]] + _rho[near.point[9]];
    break;
  case 6: //yz || zy
    result = _rho[near.point[14]] - _rho[near.point[13]] -
      _rho[near.point[12]] + _rho[near.point[11]];
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in derij function: " + dir1 + ' ' + dir2);
    break;
  }

  return result / (_h * _h * 4);
}
