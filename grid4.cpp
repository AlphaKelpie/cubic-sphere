#include "grid4.h"
#include "params.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>

Grid4::Grid4(Interval w, Interval x, Interval y, Interval z, double T, double h)
	: _nPoints{0},
		_projection{nullptr},
		_neighbour{nullptr},
    _t{T/6},
		_h{h},
		_wPoints{(int)std::round((w.max - w.min) / h) + 1},
		_xPoints{(int)std::round((x.max - x.min) / h) + 1},
		_yPoints{(int)std::round((y.max - y.min) / h) + 1},
		_zPoints{(int)std::round((z.max - z.min) / h) + 1},
		_wMin{w.min},
		_xMin{x.min},
		_yMin{y.min},
		_zMin{z.min}
{
	_nPoints = _wPoints * _xPoints * _yPoints * _zPoints;

	// Create surface and _neighbour
	Quaternion* temp = new Quaternion[_nPoints];

	_neighbour = new Neighbours[_nPoints];

  std::cout << "Construct volume and neighbour...\n";
	//for every point of the descrete domain..
  bool neg;
  bool pos;
	bool left;
	bool right;
	bool front;
	bool behind;
	bool down;
	bool top;
  int index = 0;
  double m;
  double i;
  double j;
  double k;
  for (short mi = 0; mi < _wPoints; ++mi) {
    m = w.min + _h * mi;
    neg = (mi <= 0) ? false : true;
    pos = (mi >= _wPoints - 1) ? false : true;
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
          temp[index] = {m, i, j, k};
          //..create corresponding neighbour
          _neighbour[index].point[0] = index;   // itself
          _neighbour[index].point[1] = (neg) ? (index - _xPoints * _yPoints * _zPoints) : -1;

          _neighbour[index].point[2] = (pos) ? (index + _xPoints * _yPoints * _zPoints) : -1;

          _neighbour[index].point[3] = (left) ? (index - _yPoints * _zPoints) : -1;

          _neighbour[index].point[4] = (right) ? (index + _yPoints * _zPoints) : -1;

          _neighbour[index].point[5] = (front) ? (index - _zPoints) : -1;

          _neighbour[index].point[6] = (behind) ? (index + _zPoints) : -1;

          _neighbour[index].point[7] = (down) ? (index - 1) : -1;

          _neighbour[index].point[8] = (top) ? (index + 1) : -1;

          _neighbour[index].point[9] = (neg && left) ? (index - (_xPoints + 1) * _yPoints * _zPoints) : -1;

          _neighbour[index].point[10] = (neg && right) ? (index - (_xPoints - 1) * _yPoints * _zPoints) : -1;

          _neighbour[index].point[11] = (neg && front) ? (index - (_xPoints * _yPoints + 1) * _zPoints) : -1;

          _neighbour[index].point[12] = (neg && behind) ? (index - (_xPoints * _yPoints - 1) * _zPoints) : -1;

          _neighbour[index].point[13] = (neg && down) ? (index - (_xPoints * _yPoints * _zPoints + 1)) : -1;

          _neighbour[index].point[14] = (neg && top) ? (index - (_xPoints * _yPoints * _zPoints - 1)) : -1;

          _neighbour[index].point[15] = (pos && left) ? (index + (_xPoints - 1) * _yPoints * _zPoints) : -1;

          _neighbour[index].point[16] = (pos && right) ? (index + (_xPoints + 1) * _yPoints * _zPoints) : -1;

          _neighbour[index].point[17] = (pos && front) ? (index + (_xPoints * _yPoints - 1) * _zPoints) : -1;

          _neighbour[index].point[18] = (pos && behind) ? (index + (_xPoints * _yPoints + 1) * _zPoints) : -1;

          _neighbour[index].point[19] = (pos && down) ? (index + (_xPoints * _yPoints * _zPoints - 1)) : -1;

          _neighbour[index].point[20] = (pos && top) ? (index + (_xPoints * _yPoints * _zPoints + 1)) : -1;

          _neighbour[index].point[21] = (left && front) ? (index - (_yPoints + 1) * _zPoints) : -1;

          _neighbour[index].point[22] = (left && behind) ? (index - (_yPoints - 1) * _zPoints) : -1;

          _neighbour[index].point[23] = (left && down) ? (index - (_yPoints * _zPoints + 1)) : -1;

          _neighbour[index].point[24] = (left && top) ? (index - (_yPoints * _zPoints - 1)) : -1;

          _neighbour[index].point[25] = (right && front) ? (index + (_yPoints - 1) * _zPoints) : -1;

          _neighbour[index].point[26] = (right && behind) ? (index + (_yPoints + 1) * _zPoints) : -1;

          _neighbour[index].point[27] = (right && down) ? (index + (_yPoints * _zPoints - 1)) : -1;

          _neighbour[index].point[28] = (right && top) ? (index + (_yPoints * _zPoints + 1)) : -1;

          _neighbour[index].point[29] = (front && down) ? (index - (_zPoints + 1)) : -1;

          _neighbour[index].point[30] = (front && top) ? (index - (_zPoints - 1)) : -1;

          _neighbour[index].point[31] = (behind && down) ? (index + (_zPoints - 1)) : -1;

          _neighbour[index].point[32] = (behind && top) ? (index + (_zPoints + 1)) : -1;

          ++index;
        }
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

Grid4::Grid4(std::string surfFile, std::string relFile, double T)
	: _nPoints{0},
    _t{T/6},
    _wPoints{0}, _xPoints{0}, _yPoints{0}, _zPoints{0},
    _wMin{0.}, _xMin{0.}, _yMin{0.}, _zMin{0.}
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

Grid4::~Grid4() {
	delete[] _projection;
	delete[] _neighbour;
}

void Grid4::fillFirstRho() {
  std::random_device seed;
  // std::cout << "Debug: seed = " << seed() << '\n';
  std::mt19937 gen(seed());
  std::uniform_int_distribution<int> init(0, _nPoints);
  int start_point = init(gen);

  std::pair<int[16], double[16]>& cubic = _projection[start_point];
  for (int j = 0; j < 16; ++j) {
    if (cubic.first[j] >= 0)
    _rho[cubic.first[j]] = 1. * cubic.second[j];
  }
}

void Grid4::fillIdxRho(int idx) {
  _rho[idx] = 1.;
}

void Grid4::saveProjection(std::string filename) {
	// format:
  // first row: _nPoints
	// 1 row per point to _projection: index and weight for the 16 grid
	// points around

	std::cout << "Saving projection... " << std::flush;
	std::fstream fout(filename + "p.dat", std::ios::out);

	fout << _nPoints << "\n\n";

	for (int i = 0; i < _nPoints; ++i) {
		for (int j = 0; j < 16; ++j) {
			fout << _projection[i].first[j] << ' ';
			fout << _projection[i].second[j] << ' ';
		}
		fout << "\n";
	}

	fout.close();
	std::cout << "Done!\n";
}

void Grid4::saveNeighbour(std::string filename) {
  // format:
  // first row: _nPoints
	// 1 row per point to _neighbour indexes (33 int)

	std::cout << "Saving neighbour... " << std::flush;
	std::fstream fout(filename + "n.dat", std::ios::out);

	fout << _nPoints << "\n\n";

	for (int i = 0; i < _nPoints; ++i) {
		for (int j = 0; j < 33; ++j) {
			fout << _neighbour[i].point[j] << ' ';
		}
		fout << "\n";
	}

	fout.close();
	std::cout << "Done!\n";
}

void Grid4::saveSurface(std::string filename) {
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

void Grid4::saveRho(std::string filename) {
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

void Grid4::evolve(double dt) {
  if (dt == 0) {
    dt = 0.1 * _h * _h * _h * _h;
  }

  Function nextRho = Function(_nPoints);

  double deltaRho = 0.;
  for (int idx = 0; idx < _nPoints; ++idx) {
    Quaternion const& p = _volume[idx];

    for (int j = 0; j < 4; ++j) {
      deltaRho += p[j] * (1 - 4) * der1(idx, j);
      for (int i = 0; i < 4; ++i) {
        if (i == j) {
          deltaRho += p.D(i) * der2(idx, i);
        } else {
          deltaRho += p.D(i, j) * derij(idx, i, j);
        }
      }
    }

    deltaRho *= _t * dt;
    nextRho[idx] = _rho[idx] + deltaRho;
    deltaRho = 0.;
  }

  _rho = nextRho;
}

void Grid4::evolveCN(double dt, int maxIter, double tol) {
  if (dt == 0) {
    dt = 0.1 * _h * _h * _h * _h;
  }

  double alpha = _t * dt * 0.5;

  // Explicit half-step: rhs = rho^n + alpha * L(rho^n)
  Function rhs(_nPoints);
  for (int idx = 0; idx < _nPoints; ++idx) {
    Quaternion const& p = _volume[idx];
    double Lrho = -_rho[idx]*(1-4)*4; // from sum_i[D'_i rho * a(x)]
    for (int i = 0; i < 4; ++i) {
      Lrho -= p[i]*(1-4)*der1(idx, i);  // from D'_i rho * a(x)
      for (int j = 0; j < 4; ++j) {
        if (i == j) { // from D''_ii rho * D
          Lrho += der2(idx, i, p);
        } else {  // from D''_ij rho * D
          Lrho += derij(idx, i, j, p);
        }
      }
    }
    rhs[idx] = _rho[idx] + alpha * Lrho;
  }

  // Solve [I - alpha*L] rho^{n+1} = rhs via fixed-point iteration
  Function rhoA = _rho;
  Function rhoB(_nPoints);

  for (int iter = 0; iter < maxIter; ++iter) {
    double maxDiff = 0.;

    for (int idx = 0; idx < _nPoints; ++idx) {
      Quaternion const& p = _volume[idx];
      double Lrho = -rhoA[idx]*(1-4)*4; // from sum_i[D'_i rho * a(x)]
      for (int i = 0; i < 4; ++i) {
        Lrho -= p[i]*(1-4)*der1(idx, i);  // from D'_i rho * a(x)
        for (int j = 0; j < 4; ++j) {
          if (i == j) { // from D''_ii rho * D
            Lrho += der2(idx, i, rhoA, p);
          } else {  // from D''_ij rho * D
            Lrho += derij(idx, i, j, rhoA, p);
          }
        }
      }
      rhoB[idx] = _rho[idx] + alpha * Lrho;   // ATTENZIONE: Claude aveva messo rhs[idx], io ho messo _rho[idx]

      double diff = std::abs(rhoB[idx] - rhoA[idx]);
      if (diff > maxDiff) maxDiff = diff;
    }

    std::swap(rhoA, rhoB);

    if (maxDiff < tol) {
      break;
    }
  }

  _rho = std::move(rhoA);
}

void Grid4::evolveCNDiff(double dt, int maxIter, double tol) {
  if (dt == 0) {
    dt = 0.1 * _h * _h * _h * _h;
  }

  double alpha = _t * dt * 0.5;

  // Explicit half-step: rhs = rho^n + alpha * L(rho^n)
  Function rhs(_nPoints);
  for (int idx = 0; idx < _nPoints; ++idx) {
    double Lrho = 0.;
    for (int j = 0; j < 4; ++j) {
      Lrho += der2(idx, j);
    }
    rhs[idx] = _rho[idx] + alpha * Lrho;
  }

  // Solve [I - alpha*L] rho^{n+1} = rhs via fixed-point iteration
  Function rhoA = _rho;
  Function rhoB(_nPoints);

  for (int iter = 0; iter < maxIter; ++iter) {
    double maxDiff = 0.;

    for (int idx = 0; idx < _nPoints; ++idx) {
      double Lrho = 0.;
      for (int j = 0; j < 4; ++j) {
        Lrho += der2(idx, j, rhoA);
      }
      rhoB[idx] = rhs[idx] + alpha * Lrho;

      double diff = std::abs(rhoB[idx] - rhoA[idx]);
      if (diff > maxDiff) maxDiff = diff;
    }

    std::swap(rhoA, rhoB);

    if (maxDiff < tol) {
      break;
    }
  }

  _rho = std::move(rhoA);
}

void Grid4::project() {
  Function sphere_rho(_nPoints);

  for (int i = 0; i < _nPoints; ++i) {
    double rho = _rho[i];
    if (rho == 0.) { continue; }
    std::pair<int[16], double[16]>& cubic = _projection[i];
    for (int j = 0; j < 16; ++j) {
      sphere_rho[cubic.first[j]] += rho * cubic.second[j];
    }
  }

  _rho = sphere_rho;
}


// private


double Grid4::phi(double w, double x, double y, double z) {
  Quaternion& c = Params::get().center;
  return std::sqrt(
      (w-c.w)*(w-c.w) + (x-c.x)*(x-c.x) + (y-c.y)*(y-c.y) + (z-c.z)*(z-c.z)
    ) - Params::get().radius;
}

void Grid4::createRho() {
  if (_rho.f() != nullptr) {
    std::cerr << "Warning: rho already created. Deleting it before recreating\n";
    _rho.del();
  }

  _rho = Function(_nPoints);
}

void Grid4::createProjection() {
  std::cout << "Constructing projection...\n";
  _projection = new std::pair<int[16], double[16]>[_nPoints];

  // Precompute strides for the 4D grid (w is slowest index, z is fastest)
  const int strideW = _xPoints * _yPoints * _zPoints;
  const int strideX = _yPoints * _zPoints;
  const int strideY = _zPoints;
  const int strideZ = 1;

  // Helper: given a grid index along one axis and a +1 offset, returns the
  // clamped neighbour index and flags whether it exists (i.e. is inside the grid).
  // If the point projected onto the surface lands exactly on the max boundary,
  // the cube vertex at coord+_h would be out of domain: we clamp to the boundary
  // itself and assign weight 0 (handled via wd/xd/yd/zd being 0 in that case).

  for (int i = 0; i < _nPoints; ++i) {
    std::cout << '\r' << i << '/' << _nPoints << std::flush;

    Quaternion c = closest(_volume[i]);

    // Convert c's coordinates to floating-point grid indices
    // Use round() to snap to the nearest integer — this is robust to the
    // accumulated floating-point error in c (which is a float computed from
    // operations on _h, itself not exactly representable in binary).
    // We take the floor manually: grid index of the lower-left corner.
    double fw = (c.w - _wMin) / _h;
    double fx = (c.x - _xMin) / _h;
    double fy = (c.y - _yMin) / _h;
    double fz = (c.z - _zMin) / _h;

    // Lower corner indices (clamp to valid range)
    int iw = std::clamp((int)std::floor(fw), 0, _wPoints - 2);
    int ix = std::clamp((int)std::floor(fx), 0, _xPoints - 2);
    int iy = std::clamp((int)std::floor(fy), 0, _yPoints - 2);
    int iz = std::clamp((int)std::floor(fz), 0, _zPoints - 2);

    // Interpolation fractions (distance from lower corner, in [0,1])
    double wd = std::clamp(fw - (double)iw, 0., 1.);
    double xd = std::clamp(fx - (double)ix, 0., 1.);
    double yd = std::clamp(fy - (double)iy, 0., 1.);
    double zd = std::clamp(fz - (double)iz, 0., 1.);

    // Base linear index of the lower-left-front-bottom corner
    int base = iw * strideW + ix * strideX + iy * strideY + iz * strideZ;

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
    _projection[i].first[8]  = base + strideW;
    _projection[i].first[9]  = base + strideW                + strideZ;
    _projection[i].first[10] = base + strideW   + strideY;
    _projection[i].first[11] = base + strideW   + strideY    + strideZ;
    _projection[i].first[12] = base + strideW   + strideX;
    _projection[i].first[13] = base + strideW   + strideX    + strideZ;
    _projection[i].first[14] = base + strideW   + strideX    + strideY;
    _projection[i].first[15] = base + strideW   + strideX    + strideY + strideZ;

    // Trilinear interpolation weights
    _projection[i].second[0]  = (1-wd) * (1-xd) * (1-yd) * (1-zd);
    _projection[i].second[1]  = (1-wd) * (1-xd) * (1-yd) *    zd;
    _projection[i].second[2]  = (1-wd) * (1-xd) *    yd  * (1-zd);
    _projection[i].second[3]  = (1-wd) * (1-xd) *    yd  *    zd;
    _projection[i].second[4]  = (1-wd) *    xd  * (1-yd) * (1-zd);
    _projection[i].second[5]  = (1-wd) *    xd  * (1-yd) *    zd;
    _projection[i].second[6]  = (1-wd) *    xd  *    yd  * (1-zd);
    _projection[i].second[7]  = (1-wd) *    xd  *    yd  *    zd;
    _projection[i].second[8]  =    wd  * (1-xd) * (1-yd) * (1-zd);
    _projection[i].second[9]  =    wd  * (1-xd) * (1-yd) *    zd;
    _projection[i].second[10] =    wd  * (1-xd) *    yd  * (1-zd);
    _projection[i].second[11] =    wd  * (1-xd) *    yd  *    zd;
    _projection[i].second[12] =    wd  *    xd  * (1-yd) * (1-zd);
    _projection[i].second[13] =    wd  *    xd  * (1-yd) *    zd;
    _projection[i].second[14] =    wd  *    xd  *    yd  * (1-zd);
    _projection[i].second[15] =    wd  *    xd  *    yd  *    zd;
  }

  std::cout << "\tDone!\n";
}

Quaternion Grid4::closest(Quaternion const& p) {
  double w = p.w;
  double x = p.x;
  double y = p.y;
  double z = p.z;

  double gradient[4];   //nabla_h (phi)
  gradient[0] = (phi(w+_h, x,    y,    z   ) - phi(w-_h, x,    y,    z   )) /(2*_h);
  gradient[1] = (phi(w,    x+_h, y   , z   ) - phi(w,    x-_h, y,    z   )) /(2*_h);
  gradient[2] = (phi(w,    x,    y+_h, z   ) - phi(w,    x,    y-_h, z   )) /(2*_h);
  gradient[3] = (phi(w,    x,    y,    z+_h) - phi(w,    x,    y,    z-_h)) /(2*_h);

  //if gradient = 0, the point is already on the surface
  if (gradient[0] == 0 && gradient[1] == 0 && gradient[2] == 0 && gradient[3] == 0) {
    return p;
  }

  // | nabla_h (phi) |
  double mod = std::sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1] +
    gradient[2]*gradient[2] + gradient[3]*gradient[3]);

  double dist_wxyz = phi(w,x,y,z);

  Quaternion r;
  r.w = w - gradient[0] * dist_wxyz / mod;
  r.x = x - gradient[1] * dist_wxyz / mod;
  r.y = y - gradient[2] * dist_wxyz / mod;
  r.z = z - gradient[3] * dist_wxyz / mod;

  return r;
}

bool Grid4::loadSurface(std::string filename) {
  std::cout << "Loading surface... " << std::flush;
  std::fstream fin(filename + "s.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error: surface file '"<< filename
              << "' couldn't be opened.\n";
    return false;
  }

  fin >> _nPoints;
  fin >> _h;

  Quaternion* temp = new Quaternion[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    fin >> temp[i].w;
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

bool Grid4::loadProjection(std::string filename) {
  std::cout << "Loading projection... " << std::flush;
  std::fstream fin(filename + "p.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error: projection file '"<< filename
              << "' couldn't be opened.\n";
    return false;
  }

  int nPoints;
  fin >> nPoints;
  if (_nPoints != nPoints) {
    std::cerr << "Error: projection file has uncorrect number of points\n";
    return false;
  }

  _projection = new std::pair<int[16], double[16]>[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    for (int j = 0; j < 16; ++j) {
      fin >> _projection[i].first[j];
      fin >> _projection[i].second[j];
    }
  }

  fin.close();
  std::cout << "Done!\n";
  return true;
}

bool Grid4::loadNeighbour(std::string filename) {
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
    for (int j = 0; j < 33; ++j) {
      fin >> _neighbour[i].point[j];
    }
  }

  fin.close();
  std::cout << "Done!\n";
  return true;
}

bool Grid4::loadRho(std::string filename) {
  std::cout << "Loading rho... " << std::flush;
  std::fstream fin(filename + "r.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error: rho file '"<< filename << "' couldn't be opened.\n";
    return false;
  }

  int nPoints;
  fin >> nPoints;
  if (_nPoints != nPoints) {
    std::cerr << "Error: rho file has uncorrect number of points\n";
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

double Grid4::der1(int pointIndex, int direction) const {
  return der1(pointIndex, direction, _rho);
}

double Grid4::der2(int pointIndex, int direction, Quaternion const& pos) const {
  return der2(pointIndex, direction, _rho, pos);
}

double Grid4::derij(int pointIndex, int dir1, int dir2, Quaternion const& pos) const {
  return derij(pointIndex, dir1, dir2, _rho, pos);
}

double Grid4::der1(int pointIndex, int direction, Function const& f) const {
  Neighbours& near = _neighbour[pointIndex];

  double result = 0.;

  switch (direction + 1)
  {
  case 1: //w
    result = (f[near.point[2]] - f[near.point[1]]);
    break;
  case 2: //x
    result = (f[near.point[4]] - f[near.point[3]]);
    break;
  case 3: //y
    result = (f[near.point[6]] - f[near.point[5]]);
    break;
  case 4: //z
    result = (f[near.point[8]] - f[near.point[7]]);
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in der1 function: " + direction);
    break;
  }

  return result / (_h * 2);
}

double Grid4::der2(int pointIndex, int direction, Function const& f, Quaternion const& pos) const {
  Neighbours& near = _neighbour[pointIndex];

  double result = 0.;
  int const exp = (pos == Quaternion{-1., -1., -1., -1.}) ? 0 : 1;
  double const hhalf = _h / 2;

  switch (direction + 1)
  { // ATTENZIONE: qui D è posizionato a mezzo step (come Numerical Recepies), non _h intero come derij()
  case 1: //ww
    result = std::pow(pos.D(0,hhalf),exp)*(f[near.point[2]] - f[near.point[0]]) +
      std::pow(pos.D(0,-hhalf),exp)*(f[near.point[1]] - f[near.point[0]]);
    break;
  case 2: //xx
    result = std::pow(pos.D(1,hhalf),exp)*(f[near.point[4]] - f[near.point[0]]) +
      std::pow(pos.D(1,-hhalf),exp)*(f[near.point[3]] - f[near.point[0]]);
    break;
  case 3: //yy
    result = std::pow(pos.D(2,hhalf),exp)*(f[near.point[6]] - f[near.point[0]]) +
      std::pow(pos.D(2,-hhalf),exp)*(f[near.point[5]] - f[near.point[0]]);
    break;
  case 4: //zz
    result = std::pow(pos.D(3,hhalf),exp)*(f[near.point[8]] - f[near.point[0]]) +
      std::pow(pos.D(3,-hhalf),exp)*(f[near.point[7]] - f[near.point[0]]);
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in der2 function: " + direction);
    break;
  }

  return result / (_h * _h);
}

double Grid4::derij(int pointIndex, int dir1, int dir2, Function const& f, Quaternion const& pos) const {
  Neighbours near = _neighbour[pointIndex];

  double result = 0.;
  int exp = (pos == Quaternion{-1., -1., -1., -1.}) ? 0 : 1;

  switch ((dir1 + 1) * (dir2 + 1))
  {
  case 2: //wx || xw
    result = f[near.point[16]]*std::pow(pos.D(0,1,_h,_h),exp) -
      f[near.point[15]]*std::pow(pos.D(0,1,_h,-_h),exp) -
      f[near.point[10]]*std::pow(pos.D(0,1,-_h,_h),exp) +
      f[near.point[9]]*std::pow(pos.D(0,1,-_h,-_h),exp);
    break;
  case 3: //wy || yw
    result = f[near.point[18]]*std::pow(pos.D(0,2,_h,_h),exp) -
      f[near.point[17]]*std::pow(pos.D(0,2,_h,-_h),exp) -
      f[near.point[12]]*std::pow(pos.D(0,2,-_h,_h),exp) +
      f[near.point[11]]*std::pow(pos.D(0,2,-_h,-_h),exp);
    break;
  case 4: //wz || zw
    result = f[near.point[20]]*std::pow(pos.D(0,3,_h,_h),exp) -
      f[near.point[19]]*std::pow(pos.D(0,3,_h,-_h),exp) -
      f[near.point[14]]*std::pow(pos.D(0,3,-_h,_h),exp) +
      f[near.point[13]]*std::pow(pos.D(0,3,-_h,-_h),exp);
    break;
  case 6: //xy || yx
    result = f[near.point[26]]*std::pow(pos.D(1,2,_h,_h),exp) -
      f[near.point[25]]*std::pow(pos.D(1,2,_h,-_h),exp) -
      f[near.point[22]]*std::pow(pos.D(1,2,-_h,_h),exp) +
      f[near.point[21]]*std::pow(pos.D(1,2,-_h,-_h),exp);
    break;
  case 8: //xz || zx
    result = f[near.point[28]]*std::pow(pos.D(1,3,_h,_h),exp) -
      f[near.point[27]]*std::pow(pos.D(1,3,_h,-_h),exp) -
      f[near.point[24]]*std::pow(pos.D(1,3,-_h,_h),exp) +
      f[near.point[23]]*std::pow(pos.D(1,3,-_h,-_h),exp);
    break;
  case 12: //yz || zy
    result = f[near.point[32]]*std::pow(pos.D(2,3,_h,_h),exp) -
      f[near.point[31]]*std::pow(pos.D(2,3,_h,-_h),exp) -
      f[near.point[30]]*std::pow(pos.D(2,3,-_h,_h),exp) +
      f[near.point[29]]*std::pow(pos.D(2,3,-_h,-_h),exp);
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in derij function: " + dir1 + ' ' + dir2);
    break;
  }

  return result / (_h * _h * 4);
}
