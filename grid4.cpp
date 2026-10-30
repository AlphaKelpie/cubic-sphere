#include "grid4.h"
#include "params.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>

template <typename Tf>
Grid4<Tf>::Grid4(Interval<Tf> w, Interval<Tf> x, Interval<Tf> y, Interval<Tf> z, Tf T, Tf h)
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
  
  if (Params<Tf>::get().deltaC == 0.) {  //whole hypercube
    std::cout << "Whole hypercube model\n";
    createVolNeigh(w, x, y, z);
    _indices = std::vector<int>(_nPoints);
    std::iota(_indices.begin(), _indices.end(), 0);
    _diff = std::vector<float>(_nPoints, 0);
    createProjection();
  }
  else { //spheric crown
    std::cout << "Spheric crown model: DeltaC = " << Params<Tf>::get().deltaC
              << std::endl;
    createVol(w, x, y, z);
    _indices = std::vector<int>(_nPoints);
    std::iota(_indices.begin(), _indices.end(), 0);
    _diff = std::vector<float>(_nPoints, 0);
    createMaps();
  }
	createRho();
}

template <typename Tf>
Grid4<Tf>::Grid4(std::string surfFile, std::string relFile, Tf T)
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
  if (!loadRho(relFile+"_init")) {
    std::cerr << "Warning: unable to load initial rho. Recreating";
  	createRho();
  }

  // set rho value in the center of the hypercube as zero
  _centerPoint = (_nPoints - 1) / 2;
  _rho[_centerPoint] = 0.;

  // initialise std::for_each elements
  _indices = std::vector<int>(_nPoints);
  std::iota(_indices.begin(), _indices.end(), 0);
  _diff = std::vector<float>(_nPoints, 0);
}

template <typename Tf>
Grid4<Tf>::~Grid4() {
	delete[] _projection;
	delete[] _neighbour;
}

template <typename Tf>
void Grid4<Tf>::fillFirstRho() {
  std::random_device seed;
  // std::cout << "Debug: seed = " << seed() << '\n';
  std::mt19937 gen(seed());
  std::uniform_int_distribution<int> init(0, _nPoints);
  int start_point = init(gen);
  if (start_point == _centerPoint) start_point += 10;

  std::pair<int[16], Tf[16]>& cubic = _projection[start_point];
  for (int j = 0; j < 16; ++j) {
    _rho[cubic.first[j]] = 1. * cubic.second[j];
  }
}

template <typename Tf>
void Grid4<Tf>::fillIdxRho(int idx) {
  _rho[idx] = 1.;
}

template <typename Tf>
void Grid4<Tf>::saveProjection(std::string filename) {
	// format:
  // first row: _nPoints
	// 1 row per point to _projection: index and weight for the 16 grid
	// points around

	std::cout << "Saving projection... " << std::flush;
	std::fstream fout(filename + "_p.dat", std::ios::out);

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

template <typename Tf>
void Grid4<Tf>::saveNeighbour(std::string filename) {
  // format:
  // first row: _nPoints
	// 1 row per point to _neighbour indexes (33 int)

	std::cout << "Saving neighbour... " << std::flush;
	std::fstream fout(filename + "_n.dat", std::ios::out);

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

template <typename Tf>
void Grid4<Tf>::saveSurface(std::string filename) {
  // format:
  // first row: _nPoints _h
  // 1 row per point in _volume

  std::cout << "Saving surface... " << std::flush;
  if (_volume.nPoints() == 0) {
    std::cerr << "Error: cannot save surface (probably empty)\n";
    return;
  }

  std::fstream fout(filename + "_s.dat", std::ios::out);

  fout << _nPoints << ' ' << _h << "\n\n";

  fout << _volume;

  fout.close();
  std::cout << "Done!\n";
}

template <typename Tf>
void Grid4<Tf>::saveRho(std::string filename) {
  // format:
  // first row: _nPoints
  // 1 row per _rho value

  std::cout << "Saving rho... " << std::flush;

  std::fstream fout(filename + "_r.dat", std::ios::out);

  fout << _nPoints << "\n\n";

  for (int i = 0; i < _nPoints; ++i) {
    fout << _rho[i] << '\n';
  }

  fout.close();
  std::cout << "Done!\n";
}

template <typename Tf>
void Grid4<Tf>::evolve(Tf dt) {
  if (dt == 0) {
    dt = 0.1 * _h * _h * _h * _h;
  }

  Function<Tf> nextRho(_nPoints);

  std::for_each(std::execution::par_unseq, _indices.begin(), _indices.end(), [&](int idx) {
    Quaternion<Tf> const& p = _volume[idx];

    Tf deltaRho = _rho[idx] * (1 - 4)*4;
    for (int i = 0; i < 4; ++i) {
      deltaRho -= p[i]*(1-4)*der1(idx, i);
      for (int j = 0; j < 4; ++j) {
        if (i == j) {
          deltaRho += der2(idx, i, p);
        } else {
          deltaRho += derij(idx, i, j, p);
        }
      }
    }

    deltaRho *= _t * dt;
    nextRho[idx] = _rho[idx] + deltaRho;
  });

  _rho = std::move(nextRho);
  _rho[_centerPoint] = 0.;
}

template <typename Tf>
void Grid4<Tf>::evolveCN(Tf dt, int maxIter, Tf tol) {
  if (dt == 0) {
    dt = 0.1 * _h * _h * _h * _h;
  }

  Tf alpha = _t * dt * 0.5;

  // Explicit half-step: rhs = rho^n + alpha * L(rho^n)
  Function<Tf> rhs(_nPoints);
  std::for_each(std::execution::par_unseq, _indices.begin(), _indices.end(), [&](int idx) {
    Quaternion<Tf> const& p = _volume[idx];
    Tf Lrho = -_rho[idx]*(1-4)*4; // from sum_i[D'_i rho * a(x)]
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
  });
  rhs[_centerPoint] = 0.;

  // Solve [I - alpha*L] rho^{n+1} = rhs via fixed-point iteration
  Function<Tf> rhoA = _rho;
  Function<Tf> rhoB(_nPoints);

  for (int iter = 0; iter < maxIter; ++iter) {

    std::for_each(std::execution::par_unseq, _indices.begin(), _indices.end(), [&](int idx) {
      Quaternion<Tf> const& p = _volume[idx];
      Tf Lrho = -rhoA[idx]*(1-4)*4; // from sum_i[D'_i rho * a(x)]
      for (int i = 0; i < 4; ++i) {
        Lrho -= p[i]*(1-4)*der1(idx, i, rhoA);  // from D'_i rho * a(x)
        for (int j = 0; j < 4; ++j) {
          if (i == j) { // from D''_ii rho * D
            Lrho += der2(idx, i, rhoA, p);
          } else {  // from D''_ij rho * D
            Lrho += derij(idx, i, j, rhoA, p);
          }
        }
      }
      rhoB[idx] = rhs[idx] + alpha * Lrho;

      _diff[idx] = std::abs(rhoB[idx] - rhoA[idx]);
    });

    std::vector<float>::iterator maxIt = std::max_element(
      std::execution::par_unseq, _diff.begin(), _diff.end());

    std::swap(rhoA, rhoB);

    if (*maxIt < tol) {
      break;
    }
    std::fill(_diff.begin(), _diff.end(), 0.);
  }

  rhoA[_centerPoint] = 0.;
  _rho = std::move(rhoA);
}

template <typename Tf>
void Grid4<Tf>::evolveCNDiff(Tf dt, int maxIter, Tf tol) {
  if (dt == 0) {
    dt = 0.1 * _h * _h * _h * _h;
  }

  Tf alpha = _t * dt * 0.5;

  // Explicit half-step: rhs = rho^n + alpha * L(rho^n)
  Function<Tf> rhs(_nPoints);
  for (int idx = 0; idx < _nPoints; ++idx) {
    Tf Lrho = 0.;
    for (int j = 0; j < 4; ++j) {
      Lrho += der2(idx, j);
    }
    rhs[idx] = _rho[idx] + alpha * Lrho;
  }

  // Solve [I - alpha*L] rho^{n+1} = rhs via fixed-point iteration
  Function<Tf> rhoA = _rho;
  Function<Tf> rhoB(_nPoints);

  for (int iter = 0; iter < maxIter; ++iter) {
    Tf maxDiff = 0.;

    for (int idx = 0; idx < _nPoints; ++idx) {
      Tf Lrho = 0.;
      for (int j = 0; j < 4; ++j) {
        Lrho += der2(idx, j, rhoA);
      }
      rhoB[idx] = rhs[idx] + alpha * Lrho;

      Tf diff = std::abs(rhoB[idx] - rhoA[idx]);
      if (diff > maxDiff) maxDiff = diff;
    }

    std::swap(rhoA, rhoB);

    if (maxDiff < tol) {
      break;
    }
  }

  _rho = std::move(rhoA);
}

template <typename Tf>
void Grid4<Tf>::project() {
  Function<Tf> sphere_rho(_nPoints);

  for (int i = 0; i < _nPoints; ++i) {
    Tf rho = _rho[i];
    if (rho == 0.) { continue; }
    std::pair<int[16], Tf[16]>& cubic = _projection[i];
    for (int j = 0; j < 16; ++j) {
      sphere_rho[cubic.first[j]] += rho * cubic.second[j];
    }
  }

  _rho = std::move(sphere_rho);   //ATTENZIONE
}


// private


template <typename Tf>
void Grid4<Tf>::createVolNeigh(Interval<Tf> w, Interval<Tf> x, Interval<Tf> y, Interval<Tf> z) {
  std::cout << "Construct volume and neighbour...\n";

  _nPoints = _wPoints * _xPoints * _yPoints * _zPoints;
	// Create surface
	Quaternion<Tf>* temp = new Quaternion<Tf>[_nPoints];
	_neighbour = new Neighbours[_nPoints];

  bool neg;
  bool pos;
	bool left;
	bool right;
	bool front;
	bool behind;
	bool down;
	bool top;
  int index = 0;
  Tf m;
  Tf i;
  Tf j;
  Tf k;
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
}

template <typename Tf>
void Grid4<Tf>::createVol(Interval<Tf> w, Interval<Tf> x, Interval<Tf> y, Interval<Tf> z) {
  std::cout << "Construct volume...\n";
  int cubicnPoints = _wPoints * _xPoints * _yPoints * _zPoints;

	// Create surface
	Quaternion<Tf>* temp = new Quaternion<Tf>[cubicnPoints];

	//for every point of the descrete domain..
  int fullIndex = 0;
  int index = 0;
  Tf m;
  Tf i;
  Tf j;
  Tf k;
  for (short mi = 0; mi < _wPoints; ++mi) {
    m = w.min + _h * mi;
    for (short ii = 0; ii < _xPoints; ++ii) {
      i = x.min + _h * ii;
      for (short ji = 0; ji < _yPoints; ++ji) {
        j = y.min + _h * ji;
        for (short ki = 0; ki < _zPoints; ++ki) {
          k = z.min + _h * ki;
          ++fullIndex;
          std::cout << '\r' << fullIndex << '/' << cubicnPoints << std::flush;

          if (std::abs(phi(m, i, j, k)) > Params<Tf>::get().deltaC) { continue; }

          //..add it to the volume
          temp[index] = {m, i, j, k};
          ++index;
        }
      }
    }
  }
  if (index >= cubicnPoints) {
    throw std::runtime_error("Error: _neighbour index greater than _nPoints");
  }

  _nPoints = index;
	_volume = Surface(_nPoints, temp);
	delete[] temp;
  std::cout << "\tDone! Final # of points: " << _nPoints << std::endl;
}

template <typename Tf>
void Grid4<Tf>::createProjection() {
  std::cout << "Constructing projection...\n";
  _projection = new std::pair<int[16], Tf[16]>[_nPoints];

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
    Tf fw = (c.w - _wMin) / _h;
    Tf fx = (c.x - _xMin) / _h;
    Tf fy = (c.y - _yMin) / _h;
    Tf fz = (c.z - _zMin) / _h;

    // Lower corner indices (clamp to valid range)
    int iw = std::clamp((int)std::floor(fw), 0, _wPoints - 2);
    int ix = std::clamp((int)std::floor(fx), 0, _xPoints - 2);
    int iy = std::clamp((int)std::floor(fy), 0, _yPoints - 2);
    int iz = std::clamp((int)std::floor(fz), 0, _zPoints - 2);

    // Interpolation fractions (distance from lower corner, in [0,1])
    Tf wd = fw - (Tf)iw;
    Tf xd = fx - (Tf)ix;
    Tf yd = fy - (Tf)iy;
    Tf zd = fz - (Tf)iz;
    if (wd > 1. || wd < 0.) {
      throw std::runtime_error("\nError: weight wd out of range [0, 1]");
    }
    if (xd > 1. || xd < 0.) {
      throw std::runtime_error("\nError: weight xd out of range [0, 1]");
    }
    if (yd > 1. || yd < 0.) {
      throw std::runtime_error("\nError: weight yd out of range [0, 1]");
    }
    if (zd > 1. || zd < 0.) {
      throw std::runtime_error("\nError: weight zd out of range [0, 1]");
    }

    // Base linear index of the lower-left-front-bottom corner
    int base = iw * strideW + ix * strideX + iy * strideY + iz * strideZ;

    // The 16 vertices of the 4D unit hypercube (w,x,y,z) in {0,1}^4,
    // ordered to match the weight layout used everywhere else in the code.
    _projection[i].first[0]  = base;            
    _projection[i].first[1]  = base                               + strideZ;
    _projection[i].first[2]  = base                     + strideY;
    _projection[i].first[3]  = base                     + strideY + strideZ;
    _projection[i].first[4]  = base           + strideX;
    _projection[i].first[5]  = base           + strideX           + strideZ;
    _projection[i].first[6]  = base           + strideX + strideY;
    _projection[i].first[7]  = base           + strideX + strideY + strideZ;
    _projection[i].first[8]  = base + strideW;
    _projection[i].first[9]  = base + strideW                     + strideZ;
    _projection[i].first[10] = base + strideW           + strideY;
    _projection[i].first[11] = base + strideW           + strideY + strideZ;
    _projection[i].first[12] = base + strideW + strideX;
    _projection[i].first[13] = base + strideW + strideX           + strideZ;
    _projection[i].first[14] = base + strideW + strideX + strideY;
    _projection[i].first[15] = base + strideW + strideX + strideY + strideZ;

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

template <typename Tf>
void Grid4<Tf>::createMaps() {
  std::cout << "Find neighbours and projections...\n";

  _neighbour = new Neighbours[_nPoints];
  _projection = new std::pair<int[16], Tf[16]>[_nPoints];

  for (int i = 0; i < _nPoints; ++i) {  //for each quaternion..
    std::cout << '\r' << i << '/' << _nPoints << std::flush;

    //current quaternion
    Quaternion q = _volume[i];

    //array of neighbours positions in the grid
    Quaternion<Tf> n[32];
    n[0]  = {q.w - _h, q.x     , q.y     , q.z     };
    n[1]  = {q.w + _h, q.x     , q.y     , q.z     };
    n[2]  = {q.w     , q.x - _h, q.y     , q.z     };
    n[3]  = {q.w     , q.x + _h, q.y     , q.z     };
    n[4]  = {q.w     , q.x     , q.y - _h, q.z     };
    n[5]  = {q.w     , q.x     , q.y + _h, q.z     };
    n[6]  = {q.w     , q.x     , q.y     , q.z - _h};
    n[7]  = {q.w     , q.x     , q.y     , q.z + _h};
    n[8]  = {q.w - _h, q.x - _h, q.y     , q.z     };
    n[9]  = {q.w - _h, q.x + _h, q.y     , q.z     };
    n[10] = {q.w - _h, q.x     , q.y - _h, q.z     };
    n[11] = {q.w - _h, q.x     , q.y + _h, q.z     };
    n[12] = {q.w - _h, q.x     , q.y     , q.z - _h};
    n[13] = {q.w - _h, q.x     , q.y     , q.z + _h};
    n[14] = {q.w + _h, q.x - _h, q.y     , q.z     };
    n[15] = {q.w + _h, q.x + _h, q.y     , q.z     };
    n[16] = {q.w + _h, q.x     , q.y - _h, q.z     };
    n[17] = {q.w + _h, q.x     , q.y + _h, q.z     };
    n[18] = {q.w + _h, q.x     , q.y     , q.z - _h};
    n[19] = {q.w + _h, q.x     , q.y     , q.z + _h};
    n[20] = {q.w     , q.x - _h, q.y - _h, q.z     };
    n[21] = {q.w     , q.x - _h, q.y + _h, q.z     };
    n[22] = {q.w     , q.x - _h, q.y     , q.z - _h};
    n[23] = {q.w     , q.x - _h, q.y     , q.z + _h};
    n[24] = {q.w     , q.x + _h, q.y - _h, q.z     };
    n[25] = {q.w     , q.x + _h, q.y + _h, q.z     };
    n[26] = {q.w     , q.x + _h, q.y     , q.z - _h};
    n[27] = {q.w     , q.x + _h, q.y     , q.z + _h};
    n[28] = {q.w     , q.x     , q.y - _h, q.z - _h};
    n[29] = {q.w     , q.x     , q.y - _h, q.z + _h};
    n[30] = {q.w     , q.x     , q.y + _h, q.z - _h};
    n[31] = {q.w     , q.x     , q.y + _h, q.z + _h};

    //add current quat. index in _neighbour
    _neighbour[i].point[0] = i;

    //find the closest point on the hypersphere
    Quaternion c = closest(q);

    Tf w0 = std::floor((c.w - _wMin) / _h) * _h;
    Tf x0 = std::floor((c.x - _xMin) / _h) * _h;
    Tf y0 = std::floor((c.y - _yMin) / _h) * _h;
    Tf z0 = std::floor((c.z - _zMin) / _h) * _h;
    //neg-left-front-down vertex projection quaternion coordinates
    Quaternion nlfd = {w0, x0, y0, z0};
    //pos-right-behind-top vertex projection quaternion coordinates
    Quaternion prbt = {w0 + _h, x0 + _h, y0 + _h, z0 + _h};

    //weights
    Tf wd = (c.w - w0) / _h;
    Tf xd = (c.x - x0) / _h;
    Tf yd = (c.y - y0) / _h;
    Tf zd = (c.z - z0) / _h;
    if (wd > 1. || wd < 0.) {
      throw std::runtime_error("\nError: weight wd out of range [0, 1]: " + std::to_string(wd) + '\n');
    }
    if (xd > 1. || xd < 0.) {
      throw std::runtime_error("\nError: weight xd out of range [0, 1]: " + std::to_string(xd) + '\n');
    }
    if (yd > 1. || yd < 0.) {
      throw std::runtime_error("\nError: weight yd out of range [0, 1]: " + std::to_string(yd) + '\n');
    }
    if (zd > 1. || zd < 0.) {
      throw std::runtime_error("\nError: weight zd out of range [0, 1]: " + std::to_string(zd) + '\n');
    }

    //weights
    {
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

    //search for neighbours and projection quaternion indexes on the nlfd
    //and prbt vertices
    std::for_each(std::execution::par_unseq, _indices.begin(), _indices.end(),
                  [&](int idx) {
      Quaternion<Tf> const& r = _volume[idx];

      //neighbours
           if (r == n[0] ) _neighbour[i].point[1]  = idx;
      else if (r == n[1] ) _neighbour[i].point[2]  = idx;
      else if (r == n[2] ) _neighbour[i].point[3]  = idx;
      else if (r == n[3] ) _neighbour[i].point[4]  = idx;
      else if (r == n[4] ) _neighbour[i].point[5]  = idx;
      else if (r == n[5] ) _neighbour[i].point[6]  = idx;
      else if (r == n[6] ) _neighbour[i].point[7]  = idx;
      else if (r == n[7] ) _neighbour[i].point[8]  = idx;
      else if (r == n[8] ) _neighbour[i].point[9]  = idx;
      else if (r == n[9] ) _neighbour[i].point[10] = idx;
      else if (r == n[10]) _neighbour[i].point[11] = idx;
      else if (r == n[11]) _neighbour[i].point[12] = idx;
      else if (r == n[12]) _neighbour[i].point[13] = idx;
      else if (r == n[13]) _neighbour[i].point[14] = idx;
      else if (r == n[14]) _neighbour[i].point[15] = idx;
      else if (r == n[15]) _neighbour[i].point[16] = idx;
      else if (r == n[16]) _neighbour[i].point[17] = idx;
      else if (r == n[17]) _neighbour[i].point[18] = idx;
      else if (r == n[18]) _neighbour[i].point[19] = idx;
      else if (r == n[19]) _neighbour[i].point[20] = idx;
      else if (r == n[20]) _neighbour[i].point[21] = idx;
      else if (r == n[21]) _neighbour[i].point[22] = idx;
      else if (r == n[22]) _neighbour[i].point[23] = idx;
      else if (r == n[23]) _neighbour[i].point[24] = idx;
      else if (r == n[24]) _neighbour[i].point[25] = idx;
      else if (r == n[25]) _neighbour[i].point[26] = idx;
      else if (r == n[26]) _neighbour[i].point[27] = idx;
      else if (r == n[27]) _neighbour[i].point[28] = idx;
      else if (r == n[28]) _neighbour[i].point[29] = idx;
      else if (r == n[29]) _neighbour[i].point[30] = idx;
      else if (r == n[30]) _neighbour[i].point[31] = idx;
      else if (r == n[31]) _neighbour[i].point[32] = idx;

      //nlfd and prbt projections
           if (r == nlfd) _projection[i].first[0]  = idx;
      else if (r == prbt) _projection[i].first[15] = idx;
    });
  }

  std::cout << '\n';

  //for each quaternion find the projections via nlfd and prbt vertices
  std::for_each(std::execution::par_unseq, _indices.begin(), _indices.end(),
                [&](int idx) {
    std::cout << '\r' << idx << '/' << _nPoints << std::flush;

    Neighbours const& neigh_nlfd = _neighbour[_projection[idx].first[0]];
    Neighbours const& neigh_prbt = _neighbour[_projection[idx].first[15]];

    //the other 14 vertices of projection
    _projection[idx].first[1]  = neigh_nlfd.point[8];
    _projection[idx].first[2]  = neigh_nlfd.point[6];
    _projection[idx].first[3]  = neigh_nlfd.point[32];
    _projection[idx].first[4]  = neigh_nlfd.point[4];
    _projection[idx].first[5]  = neigh_nlfd.point[28];
    _projection[idx].first[6]  = neigh_nlfd.point[26];
    _projection[idx].first[7]  = neigh_prbt.point[1];
    _projection[idx].first[8]  = neigh_nlfd.point[2];
    _projection[idx].first[9]  = neigh_nlfd.point[20];
    _projection[idx].first[10] = neigh_nlfd.point[18];
    _projection[idx].first[11] = neigh_prbt.point[3];
    _projection[idx].first[12] = neigh_nlfd.point[16];
    _projection[idx].first[13] = neigh_prbt.point[5];
    _projection[idx].first[14] = neigh_prbt.point[7];
  });

  std::cout << "\tDone!\n";
}

template <typename Tf>
void Grid4<Tf>::createRho() {
  if (_rho.f() != nullptr) {
    std::cerr << "Warning: rho already created. Deleting it before recreating\n";
    _rho.del();
  }

  _rho = Function<Tf>(_nPoints);
}

template <typename Tf>
Tf Grid4<Tf>::phi(Tf w, Tf x, Tf y, Tf z) {
  Quaternion<Tf>& c = Params<Tf>::get().center;
  return std::sqrt(
      (w-c.w)*(w-c.w) + (x-c.x)*(x-c.x) + (y-c.y)*(y-c.y) + (z-c.z)*(z-c.z)
    ) - Params<Tf>::get().radius;
}

template <typename Tf>
Quaternion<Tf> Grid4<Tf>::closest(Quaternion<Tf> const& p) {
  Tf w = p.w;
  Tf x = p.x;
  Tf y = p.y;
  Tf z = p.z;

  Tf gradient[4];   //nabla_h (phi)
  gradient[0] = (phi(w+_h, x,    y,    z   ) - phi(w-_h, x,    y,    z   )) /(2*_h);
  gradient[1] = (phi(w,    x+_h, y   , z   ) - phi(w,    x-_h, y,    z   )) /(2*_h);
  gradient[2] = (phi(w,    x,    y+_h, z   ) - phi(w,    x,    y-_h, z   )) /(2*_h);
  gradient[3] = (phi(w,    x,    y,    z+_h) - phi(w,    x,    y,    z-_h)) /(2*_h);

  //if gradient = 0, the point is already on the surface
  if (gradient[0] == 0 && gradient[1] == 0 && gradient[2] == 0 && gradient[3] == 0) {
    return p;
  }

  // | nabla_h (phi) |
  Tf mod = std::sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1] +
    gradient[2]*gradient[2] + gradient[3]*gradient[3]);

  Tf dist_wxyz = phi(w,x,y,z);

  Quaternion<Tf> r;
  r.w = w - gradient[0] * dist_wxyz / mod;
  r.x = x - gradient[1] * dist_wxyz / mod;
  r.y = y - gradient[2] * dist_wxyz / mod;
  r.z = z - gradient[3] * dist_wxyz / mod;

  return r;
}

template <typename Tf>
bool Grid4<Tf>::loadSurface(std::string filename) {
  std::cout << "Loading surface... " << std::flush;
  std::fstream fin(filename + "_s.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error: surface file '"<< filename
              << "' couldn't be opened.\n";
    return false;
  }

  fin >> _nPoints;
  fin >> _h;

  Quaternion<Tf>* temp = new Quaternion<Tf>[_nPoints];
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

template <typename Tf>
bool Grid4<Tf>::loadProjection(std::string filename) {
  std::cout << "Loading projection... " << std::flush;
  std::fstream fin(filename + "_p.dat", std::ios::in);
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

  _projection = new std::pair<int[16], Tf[16]>[_nPoints];
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

template <typename Tf>
bool Grid4<Tf>::loadNeighbour(std::string filename) {
  std::cout << "Loading neighbour... " << std::flush;
  std::fstream fin(filename + "_n.dat", std::ios::in);
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

template <typename Tf>
bool Grid4<Tf>::loadRho(std::string filename) {
  std::cout << "Loading rho... " << std::flush;
  std::fstream fin(filename + "_r.dat", std::ios::in);
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

  _rho = Function<Tf>(_nPoints);
  for (int i = 0; i < _nPoints; ++i) {
    fin >> _rho[i];
  }

  fin.close();
  std::cout << "Done!\n";
  return true;
}

template <typename Tf>
Tf Grid4<Tf>::der1(int pointIndex, int direction) const {
  return der1(pointIndex, direction, _rho);
}

template <typename Tf>
Tf Grid4<Tf>::der2(int pointIndex, int direction, Quaternion<Tf> const& pos) const {
  return der2(pointIndex, direction, _rho, pos);
}

template <typename Tf>
Tf Grid4<Tf>::derij(int pointIndex, int dir1, int dir2, Quaternion<Tf> const& pos) const {
  return derij(pointIndex, dir1, dir2, _rho, pos);
}

template <typename Tf>
Tf Grid4<Tf>::der1(int pointIndex, int direction, Function<Tf> const& f) const {
  Neighbours& near = _neighbour[pointIndex];

  Tf result = 0.;

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

template <typename Tf>
Tf Grid4<Tf>::der2(int pointIndex, int direction, Function<Tf> const& f, Quaternion<Tf> const& pos) const {
  Neighbours& near = _neighbour[pointIndex];

  Tf result = 0.;
  int const exp = (pos == Quaternion<Tf>{-1., -1., -1., -1.}) ? 0 : 1;

  switch (direction + 1)
  { // no problem about D+/-(1/2) because D constant over direction
  case 1: //ww
    result = std::pow(pos.D(0),exp)*(f[near.point[2]] - f[near.point[0]]) +
      std::pow(pos.D(0),exp)*(f[near.point[1]] - f[near.point[0]]);
    break;
  case 2: //xx
    result = std::pow(pos.D(1),exp)*(f[near.point[4]] - f[near.point[0]]) +
      std::pow(pos.D(1),exp)*(f[near.point[3]] - f[near.point[0]]);
    break;
  case 3: //yy
    result = std::pow(pos.D(2),exp)*(f[near.point[6]] - f[near.point[0]]) +
      std::pow(pos.D(2),exp)*(f[near.point[5]] - f[near.point[0]]);
    break;
  case 4: //zz
    result = std::pow(pos.D(3),exp)*(f[near.point[8]] - f[near.point[0]]) +
      std::pow(pos.D(3),exp)*(f[near.point[7]] - f[near.point[0]]);
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in der2 function: " + direction);
    break;
  }

  return result / (_h * _h);
}

template <typename Tf>
Tf Grid4<Tf>::derij(int pointIndex, int dir1, int dir2, Function<Tf> const& f, Quaternion<Tf> const& pos) const {
  Neighbours near = _neighbour[pointIndex];

  Tf result = 0.;
  int exp = (pos == Quaternion<Tf>{-1., -1., -1., -1.}) ? 0 : 1;

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

template class Grid4<double>;
template class Grid4<float>;
