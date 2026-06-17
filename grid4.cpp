#include "grid4.h"
#include "params.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

Grid4::Grid4(Interval w, Interval x, Interval y, Interval z, float T, float h)
	: _nPoints{0},
		_projection{nullptr},
		_neighbour{nullptr},
    _t{T/4},
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
  int neg;
  int pos;
	int left;
	int right;
	int front;
	int behind;
	int down;
	int top;
  int index = 0;
  for (float m = w.min; m < w.max + 1E-3; m+=_h) {
    neg = (m <= w.min) ? -1 : 1;
    pos = (m >= w.max - _h) ? -1 : 1;
    for (float i = x.min; i < x.max + 1E-3; i+=_h) {
      left = (i <= x.min) ? -1 : 1;
      right = (i >= x.max - _h) ? -1 : 1;
      for (float j = y.min; j < y.max + 1E-3; j+=_h) {
        front = (j <= y.min) ? -1 : 1;
        behind = (j >= y.max - _h) ? -1 : 1;
        for (float k = z.min; k < z.max + 1E-3; k+=_h) {
          std::cout << '\r' << index << '/' << _nPoints << std::flush;
          down = (k <= z.min) ? -1 : 1;
          top = (k >= z.max - _h) ? -1 : 1;
          //..add it to the volume and..
          temp[index] = {m, i, j, k};
          //..create corresponding neighbour
          _neighbour[index].point[0] = index;   // itself
          _neighbour[index].point[1] = std::max(
            abs(index - _xPoints * _yPoints * _zPoints) * neg, -1);

          _neighbour[index].point[2] = std::max(
            abs(index + _xPoints * _yPoints * _zPoints) * pos, -1);

          _neighbour[index].point[3] = std::max(
            abs(index - _yPoints * _zPoints) * left, -1);

          _neighbour[index].point[4] = std::max(
            abs(index + _yPoints * _zPoints) * right, -1);

          _neighbour[index].point[5] = std::max(
            abs(index - _zPoints) * front, -1);

          _neighbour[index].point[6] = std::max(
            abs(index + _zPoints) * behind, -1);

          _neighbour[index].point[7] = std::max(
            abs(index - 1) * down, -1);

          _neighbour[index].point[8] = std::max(
            abs(index + 1) * top, -1);

          _neighbour[index].point[9] = std::max(
            abs(index - (_xPoints + 1) * _yPoints * _zPoints) * sgn(neg, left), -1);

          _neighbour[index].point[10] = std::max(
            abs(index - (_xPoints - 1) * _yPoints * _zPoints) * sgn(neg, right), -1);

          _neighbour[index].point[11] = std::max(
            abs(index - (_xPoints * _yPoints + 1) * _zPoints) * sgn(neg, front), -1);

          _neighbour[index].point[12] = std::max(
            abs(index - (_xPoints * _yPoints - 1) * _zPoints) * sgn(neg, behind), -1);

          _neighbour[index].point[13] = std::max(
            abs(index - (_xPoints * _yPoints * _zPoints + 1)) * sgn(neg, down), -1);

          _neighbour[index].point[14] = std::max(
            abs(index - (_xPoints * _yPoints * _zPoints - 1)) * sgn(neg, top), -1);

          _neighbour[index].point[15] = std::max(
            abs(index + (_xPoints - 1) * _yPoints * _zPoints) * sgn(pos, left), -1);

          _neighbour[index].point[16] = std::max(
            abs(index + (_xPoints + 1) * _yPoints * _zPoints) * sgn(pos, right), -1);

          _neighbour[index].point[17] = std::max(
            abs(index + (_xPoints * _yPoints - 1) * _zPoints) * sgn(pos, front), -1);

          _neighbour[index].point[18] = std::max(
            abs(index + (_xPoints * _yPoints + 1) * _zPoints) * sgn(pos, behind), -1);

          _neighbour[index].point[19] = std::max(
            abs(index + (_xPoints * _yPoints * _zPoints - 1)) * sgn(pos, down), -1);

          _neighbour[index].point[20] = std::max(
            abs(index + (_xPoints * _yPoints * _zPoints + 1)) * sgn(pos, top), -1);

          _neighbour[index].point[21] = std::max(
            abs(index - (_yPoints + 1) * _zPoints) * sgn(left, front), -1);

          _neighbour[index].point[22] = std::max(
            abs(index - (_yPoints - 1) * _zPoints) * sgn(left, behind), -1);

          _neighbour[index].point[23] = std::max(
            abs(index - (_yPoints * _zPoints + 1)) * sgn(left, down), -1);

          _neighbour[index].point[24] = std::max(
            abs(index - (_yPoints * _zPoints - 1)) * sgn(left, top), -1);

          _neighbour[index].point[25] = std::max(
            abs(index + (_yPoints - 1) * _zPoints) * sgn(right, front), -1);

          _neighbour[index].point[26] = std::max(
            abs(index + (_yPoints + 1) * _zPoints) * sgn(right, behind), -1);

          _neighbour[index].point[27] = std::max(
            abs(index + (_yPoints * _zPoints - 1)) * sgn(right, down), -1);

          _neighbour[index].point[28] = std::max(
            abs(index + (_yPoints * _zPoints + 1)) * sgn(right, top), -1);

          _neighbour[index].point[29] = std::max(
            abs(index - (_zPoints + 1)) * sgn(front, down), -1);

          _neighbour[index].point[30] = std::max(
            abs(index - (_zPoints - 1)) * sgn(front, top), -1);

          _neighbour[index].point[31] = std::max(
            abs(index + (_zPoints - 1)) * sgn(behind, down), -1);

          _neighbour[index].point[32] = std::max(
            abs(index + (_zPoints + 1)) * sgn(behind, top), -1);

          ++index;
        }
      }
    }
  }
  if (index != _nPoints) {
    throw std::runtime_error("Error: _neighbour index different than _nPoints");
  }

	_volume = Volume(_nPoints, temp);
	delete[] temp;
  std::cout << "\tDone!\n";

	createProjection();
	createRho();
}

Grid4::Grid4(std::string surfFile, std::string relFile, float T)
	: _nPoints{0},
    _t{T/4},
    _wPoints{0}, _xPoints{0}, _yPoints{0}, _zPoints{0},
    _wMin{0.f}, _xMin{0.f}, _yMin{0.f}, _zMin{0.f}
{
  if (!loadSurface(surfFile)) {
		throw std::runtime_error{"loadSurface() cannot open file."};
  }
	if (!loadRelations(relFile)) {
		throw std::runtime_error{"loadRelation() cannot open file."};
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

void Grid4::saveRelations(std::string filename) {
	// format:
  // first row: _nPoints
	// 1 row per point to _projection: index and weight for the 8 grid
	// points around
	// 1 row per point to _neighbour indexes (19 int)

	std::cout << "Saving nPoints, projection and neighbour... " << std::flush;
	std::fstream fout(filename + "pn.dat", std::ios::out);

	fout << _nPoints << "\n\n";

	for (int i = 0; i < _nPoints; ++i) {
		for (int j = 0; j < 16; ++j) {
			fout << _projection[i].first[j] << ' ';
			fout << _projection[i].second[j] << ' ';
		}
		fout << "\n";
	}

	fout << "\n";

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

void Grid4::evolve(float dt) {
  if (dt == 0) {
    dt = 0.1 * _h * _h * _h * _h;
  }

  Function nextRho = Function(_nPoints);

  float deltaRho = 0.;
  for (int idx = 0; idx < _nPoints; ++idx) {
    Quaternion const& p = _volume[idx];

    for (int j = 0; j < 4; ++j) {
      deltaRho = p[j] * (1 - 4) * der1(idx, j);
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
  }

  _rho = nextRho;
}

void Grid4::project() {
  Function sphere_rho(_nPoints);

  for (int i = 0; i < _nPoints; ++i) {
    float rho = _rho[i];
    if (abs(rho) <= 1E-14) { continue; }
    std::pair<int[16], float[16]>& cubic = _projection[i];
    for (int j = 0; j < 16; ++j) {
      sphere_rho[cubic.first[j]] += rho * cubic.second[j];
    }
  }

  _rho = sphere_rho;
}


// private


float Grid4::phi(float w, float x, float y, float z) {
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

  fillFirstRho();
}

void Grid4::fillFirstRho() {
  std::random_device seed;
  // std::cout << "Debug: seed = " << seed() << '\n';
  std::mt19937 gen(seed());
  std::uniform_int_distribution<int> init(0, _nPoints);
  int start_point = init(gen);

  std::pair<int[16], float[16]>& cubic = _projection[start_point];
  for (int j = 0; j < 16; ++j) {
    if (cubic.first[j] >= 0)
    _rho[cubic.first[j]] = 1. * cubic.second[j];
  }
}

void Grid4::createProjection() {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::cout << "Constructing projection...\n";
  _projection = new std::pair<int[16], float[16]>[_nPoints];

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
    float fw = (c.w - _wMin) / _h;
    float fx = (c.x - _xMin) / _h;
    float fy = (c.y - _yMin) / _h;
    float fz = (c.z - _zMin) / _h;

    // Lower corner indices (clamp to valid range)
    int iw = std::clamp((int)std::floor(fw), 0, _wPoints - 2);
    int ix = std::clamp((int)std::floor(fx), 0, _xPoints - 2);
    int iy = std::clamp((int)std::floor(fy), 0, _yPoints - 2);
    int iz = std::clamp((int)std::floor(fz), 0, _zPoints - 2);

    // Interpolation fractions (distance from lower corner, in [0,1])
    float wd = std::clamp(fw - (float)iw, 0.f, 1.f);
    float xd = std::clamp(fx - (float)ix, 0.f, 1.f);
    float yd = std::clamp(fy - (float)iy, 0.f, 1.f);
    float zd = std::clamp(fz - (float)iz, 0.f, 1.f);

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

    // Trilinear interpolation weights (identical formula to the original code)
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

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::minutes>(
    end_time - start_time);
  std::cout << "\tDone! (" << duration.count() << " minutes)\n";
}

Quaternion Grid4::closest(Quaternion const& p) {
  float w = p.w;
  float x = p.x;
  float y = p.y;
  float z = p.z;

  float gradient[4];   //nabla_h (phi)
  gradient[0] = (phi(w+_h, x,    y,    z   ) - phi(w-_h, x,    y,    z   )) /(2*_h);
  gradient[1] = (phi(w,    x+_h, y   , z   ) - phi(w,    x-_h, y,    z   )) /(2*_h);
  gradient[2] = (phi(w,    x,    y+_h, z   ) - phi(w,    x,    y-_h, z   )) /(2*_h);
  gradient[3] = (phi(w,    x,    y,    z+_h) - phi(w,    x,    y,    z-_h)) /(2*_h);

  //if gradient = 0, the point is already on the surface
  if (gradient[0] == 0 && gradient[1] == 0 && gradient[2] == 0 && gradient[3] == 0) {
    return p;
  }

  // | nabla_h (phi) |
  float mod = std::sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1] +
    gradient[2]*gradient[2] + gradient[3]*gradient[3]);

  float dist_wxyz = phi(w,x,y,z);

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

  _volume = Volume(_nPoints, temp);
  delete[] temp;

  fin.close();
  std::cout << "Done!\n";
  return true;
}

bool Grid4::loadRelations(std::string filename) {
  std::cout << "Loading relations... " << std::flush;
  std::fstream fin(filename + "pn.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error: relations file '"<< filename
              << "' couldn't be opened.\n";
    return false;
  }

  int nPoints;
  fin >> nPoints;
  if (_nPoints != nPoints) {
    std::cerr << "Error: relations file has uncorrect number of points\n";
    return false;
  }

  _projection = new std::pair<int[16], float[16]>[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    for (int j = 0; j < 16; ++j) {
      fin >> _projection[i].first[j];
      fin >> _projection[i].second[j];
    }
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

float Grid4::der1(int pointIndex, int direction) const {
  Neighbours& near = _neighbour[pointIndex];

  float result = 0.;

  switch (direction + 1)
  {
  case 1: //w
    result = (_rho[near.point[2]] - _rho[near.point[1]]);
    break;
  case 2: //x
    result = (_rho[near.point[4]] - _rho[near.point[3]]);
    break;
  case 3: //y
    result = (_rho[near.point[6]] - _rho[near.point[5]]);
    break;
  case 4: //z
    result = (_rho[near.point[8]] - _rho[near.point[7]]);
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in der1 function: " + direction);
    break;
  }

  return result / (_h * 2);

}

float Grid4::der2(int pointIndex, int direction) const {
  Neighbours& near = _neighbour[pointIndex];

  float result = 0.;

  switch (direction + 1)
  {
  case 1: //ww
    result = _rho[near.point[2]] + _rho[near.point[1]];
    break;
  case 2: //xx
    result = _rho[near.point[4]] + _rho[near.point[3]];
    break;
  case 3: //yy
    result = _rho[near.point[6]] + _rho[near.point[5]];
    break;
  case 4: //zz
    result = _rho[near.point[8]] + _rho[near.point[7]];
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in der2 function: " + direction);
    break;
  }

  result -= (_rho[near.point[0]] * 2);
  return result / (_h * _h);
}

float Grid4::derij(int pointIndex, int dir1, int dir2) const {
  Neighbours near = _neighbour[pointIndex];

  float result = 0.;

  switch ((dir1 + 1) * (dir2 + 1))
  {
  case 2: //wx || xw
    result = _rho[near.point[16]] - _rho[near.point[15]] -
      _rho[near.point[10]] + _rho[near.point[9]];
    break;
  case 3: //wy || yw
    result = _rho[near.point[18]] - _rho[near.point[17]] -
      _rho[near.point[12]] + _rho[near.point[11]];
    break;
  case 4: //wz || zw
    result = _rho[near.point[20]] - _rho[near.point[19]] -
      _rho[near.point[14]] + _rho[near.point[13]];
    break;
  case 6: //xy || yx
    result = _rho[near.point[26]] - _rho[near.point[25]] -
      _rho[near.point[22]] + _rho[near.point[21]];
    break;
  case 8: //xz || zx
    result = _rho[near.point[28]] - _rho[near.point[27]] -
      _rho[near.point[24]] + _rho[near.point[23]];
    break;
  case 12: //yz || zy
    result = _rho[near.point[32]] - _rho[near.point[31]] -
      _rho[near.point[30]] + _rho[near.point[29]];
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in derij function: " + dir1 + ' ' + dir2);
    break;
  }

  return result / (_h * _h * 4);
}

int Grid4::sgn(int val1, int val2) {
  if (val1 < 0 || val2 < 0) { return -1; }
  return 1;
}
