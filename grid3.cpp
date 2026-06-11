#include "grid3.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

Grid3::Grid3(Interval x, Interval y, Interval z, double T, double h)
	: _nPoints{0},
		_projection{nullptr},
		_neighbour{nullptr},
    _t{T/4},
		_h{h},
		_delta{1.1 * sqrt(3) * h}
{
	int xPoints = (x.max - x.min) / _h;
	int yPoints = (y.max - y.min) / _h;
	int zPoints = (z.max - z.min) / _h;
	_nPoints = xPoints * yPoints * zPoints;

	// int spoints = 0;
	// Create surface and get _nPoints
	Point* temp = new Point[_nPoints];
	// Point* temp_s = new Point[_nPoints];
	_neighbour = new Neighbours[_nPoints];

	//for every point of the descrete domain..
	int down;
	int top;
	int behind;
	int front;
	int right;
	int left;
  int index = 0;
	for (double i = x.min; i < x.max; i+=_h) {
		left = (i <= x.min) ? -1 : 1;
		right = (i >= x.max - _h) ? -1 : 1;
		for (double j = y.min; j < y.max; j+=_h) {
			front = (j <= y.min) ? -1 : 1;
			behind = (j >= y.max - _h) ? -1 : 1;
			for (double k = z.min; k < z.max; k+=_h) {
				down = (k <= z.min) ? -1 : 1;
				top = (k >= z.max - _h) ? -1 : 1;
				//..add it to the volume and..
				temp[index] = {i, j, k};
				//..create corresponding neighbour
				_neighbour[index].point[0] = index;   // itself
				_neighbour[index].point[1] = std::max(
          abs(index - yPoints * zPoints) * left, -1);
				_neighbour[index].point[2] = std::max(
          (index + yPoints * zPoints) * right, -1);
				_neighbour[index].point[3] = std::max(
          abs(index - zPoints) * front, -1);
				_neighbour[index].point[4] = std::max(
          (index + zPoints) * behind, -1);
				_neighbour[index].point[5] = std::max(abs(index - 1) * down, -1);
				_neighbour[index].point[6] = std::max((index + 1) * top, -1);

				_neighbour[index].point[7] = std::max(
					abs(index - (yPoints + 1) * zPoints) * sgn(left, front), -1);
				_neighbour[index].point[8] = std::max(
					abs(index - (yPoints - 1) * zPoints) * sgn(left, behind), -1);
				_neighbour[index].point[9] = std::max(
					abs(index - (yPoints * zPoints + 1)) * sgn(left, down), -1);
				_neighbour[index].point[10] = std::max(
					abs(index - (yPoints * zPoints - 1)) * sgn(left, top), -1);
				_neighbour[index].point[11] = std::max(
					abs(index - (zPoints + 1)) * sgn(front, down), -1);
				_neighbour[index].point[12] = std::max(
					abs(index - (zPoints - 1)) * sgn(front, top), -1);
				_neighbour[index].point[13] = std::max(
					(index + (zPoints - 1)) * sgn(behind, down), -1);
				_neighbour[index].point[14] = std::max(
					(index + (zPoints + 1)) * sgn(behind, top), -1);
				_neighbour[index].point[15] = std::max(
					(index + (yPoints - 1) * zPoints) * sgn(right, front), -1);
				_neighbour[index].point[16] = std::max(
					(index + (yPoints + 1) * zPoints) * sgn(right, behind), -1);
				_neighbour[index].point[17] = std::max(
					(index + (yPoints * zPoints - 1)) * sgn(right, down), -1);
				_neighbour[index].point[18] = std::max(
					(index + (yPoints * zPoints + 1)) * sgn(right, top), -1);
				++index;
			}
		}
	}
  if (index != _nPoints) {
    throw std::runtime_error("Error: _neighbour index different than _nPoints");
  }

	_volume = Surface(_nPoints, temp);
	// _sphere = Surface(spoints, temp_s);
	delete[] temp;
	// delete[] temp_s;

	createProjection();
	createRho();
}

Grid3::Grid3(std::string surfFile, std::string relFile, double T)
	: _nPoints{0},
    _t{T/4}
{
  if (!loadSurface(surfFile)) {
		throw std::runtime_error{"loadSurface() cannot open file."};
  }
	if (!loadRelations(relFile)) {
		throw std::runtime_error{"loadRelation() cannot open file."};
	}
	createRho();
}

Grid3::~Grid3() {
	delete[] _projection;
	delete[] _neighbour;
}

void Grid3::saveRelations(std::string filename) {
	// format:
  // first row: _nPoints
	// 1 row per point to _projection: index and weight for the 8 grid
	// points around
	// 1 row per point to _neighbour indexes (19 int)

	std::cout << "Saving nPoints, projection and neighbour...     ";
	std::fstream fout(filename + "_pn.dat", std::ios::out);

	fout << _nPoints << "\n\n";

	for (int i = 0; i < _nPoints; ++i) {
		for (int j = 0; j < 8; ++j) {
			fout << _projection[i].first[j] << ' ';
			fout << _projection[i].second[j] << ' ';
		}
		fout << "\n";
	}

	fout << "\n";

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

  std::cout << "Saving surface...     ";
  if (_volume.nPoints() == 0) {
    std::cerr << "Error, cannot save surface of a simulation created from a map.\n";
    return;
  }

  std::fstream fout(filename + "_s.dat", std::ios::out);

  fout << _nPoints << ' ' << _h << "\n\n";

  fout << _volume;

  fout.close();
  std::cout << "Done!\n";
}

void Grid3::saveRho(std::string filename) {
  // format:
  // first row: _nPoints
  // 1 row per _rho value

  std::cout << "Saving rho...     ";

  std::fstream fout(filename + "_r.dat", std::ios::out);

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
    if (rho == 0) { continue; }
    std::pair<int[8], double[8]>& cubic = _projection[i];
    for (int j = 0; j < 8; ++j) {
      sphere_rho[cubic.first[j]] += rho * cubic.second[j];
    }
  }

  _rho = sphere_rho;
}


// private


double Grid3::phi(double x, double y, double z) {
  //default function: a sphere centred in (5,5,5) with r=4.5
  return std::sqrt((x-5)*(x-5) + (y-5)*(y-5) + (z-5)*(z-5)) - 4.5;
}

void Grid3::createRho() {
  if (_rho.f() != nullptr) {
    std::cerr << "Warning: v already created. Deleting it before recreating\n";
    _rho.del();
  }

  _rho = Function(_nPoints);

  fillFirstRho();
}

void Grid3::fillFirstRho() {
  std::random_device seed;
  std::cout << "Debug: seed = " << seed() << '\n';
  std::mt19937 gen(seed());
  std::uniform_int_distribution<int> init(0, _nPoints);
  int start_point = init(gen);

  std::pair<int[8], double[8]>& cubic = _projection[start_point];
  for (int j = 0; j < 8; ++j) {
    _rho[cubic.first[j]] += 1. * cubic.second[j];
  }
}

void Grid3::createProjection() {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::cout << "Constructing projection...     ";
  _projection = new std::pair<int[8], double[8]>[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    Point c = closest(_volume[i]);
    
    //find the left-front-down grid point nearest to c
    double x0 = (int)(c.x/_h) *_h;
    double y0 = (int)(c.y/_h) *_h;
    double z0 = (int)(c.z/_h) *_h;

    //find the indexes of the cubic nearest points to c
      for (int k = 0; k < _nPoints; ++k) {
      Point s = _volume[k];
      
      if      (s == Point{x0,    y0,    z0   })   _projection[i].first[0] = k;
      else if (s == Point{x0,    y0,    z0+_h})   _projection[i].first[1] = k;
      else if (s == Point{x0,    y0+_h, z0   })   _projection[i].first[2] = k;
      else if (s == Point{x0,    y0+_h, z0+_h})   _projection[i].first[3] = k;
      else if (s == Point{x0+_h, y0,    z0   })   _projection[i].first[4] = k;
      else if (s == Point{x0+_h, y0,    z0+_h})   _projection[i].first[5] = k;
      else if (s == Point{x0+_h, y0+_h, z0   })   _projection[i].first[6] = k;
      else if (s == Point{x0+_h, y0+_h, z0+_h})   _projection[i].first[7] = k;
    }

    //find the weights of the interpolation
    double xd = (c.x - x0) /_h;
    double yd = (c.y - y0) /_h;
    double zd = (c.z - z0) /_h;

    _projection[i].second[0] = (1-xd) * (1-yd) * (1-zd);
    _projection[i].second[1] = (1-xd) * (1-yd) *   zd;
    _projection[i].second[2] = (1-xd) *   yd   * (1-zd);
    _projection[i].second[3] = (1-xd) *   yd   *   zd;
    _projection[i].second[4] =   xd   * (1-yd) * (1-zd);
    _projection[i].second[5] =   xd   * (1-yd) *   zd;
    _projection[i].second[6] =   xd   *   yd   * (1-zd);
    _projection[i].second[7] =   xd   *   yd   *   zd;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::minutes>(
    end_time - start_time);
  std::cout << "Done! (" << duration.count() << " minutes)\n";
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
  std::cout << "Loading surface...     ";
  std::fstream fin(filename + "_s.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error. Surface file '"<< filename << "' couldn't be opened.\n";
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
  std::cout << "Done.\n";
  return true;
}

bool Grid3::loadRelations(std::string filename) {
  std::cout << "Loading relations...     ";
  std::fstream fin(filename + "_pn.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error. Relations file '"<< filename << "' couldn't be opened.\n";
    return false;
  }

  int nPoints;
  fin >> nPoints;
  if (_nPoints != nPoints) {
    std::cerr << "Error: Relations file has different number of points of surface.\n";
    return false;
  }

  _projection = new std::pair<int[8], double[8]>[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    for (int j = 0; j < 8; ++j) {
      fin >> _projection[i].first[j];
      fin >> _projection[i].second[j];
    }
  }

  _neighbour = new Neighbours[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    for (int j = 0; j < 19; ++j) {
      fin >> _neighbour[i].point[j];
    }
  }

  fin.close();
  std::cout << "Done.\n";
  return true;
}

bool Grid3::loadRho(std::string filename) {
  std::cout << "Loading rho...     ";
  std::fstream fin(filename + "_r.dat", std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error. Rho file '"<< filename << "' couldn't be opened.\n";
    return false;
  }

  int nPoints;
  fin >> nPoints;
  if (_nPoints != nPoints) {
    std::cerr << "Error: Rho file has different number of points of surface.\n";
    return false;
  }

  _rho = Function(_nPoints);
  for (int i = 0; i < _nPoints; ++i) {
    fin >> _rho[i];
  }

  fin.close();
  std::cout << "Done.\n";
  return true;
}

double Grid3::der1(int pointIndex, int direction) {
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

double Grid3::der2(int pointIndex, int direction) {
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

double Grid3::derij(int pointIndex, int dir1, int dir2) {
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

int Grid3::sgn(int val1, int val2) {
  if (val1 < 0 || val2 < 0) { return -1; }
  return 1;
}
