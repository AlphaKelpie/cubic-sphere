#include "grid3.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

Grid3::Grid3(Interval x, Interval y, Interval z)
	: _nPoints{0},
		_projection{nullptr},
		_neighbour{nullptr},
		_h{0.1},
		_delta{1.1 * sqrt(3) * _h}
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
				temp[_nPoints] = {i, j, k};
				//..create corresponding neighbour
				_neighbour[_nPoints].point[0] = _nPoints;   // itself
				_neighbour[_nPoints].point[1] = std::max(
          abs(_nPoints - yPoints * zPoints) * left, -1);
				_neighbour[_nPoints].point[2] = std::max(
          (_nPoints + yPoints * zPoints) * right, -1);
				_neighbour[_nPoints].point[3] = std::max(
          abs(_nPoints - zPoints) * front, -1);
				_neighbour[_nPoints].point[4] = std::max(
          (_nPoints + zPoints) * behind, -1);
				_neighbour[_nPoints].point[5] = std::max(abs(_nPoints - 1) * down, -1);
				_neighbour[_nPoints].point[6] = std::max((_nPoints + 1) * top, -1);

				_neighbour[_nPoints].point[7] = std::max(
					abs(_nPoints - (yPoints + 1) * zPoints) * sgn(left, front), -1);
				_neighbour[_nPoints].point[8] = std::max(
					abs(_nPoints - (yPoints - 1) * zPoints) * sgn(left, behind), -1);
				_neighbour[_nPoints].point[9] = std::max(
					abs(_nPoints - (yPoints * zPoints + 1)) * sgn(left, down), -1);
				_neighbour[_nPoints].point[10] = std::max(
					abs(_nPoints - (yPoints * zPoints - 1)) * sgn(left, top), -1);
				_neighbour[_nPoints].point[11] = std::max(
					abs(_nPoints - (zPoints + 1)) * sgn(front, down), -1);
				_neighbour[_nPoints].point[12] = std::max(
					abs(_nPoints - (zPoints - 1)) * sgn(front, top), -1);
				_neighbour[_nPoints].point[13] = std::max(
					(_nPoints + (zPoints - 1)) * sgn(behind, down), -1);
				_neighbour[_nPoints].point[14] = std::max(
					(_nPoints + (zPoints + 1)) * sgn(behind, top), -1);
				_neighbour[_nPoints].point[15] = std::max(
					(_nPoints + (yPoints - 1) * zPoints) * sgn(right, front), -1);
				_neighbour[_nPoints].point[16] = std::max(
					(_nPoints + (yPoints + 1) * zPoints) * sgn(right, behind), -1);
				_neighbour[_nPoints].point[17] = std::max(
					(_nPoints + (yPoints * zPoints - 1)) * sgn(right, down), -1);
				_neighbour[_nPoints].point[18] = std::max(
					(_nPoints + (yPoints * zPoints + 1)) * sgn(right, top), -1);
				++_nPoints;
			}
		}
	}
	_volume = Surface(_nPoints, temp);
	// _sphere = Surface(spoints, temp_s);
	delete[] temp;
	// delete[] temp_s;

	createProjection();
	createRho();
}

Grid3::Grid3(std::string mapFile)
	: _nPoints{0},
		_projection{nullptr},
		_volume{Surface()}
	//   _sphere{Surface()}
{
	if (!loadMap(mapFile)) {
		throw std::runtime_error{"loadMap() cannot open file."};
	}
	createRho();
}

Grid3::~Grid3() {
	delete[] _projection;
	delete[] _neighbour;
}

void Grid3::saveRelations(std::string filename) {
	//format: 1 row _h and _nPoints
	// 1 row per point to _projection: index and weight for the 8 grid
	// points around
	// 1 row per point to _neighbour indexes: itself top down behind
	// front right left

	std::cout << "Saving h, nPoints, projection and neighbour...     ";
	std::fstream fout(filename, std::ios::out);

	fout << _h << ' ' << _nPoints << "\n\n";

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
  std::cout << "Saving surface...     ";
  if (_volume.nPoints() == 0) {
    std::cerr << "Error, cannot save surface of a simulation created from a map.\n";
    return;
  }

  std::fstream fout(filename, std::ios::out);
  fout << _volume;
  fout.close();

  std::cout << "Done!\n";
}

void Grid3::saveRho(std::string filename) {
  std::fstream fout(filename, std::ios::out);
  for (int i = 0; i < _nPoints; ++i)
    fout << _rho[i] << '\n';

  fout.close();
}

void Grid3::evolve(double dt = 0) {

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
  std::cout << "Constructing map...     ";
  _projection = new std::pair<int[8], double[8]>[_nPoints];
  for (int i = 0; i < _nPoints; ++i) {
    Point c = closest(_volume[i]);
    
    //(A)  find the indexes of the 8 surface points surrounding the closest point of point[i]:
    //(A1) find the points (actually, only the left-front-down)
    double x0 = (int)(c.x/_h) *_h;
    double y0 = (int)(c.y/_h) *_h;
    double z0 = (int)(c.z/_h) *_h;

      //(A2) find their indexes in _volume
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

    //(B) find the weights of the interpolation
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

bool Grid3::loadMap(std::string filename) {
  std::cout << "Loading map...     ";
  std::fstream fin(filename, std::ios::in);
  if (!fin.is_open()) {
    std::cerr << "Error. Projection file '"<< filename << "' couldn't be opened.\n";
    return false;
  }

  fin >> _h;
  fin >> _nPoints;

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

  std::cout << "Done.\n";
  return true;
}

double Grid3::der1(int pointIndex, char direction) {
  Neighbours& near = _neighbour[pointIndex];

  double result = 0.;

  switch (direction)
  {
  case 'x':
    result = (_rho[near.point[2]] - _rho[near.point[1]]);
    break;
  case 'y':
    result = (_rho[near.point[4]] - _rho[near.point[3]]);
    break;
  case 'z':
    result = (_rho[near.point[6]] - _rho[near.point[5]]);
    break;
  default:
    throw std::invalid_argument(
      "Error: invalid direction in der1 function: " + direction);
    break;
  }

  return result / (_h * 2);

}

double Grid3::der2(int pointIndex, char direction) {
  Neighbours& near = _neighbour[pointIndex];

  double result = 0.;

  switch (direction)
  {
  case 'x':
    result = _rho[near.point[2]] + _rho[near.point[1]];
    break;
  case 'y':
    result = _rho[near.point[4]] + _rho[near.point[3]];
    break;
  case 'z':
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

double Grid3::derij(int pointIndex, std::string directions) {
  Neighbours near = _neighbour[pointIndex];

  double result = 0.;

  if (directions.size() != 2) {
    throw std::invalid_argument(
      "Error: derij directions must be size 2: " + directions);
  }
  if (directions.find('x') != std::string::npos) {
    if (directions.find('y') != std::string::npos) {
      // xy || yx
      result = _rho[near.point[16]] - _rho[near.point[15]] -
               _rho[near.point[8]] + _rho[near.point[7]];
    } else if (directions.find('z') != std::string::npos) {
      // xz || zx
      result = _rho[near.point[18]] - _rho[near.point[17]] -
               _rho[near.point[10]] + _rho[near.point[9]];
    } else {
      throw std::invalid_argument(
        "Error: derij directions has 'x' and an invalid character: " +
        directions);
    }
  }
  else if (directions.find('y') != std::string::npos) {
    if (directions.find('z') != std::string::npos) {
      // yz || zy
      result = _rho[near.point[14]] - _rho[near.point[13]] -
               _rho[near.point[12]] + _rho[near.point[11]];
    } else {
      throw std::invalid_argument(
        "Error: derij directions has 'y', not 'x' and an invalid character: " +
        directions);
    }
  } else {
    throw std::invalid_argument(
      "Error: derij directions has neither 'x' or 'y': " + directions);
  }

  return result / (_h * _h * 4);
}

int Grid3::sgn(int val1, int val2) {
  if (val1 < 0 || val2 < 0) { return -1; }
  return 1;
}
