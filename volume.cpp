#include "volume.h"
#include <limits>
#include <iostream>
#include <iomanip>

Volume::Volume(int nPoints, Quaternion data) :
                _nPoints{nPoints},
                _data{new Quaternion[nPoints]} {
  std::fill(_data, _data + nPoints, data);
}

Volume::Volume(int nPoints, Quaternion *data) :
                _nPoints{nPoints},
                _data{new Quaternion[nPoints]} {
  std::copy(data, data + nPoints, _data);
}

Volume::Volume(const Volume &src) :
                _nPoints{src._nPoints},
                _data{new Quaternion[src._nPoints]} {
  
  std::copy(src._data, src._data + src._nPoints, _data);

}

Volume::Volume(Volume &&src) : 
              _nPoints{src._nPoints},
              _data{src._data} {
  src._nPoints = 0;
  src._data = nullptr;
}

Volume &Volume::operator=(const Volume &src) {
  // Guard self assignment
  if (this == &src)
    return *this;

  if (_nPoints != src._nPoints) {           // resource in *this cannot be reused
    Quaternion* temp = new Quaternion[src._nPoints];  // allocate resource, if throws, do nothing
    delete[] _data;                         // release resource in *this
    _data = temp;
    _nPoints = src._nPoints;
  } 

  std::copy(src._data, src._data + src._nPoints, _data);
  return *this;
}

Volume &Volume::operator=(Volume &&src) {
  // Guard self assignment
  if (this == &src)
    return *this;

  delete[] _data;       // release resource in *this
  _data = src._data;
  _nPoints = src._nPoints;
  src._data = nullptr;  // leave src in valid state
  src._nPoints = 0;

  return *this;
}

Volume::~Volume() {
  delete[] _data;
}

std::ostream &operator<<(std::ostream &os, const Volume &obj)
{
  //format: 4 cols of value: w x y z
  for (int i = 0; i < obj._nPoints; ++i) {
    os << obj._data[i].w << ' ' << obj._data[i].x << ' ' << obj._data[i].y << ' ' << obj._data[i].z << '\n';
  }
  return os;
}