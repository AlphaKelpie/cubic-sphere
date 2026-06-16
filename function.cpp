#include "function.h"

#include <algorithm>
#include <iostream>

Function::Function() : _nPoints{0}, _f{nullptr} {}

Function::Function(int np)
  : _nPoints{np},
    _f{new float[np]}
{
  std::fill(_f, _f + np, 0.);
}

Function::Function(const Function &src)
  : _nPoints{src._nPoints},
    _f{new float[_nPoints]}
{
  std::copy(src._f, src._f + src._nPoints, _f);
}

Function::Function(Function &&src)
  : _nPoints{src._nPoints},
    _f{src._f}
{
  src._nPoints = 0;
  src._f = nullptr;
}

Function& Function::operator=(const Function &src) {
  // Guard self assignment
  if (this == &src) { return *this; }

  if (_nPoints != src._nPoints) {           // resource in *this cannot be reused
    float* temp = new float[src._nPoints];  // allocate resource, if throws, do nothing
    delete[] _f;                         // release resource in *this
    _f = temp;
    _nPoints = src._nPoints;
  } 

  std::copy(src._f, src._f + src._nPoints, _f);
  return *this;
}

Function& Function::operator=(Function &&src) {
  // Guard self assignment
  if (this == &src) { return *this; }

  delete[] _f;       // release resource in *this
  _f = src._f;
  _nPoints = src._nPoints;
  src._f = nullptr;  // leave src in valid state
  src._nPoints = 0;

  return *this;
}

Function::~Function() {
  delete[] _f;
}

float Function::operator[](int idx) const {
  if (idx >= _nPoints) {
    throw std::out_of_range("Error: index in 'float Function::[]' bigger than _nPoints");
  }
  if (idx < 0) { return 0.0; }
  return _f[idx];
}

float& Function::operator[](int idx) {
  if (idx >= _nPoints) {
    throw std::out_of_range("Error: index in 'float& Function::[]' bigger than _nPoints");
  }
  if (idx < 0) {
    throw std::out_of_range("Error: index in 'float& Function::[]' lower than 0");
  }
  return _f[idx];
}

float* Function::f() const {
  return _f;
}

void Function::del() {
  delete[] _f;
  _f = nullptr;
}
