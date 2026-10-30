#include "function.h"

#include <algorithm>
#include <iostream>

template<typename Tf>
Function<Tf>::Function() : _nPoints{0}, _f{nullptr} {}

template<typename Tf>
Function<Tf>::Function(int np)
  : _nPoints{np},
    _f{new Tf[np]}
{
  std::fill(_f, _f + np, 0.);
}

template<typename Tf>
Function<Tf>::Function(const Function &src)
  : _nPoints{src._nPoints},
    _f{new Tf[_nPoints]}
{
  std::copy(src._f, src._f + src._nPoints, _f);
}

template<typename Tf>
Function<Tf>::Function(Function &&src)
  : _nPoints{src._nPoints},
    _f{src._f}
{
  src._nPoints = 0;
  src._f = nullptr;
}

template<typename Tf>
Function<Tf>& Function<Tf>::operator=(const Function<Tf> &src) {
  // Guard self assignment
  if (this == &src) { return *this; }

  if (_nPoints != src._nPoints) {           // resource in *this cannot be reused
    Tf* temp = new Tf[src._nPoints];  // allocate resource, if throws, do nothing
    delete[] _f;                         // release resource in *this
    _f = temp;
    _nPoints = src._nPoints;
  } 

  std::copy(src._f, src._f + src._nPoints, _f);
  return *this;
}

template<typename Tf>
Function<Tf>& Function<Tf>::operator=(Function<Tf> &&src) {
  // Guard self assignment
  if (this == &src) { return *this; }

  delete[] _f;       // release resource in *this
  _f = src._f;
  _nPoints = src._nPoints;
  src._f = nullptr;  // leave src in valid state
  src._nPoints = 0;

  return *this;
}

template<typename Tf>
Function<Tf>::~Function() {
  delete[] _f;
}

template<typename Tf>
Tf Function<Tf>::operator[](int idx) const {
  if (idx >= _nPoints) {
    throw std::out_of_range("Error: index in 'Tf Function::[]' bigger than _nPoints");
  }
  if (idx < 0) { return 0.0; }
  return _f[idx];
}

template<typename Tf>
Tf& Function<Tf>::operator[](int idx) {
  if (idx >= _nPoints) {
    throw std::out_of_range("Error: index in 'Tf& Function::[]' bigger than _nPoints");
  }
  if (idx < 0) {
    throw std::out_of_range("Error: index in 'Tf& Function::[]' lower than 0");
  }
  return _f[idx];
}

template<typename Tf>
Tf* Function<Tf>::f() const {
  return _f;
}

template<typename Tf>
void Function<Tf>::del() {
  delete[] _f;
  _f = nullptr;
}

template class Function<double>;
template class Function<float>;
