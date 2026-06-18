#ifndef SURFACE_CPP
#define SURFACE_CPP

#include "surface.h"
#include <limits>
#include <iostream>
#include <iomanip>

template<typename T>
Surface<T>::Surface(int nPoints, T data) :
                _nPoints{nPoints},
                _data{new T[nPoints]} {
  std::fill(_data, _data + nPoints, data);
}

template<typename T>
Surface<T>::Surface(int nPoints, T *data) :
                _nPoints{nPoints},
                _data{new T[nPoints]} {
  std::copy(data, data + nPoints, _data);
}

template<typename T>
Surface<T>::Surface(const Surface &src) :
                _nPoints{src._nPoints},
                _data{new T[src._nPoints]} {
  
  std::copy(src._data, src._data + src._nPoints, _data);

}

template<typename T>
Surface<T>::Surface(Surface &&src) :
              _nPoints{src._nPoints},
              _data{src._data} {
  src._nPoints = 0;
  src._data = nullptr;
}

template<typename T>
Surface<T> &Surface<T>::operator=(const Surface &src) {
  // Guard self assignment
  if (this == &src)
    return *this;

  if (_nPoints != src._nPoints) {           // resource in *this cannot be reused
    T* temp = new T[src._nPoints];  // allocate resource, if throws, do nothing
    delete[] _data;                         // release resource in *this
    _data = temp;
    _nPoints = src._nPoints;
  } 

  std::copy(src._data, src._data + src._nPoints, _data);
  return *this;
}

template<typename T>
Surface<T> &Surface<T>::operator=(Surface &&src) {
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

template<typename T>
Surface<T>::~Surface() {
  delete[] _data;
}

#endif