#ifndef SURFACE_H
#define SURFACE_H

#include "utils.hpp"
#include <ostream>

template<typename T>
class Surface {
  int _nPoints;
  T* _data;
  
 public:
  Surface(int nPoints = 0, T data = Point{});
  Surface(int nPoints, T* data);
  Surface(const Surface &src);            //copy constructor
  Surface(Surface &&src);			            //move constructor
  Surface& operator=(const Surface &src); //copy assignment
  Surface& operator=(Surface &&src);	    //move assignment
  ~Surface();                             //destructor

  int nPoints() { return _nPoints; };
  T operator[](int index) const { return _data[index]; };

  friend std::ostream& operator<<(std::ostream& os, const Surface<T>& obj) {
    for (int i = 0; i < obj._nPoints; ++i) {
      os << obj._data[i] << '\n';  // chiama operator<<(ostream, T)
    }
    return os;
  }
};

#include "surface.cpp"
#endif  //SURFACE_H