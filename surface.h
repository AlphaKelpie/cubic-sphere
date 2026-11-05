#ifndef SURFACE_H
#define SURFACE_H

#include <ostream>

#include "utils.hpp"

template <typename T>
class Surface {
  int _nPoints;
  T*  _data;

 public:
  Surface(int nPoints = 0, T data = T{});
  Surface(int nPoints, T* data);
  Surface(const Surface& src);  // copy constructor
  Surface(Surface&& src);  // move constructor
  Surface& operator=(const Surface& src);  // copy assignment
  Surface& operator=(Surface&& src);  // move assignment
  ~Surface();  // destructor

  int nPoints() { return _nPoints; };
  T   operator[](int index) const { return _data[index]; };

  friend std::ostream& operator<<(std::ostream& os, const Surface<T>& obj) {
    for (int i = 0; i < obj._nPoints; ++i) {
      os << obj._data[i] << '\n';  // chiama operator<<(ostream, T)
    }
    return os;
  }

  inline size_t kdtree_get_point_count() const {
    return static_cast<size_t>(_nPoints);
  }

  inline double kdtree_get_pt(size_t const idx, size_t const dim) const {
    if (dim == 0) return static_cast<double>(_data[idx][0]);
    if (dim == 1) return static_cast<double>(_data[idx][1]);
    if (dim == 2) return static_cast<double>(_data[idx][2]);
    if (dim == 3) return static_cast<double>(_data[idx][3]);

    return 0.;
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }
};

#endif  // SURFACE_H