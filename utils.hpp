#ifndef UTILS_HPP
#define UTILS_HPP

#include <ostream>

template <typename Tf>
struct Interval {
  Tf min;
  Tf max;
};

template <typename Tf>
struct Point
{
  Tf x = 0;
  Tf y = 0;
  Tf z = 0;

  friend bool operator==(const Point& lhs, const Point& rhs) {
    if(std::abs(lhs.x-rhs.x) <= 1E-14 && std::abs(lhs.y-rhs.y) <= 1E-14 &&
      std::abs(lhs.z-rhs.z) <= 1E-14)
      return true;

    return false; 
  }

  friend std::ostream& operator<<(std::ostream& os, const Point& obj) {
    os << obj.x << " " << obj.y << " " << obj.z;
    return os;
  }

  Tf operator[](int coordinate) const {
    switch (coordinate)
    {
    case 0:
      return x;
      break;
    case 1:
      return y;
      break;
    case 2:
      return z;
      break;
    
    default:
      throw std::invalid_argument(
        "operator[] accepts only [0, 1, 2], provided " + coordinate);
      break;
    }
  }

  // ATTENZIONE const
  Tf D(int i) const {
    return x*x + y*y + z*z - this->operator[](i)*this->operator[](i);
  }

  // ATTENZIONE const
  Tf D(int i, int j) const {
    return -this->operator[](i) * this->operator[](j);
  }
};

template <typename Tf>
struct Quaternion
{
  Tf w = 0;
  Tf x = 0;
  Tf y = 0;
  Tf z = 0;

  friend bool operator==(const Quaternion& lhs, const Quaternion& rhs) {
    if (std::abs(lhs.w-rhs.w) <= 1E-14 && std::abs(lhs.x-rhs.x) <= 1E-14 &&
      std::abs(lhs.y-rhs.y) <= 1E-14 && std::abs(lhs.z-rhs.z) <= 1E-14)
      return true;

    return false; 
  }

  friend std::ostream& operator<<(std::ostream& os, const Quaternion& obj) {
    os << obj.w << " " << obj.x << " " << obj.y << " " << obj.z;
    return os;
  }

  Tf operator[](int coordinate) const {
    switch (coordinate)
    {
    case 0:
      return w;
      break;
    case 1:
      return x;
      break;
    case 2:
      return y;
      break;
    case 3:
      return z;
      break;
    
    default:
      throw std::invalid_argument(
        "operator[] accepts only [0, 1, 2, 3], provided " + coordinate);
      break;
    }
  }

  // ATTENZIONE const
  Tf D(int i) const {
    Tf result = 0.;
    for (int j = 0; j < 4; ++j) {
      if (j != i) {
        result += this->operator[](j)*this->operator[](j);
      }
    }
    return result;
  }

  // ATTENZIONE const
  Tf D(int i, int j, Tf delta_i=0., Tf delta_j=0.) const {
    return -(this->operator[](i)+delta_i) * (this->operator[](j)+delta_j);
  }
};

template <typename Tf>
inline Tf absolute(Tf x) {
  if (x < 0)
    return -x;
  return x;
}

#endif //UTILS_HPP
