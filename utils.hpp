#ifndef UTILS_HPP
#define UTILS_HPP

#include <ostream>

struct Interval {
  float min;
  float max;
};

struct Point
{
  float x = 0;
  float y = 0;
  float z = 0;

  friend bool operator==(const Point& lhs, const Point& rhs) {
    /*if(std::abs(lhs.x-rhs.x) < std::numeric_limits<float>::epsilon() &&
     std::abs(lhs.y-rhs.y) < std::numeric_limits<float>::epsilon() &&
     std::abs(lhs.z-rhs.z) < std::numeric_limits<float>::epsilon()) {
    */
    if(std::abs(lhs.x-rhs.x) <= 1E-14 && std::abs(lhs.y-rhs.y) <= 1E-14 && std::abs(lhs.z-rhs.z) <= 1E-14)
      return true;

    return false; 
  }

  friend std::ostream& operator<<(std::ostream& os, const Point& obj) {
    os << '(' << obj.x << ", " << obj.y << ", " << obj.z << ')';
    return os;
  }

  float operator[](int coordinate) {
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

  float D(int i) {
    return x*x + y*y + z*z - this->operator[](i)*this->operator[](i);
  }

  float D(int i, int j) {
    return -this->operator[](i) * this->operator[](j);
  }
};

struct Quaternion
{
  float w = 0;
  float x = 0;
  float y = 0;
  float z = 0;

  friend bool operator==(const Quaternion& lhs, const Quaternion& rhs) {
    if(std::abs(lhs.w-rhs.w) <= 1E-14 && std::abs(lhs.x-rhs.x) <= 1E-14 &&
      std::abs(lhs.y-rhs.y) <= 1E-14 && std::abs(lhs.z-rhs.z) <= 1E-14)
      return true;

    return false; 
  }

  friend std::ostream& operator<<(std::ostream& os, const Quaternion& obj) {
    os << '(' << obj.w << ", " << obj.x << ", " << obj.y << ", " << obj.z << ')';
    return os;
  }

  float operator[](int coordinate) const {
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

  float D(int i) const {
    // ATTENZIONE
    return w*w + x*x + y*y + z*z - this->operator[](i)*this->operator[](i);
  }

  float D(int i, int j) const {
    // ATTENZIONE
    return -this->operator[](i) * this->operator[](j);
  }
};
  
inline float absolute(float x) {
  if (x < 0)
    return -x;
  return x;
}

#endif //UTILS_HPP
