#ifndef VOLUME_H
#define VOLUME_H

#include "utils.hpp"
#include <ostream>

//to do: template class T
class Volume {
  int _nPoints;
  Quaternion* _data;
  
 public:
  Volume(int nPoints = 0, Quaternion data = Quaternion{});
  Volume(int nPoints, Quaternion* data);
  Volume(const Volume &src);            //copy constructor
  Volume(Volume &&src);			            //move constructor
  Volume& operator=(const Volume &src); //copy assignment
  Volume& operator=(Volume &&src);	    //move assignment
  ~Volume();                             //destructor

  int nPoints() { return _nPoints; };
  Quaternion operator[](int index) const { return _data[index]; };

  friend std::ostream& operator<<(std::ostream& os, const Volume& obj);
};

#endif  //VOLUME_H