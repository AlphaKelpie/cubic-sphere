#ifndef PARAMS_H
#define PARAMS_H

#include "utils.hpp"
#include <fstream>
#include <stdexcept>
#include <string>

template <typename Tf>
class Params {
  public:
  // Parameters
  std::string path;
  std::string pathdata;
  Tf T;
  Tf h;
  Tf deltaT{};
  Tf deltaC{};
  Interval<Tf> w;
  Interval<Tf> x;
  Interval<Tf> y;
  Interval<Tf> z;
  Tf radius;
  Quaternion<Tf> center;
  int total{2002};
  int step{100};

  // Global istance access
  static Params& get() {
    if (!_instance) {
      throw std::runtime_error("Params not initialized! Call Params::load() first.");
    }
    return *_instance;
  }

  // Load file and create istance
  static void load(std::string filename) {
    _instance = new Params(filename);
  }

  private:
  inline static Params* _instance = nullptr;

  Params(std::string filename) {
    std::fstream fin(filename, std::ios::in);
    if (!fin.is_open()) {
      throw std::runtime_error("Cannot open params file: " + filename);
    }

    std::string key, value;
    Tf w_min = 0.;
    Tf w_max = 4.;
    Tf x_min = 0.;
    Tf x_max = 4.;
    Tf y_min = 0.;
    Tf y_max = 4.;
    Tf z_min = 0.;
    Tf z_max = 4.;
    Tf cw = 2.;
    Tf cx = 2.;
    Tf cy = 2.;
    Tf cz = 2.;
    while (fin >> key >> value) {
      if (key == "path") path = value;
      else if (key == "pathdata") pathdata = value;
      else if (key == "T") T = std::stof(value);
      else if (key == "h") h = std::stof(value);
      else if (key == "deltaT") deltaT = std::stof(value);
      else if (key == "deltaC") deltaC = std::stof(value);
      else if (key == "w_min") w_min = std::stof(value);
      else if (key == "w_max") w_max = std::stof(value);
      else if (key == "x_min") x_min = std::stof(value);
      else if (key == "x_max") x_max = std::stof(value);
      else if (key == "y_min") y_min = std::stof(value);
      else if (key == "y_max") y_max = std::stof(value);
      else if (key == "z_min") z_min = std::stof(value);
      else if (key == "z_max") z_max = std::stof(value);
      else if (key == "radius") radius = std::stof(value);
      else if (key == "cw") cw = std::stof(value);
      else if (key == "cx") cx = std::stof(value);
      else if (key == "cy") cy = std::stof(value);
      else if (key == "cz") cz = std::stof(value);
      else if (key == "total") total = std::stoi(value) + 2;
      else if (key == "step") step = std::stoi(value);
    }
    w = {w_min, w_max};
    x = {x_min, x_max};
    y = {y_min, y_max};
    z = {z_min, z_max};
    center = {cw, cx, cy, cz};
  }
};

#endif
