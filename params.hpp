#ifndef PARAMS_H
#define PARAMS_H

#include "utils.hpp"
#include <fstream>
#include <stdexcept>
#include <string>

class Params {
  public:
  // Parameters
  std::string path;
  float T;
  float h;
  Interval w;
  Interval x;
  Interval y;
  Interval z;
  float radius;
  Quaternion center;

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
    float w_min = 0.;
    float w_max = 4.;
    float x_min = 0.;
    float x_max = 4.;
    float y_min = 0.;
    float y_max = 4.;
    float z_min = 0.;
    float z_max = 4.;
    float cw = 2.;
    float cx = 2.;
    float cy = 2.;
    float cz = 2.;
    while (fin >> key >> value) {
      if (key == "path") path = value;
      else if (key == "T") T = std::stof(value);
      else if (key == "h") h = std::stof(value);
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
    }
    w = {w_min, w_max};
    x = {x_min, x_max};
    y = {y_min, y_max};
    z = {z_min, z_max};
    center = {cw, cx, cy, cz};
  }
};

#endif
