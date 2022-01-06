#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H

#include <string>

class CostFunction {
public:
  CostFunction(const std::string cost);

protected:
  std::string cost;
};

#endif