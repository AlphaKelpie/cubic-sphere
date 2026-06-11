#ifndef FUNCTION_H
#define FUNCION_H

class Function {
  private:
  int _nPoints;
  double* _f;

  public:
  Function();                               //default constructor
  Function(int np);                         //zero _f constructor
  Function(const Function &src);            //copy constructor
  Function(Function &&src);                 //move constructor
  Function& operator=(const Function &src); //copy assignment
  Function& operator=(Function &&src);      //move assignment
  ~Function();                              //destructor

  double operator[](int idx) const;         //idx element of _f
  double& operator[](int idx);              //idx element of _f

  double* f() const;                        //get _f
  void del();                               //set _f to nullptr
};

#endif
