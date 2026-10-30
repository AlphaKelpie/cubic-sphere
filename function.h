#ifndef FUNCTION_H
#define FUNCTION_H

template <typename Tf>
class Function {
  private:
  int _nPoints;
  Tf* _f;

  public:
  Function();                               //default constructor
  Function(int np);                         //zero _f constructor
  Function(const Function &src);            //copy constructor
  Function(Function &&src);                 //move constructor
  Function& operator=(const Function &src); //copy assignment
  Function& operator=(Function &&src);      //move assignment
  ~Function();                              //destructor

  Tf operator[](int idx) const;         //idx element of _f
  Tf& operator[](int idx);              //idx element of _f

  Tf* f() const;                        //get _f
  void del();                               //set _f to nullptr
};

#endif
