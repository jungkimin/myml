/*activation.h*/
#ifndef __ACTIVATION__
#define __ACTIVATION__
#include <math.h>
#include <iostream>
enum ACTIVATION { SIGMOID, RELU, TANH, IDENTITY, SWISH, SOFTMAX };

class Activation {
public:
	ACTIVATION function;
	Activation(){}
	void print_what_it_is() {
		if(function == SIGMOID) {
			std::cout <<"SIGMOID";
		}
		else if (function == RELU) {
			std::cout << "RELU";
		}
		else if (function == TANH) {
			std::cout << "TANH";
		}
		else if (function == IDENTITY) {
			std::cout << "IDENTITY";
		}
		else if (function == SWISH) {
			std::cout << "SWISH";
		}
		else if (function == SOFTMAX) {
			std::cout << "SOFTMAX";
		}
	}
	virtual void apply(double* dest, double* src, int len) = 0;
	virtual void feedback(double* list, int len) = 0;
};
class Sigmoid : public Activation {
public:
	Sigmoid() {
		function = SIGMOID;
	}
	void apply(double* dest, double* src, int len);
	void feedback(double* list, int len);
};
class Tanh : public Activation {
public:
	Tanh() {
		function = TANH;
	}
	void apply(double* dest, double* src, int len);
	void feedback(double* list, int len);
};
class Relu : public Activation {
public:
	Relu() {
		function = RELU;
	}
	void apply(double* dest, double* src, int len);
	void feedback(double* list, int len);
};

class Softmax : public Activation {
public:
	Softmax() {
		function = SOFTMAX;
	}
	void apply(double* dest, double* src, int len);
	void feedback(double* list, int len);
};

class Identity : public Activation {
public:
	Identity() {
		function = IDENTITY;
	}
	void apply(double* dest, double* src, int len);
	void feedback(double* list, int len);
};

class Swish : public Activation {
public:
	Swish() {
		function = SWISH;
	}
	void apply(double* dest, double* src, int len);
	void feedback(double* list, int len);
};
#endif // !__ACTIVATION__


