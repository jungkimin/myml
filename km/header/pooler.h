/*pooler.h*/
#ifndef __POOLER__
#define __POOLER__
#include <iostream>
enum POOLER { MAX_POOLER, AVG_POOLER, NO_POOLER };
class Pooler {
public:
	POOLER model;
	int input_width;
	int channel;
	Pooler();
	virtual void apply(double* dest, double* src) = 0;
	virtual void feedback(double* dest, double* src) = 0;
	void print_what_it_is() {
		if (model == AVG_POOLER) {
			std::cout << "AVG_POOLER";
		}
		else if (model == MAX_POOLER) {
			std::cout << "MAX_POOLER";
		}
		else {
			std::cout << "NO_POOLER";
		}
	}
	virtual ~Pooler() {}
};
class AvgPooler : public Pooler {
public:
	AvgPooler();
	AvgPooler(int c, int w);
	void apply(double* dest, double* src);
	void feedback(double* dest, double* src);
};

class NoPooler : public Pooler {
public:
	NoPooler() {
		model = NO_POOLER;
	}
	void apply(double* dest, double* src) {}
	void feedback(double* dest, double* src) {}
};

class MaxPooler : public Pooler {
public:
	double* contribute_table;
	bool alloced;
	MaxPooler();
	MaxPooler(int c, int w);
	void apply(double* dest, double* src);
	void feedback(double* dest, double* src);
	virtual ~MaxPooler() {
		if (alloced == true) {
			alloced = false;
		}
	}

};
#endif // !__POOLER__

