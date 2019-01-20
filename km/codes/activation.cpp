/*activation.cpp*/
#include "activation.h"

void Sigmoid::apply(double* dest, double* src, int len) {
	for (int i = 0; i < len; i++) {
		if (src[i] >= 0) {
			dest[i] = 1.0 / (1.0 + exp(-src[i]));
		}
		else {
			dest[i] = exp(src[i]) / (1.0 + exp(src[i]));
		}
	}
}

void Sigmoid::feedback(double* list, int len) {
	double sigmoid;
	for (int i = 0; i < len; i++) {
		if (list[i] >= 0) {
			sigmoid = 1.0 / (1.0 + exp(-list[i]));
		}
		else {
			sigmoid = exp(list[i]) / (1.0 + exp(list[i]));
		}
		list[i] = (1 - sigmoid)*sigmoid;
	}
}

void Tanh::apply(double* dest, double* src, int len) {
	double enx;
	double ex;
	for (int i = 0; i < len; i++) {
		if (src[i] >= 0) {
			enx = exp(-src[i]);
			ex = 1.0 / enx;
		}
		else {
			ex = exp(src[i]);
			enx = 1.0 / ex;
		}
		dest[i] = (ex - enx) / (ex + enx);
	}
}

void Tanh::feedback(double* list, int len) {
	double enx;
	double ex;
	for (int i = 0; i < len; i++) {
		if (list[i] >= 0) {
			enx = exp(-list[i]);
			ex = 1.0 / enx;
		}
		else {
			ex = exp(list[i]);
			enx = 1.0 / ex;
		}
		list[i] = 1.0 - ((ex - enx)*(ex - enx)) / ((ex + enx)*(ex + enx));
	}
}

void Relu::apply(double* dest, double* src, int len) {

	for (int i = 0; i < len; i++) {
		if (src[i] >= 0) {
			dest[i] = src[i];
		}
		else {
			dest[i] = 0;
		}
	}
}

void Relu::feedback(double* list, int len) {
	for (int i = 0; i < len; i++) {
		if (list[i] >= 0) {
			list[i] = 1;
		}
		else {
			list[i] = 0;
		}
	}
}

void Softmax::apply(double* dest, double* src, int len) {
	double max = src[0];
	for (int i = 1; i < len; i++) {
		if (max < src[i]) {
			max = src[i];
		}
	}
	double base = 0.0;
	for (int i = 0; i < len; i++) {
		dest[i] = exp(src[i] - max);
		base += dest[i];
	}
	for (int i = 0; i < len; i++) {
		dest[i] = dest[i] / base;
	}
}

void Softmax::feedback(double* list, int len) {

}

void Identity::apply(double* dest, double* src, int len) {
	for (int i = 0; i < len; i++) {
		dest[i] = src[i];
	}
}
void Identity::feedback(double* list, int len) {
	for (int i = 0; i < len; i++) {
		list[i] = 1;
	}
}

void Swish::apply(double* dest, double* src, int len) { //swish function
	for (int i = 0; i < len; i++) {
		if (src[i] >= 0) {
			dest[i] = src[i] / (1.0 + exp(-src[i]));
		}
		else {
			dest[i] = (src[i] * exp(src[i])) / (1.0 + exp(src[i]));
		}
	}
}

void Swish::feedback(double* list, int len) { //swish derivative function
	double sigmoid;
	for (int i = 0; i < len; i++) {
		if (list[i] >= 0) {
			sigmoid = 1.0 / (1.0 + exp(-list[i]));
		}
		else {
			sigmoid = exp(list[i]) / (1.0 + exp(list[i]));
		}
		list[i] = sigmoid + list[i] * sigmoid*(1.0 - sigmoid);
	}
}

