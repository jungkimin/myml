/*pooler.cpp*/
#include "pooler.h"

Pooler::Pooler() {

}
AvgPooler::AvgPooler() {
	model = AVG_POOLER;
}
AvgPooler::AvgPooler(int c, int w) {
	model = AVG_POOLER;
	input_width = w;
	channel = c;
}
void AvgPooler::apply(double* dest, double* src) {

	int src_width = input_width;
	if (src_width % 2 == 0) {
		int dest_width = input_width / 2;
		int d_c_step = dest_width * dest_width;
		int s_c_step = src_width * src_width;
		double accumulator = 0.0;
		for (int i = 0; i < channel*d_c_step; i++) {
			dest[i] = 0.0;
		}
		for (int c = 0; c < channel; c++) {
			for (int i = 0; i < src_width; i += 2) {
				for (int j = 0; j < src_width; j += 2) {
					accumulator = src[c*s_c_step + i * src_width + j];
					accumulator += src[c*s_c_step + i * src_width + j + 1];
					accumulator += src[c*s_c_step + (i + 1) * src_width + j];
					accumulator += src[c*s_c_step + (i + 1) * src_width + (j + 1)];
					dest[c*d_c_step + i / 2 * dest_width + j / 2] += 0.25*accumulator;
				}
			}
		}
	}
	else {
		int dest_width = (input_width / 2) + 1;
		int d_c_step = dest_width * dest_width;
		int s_c_step = src_width * src_width;
		double accumulator = 0.0;
		for (int i = 0; i < channel*d_c_step; i++) {
			dest[i] = 0.0;
		}
		for (int c = 0; c < channel; c++) {
			for (int i = 0; i < src_width - 1; i += 2) {
				for (int j = 0; j < src_width - 1; j += 2) {
					accumulator = src[c*s_c_step + i * src_width + j];
					accumulator += src[c*s_c_step + i * src_width + j + 1];
					accumulator += src[c*s_c_step + (i + 1) * src_width + j];
					accumulator += src[c*s_c_step + (i + 1) * src_width + (j + 1)];
					dest[c*d_c_step + i / 2 * dest_width + j / 2] += 0.25*accumulator;
				}
			}

			for (int i = 0; i < src_width - 1; i += 2) {
				accumulator = src[c*s_c_step + i * src_width + (src_width - 1)];
				accumulator += src[c*s_c_step + (i + 1)* src_width + (src_width - 1)];
				dest[c*d_c_step + (i / 2)*dest_width + (dest_width - 1)] = 0.5*accumulator;

				accumulator = src[c*s_c_step + (src_width - 1) * src_width + i];
				accumulator += src[c*s_c_step + (src_width - 1) * src_width + (i + 1)];
				dest[c*d_c_step + (dest_width - 1)*dest_width + (i / 2)] = 0.5*accumulator;
			}
			dest[c*d_c_step + (dest_width - 1)*dest_width + (dest_width - 1)] = src[c*s_c_step + (src_width - 1) * src_width + (src_width - 1)];
		}

	}
}
void AvgPooler::feedback(double* dest, double* src) {
	int dest_width = input_width;
	if (dest_width % 2 == 0) {
		int src_width = input_width / 2;
		int s_c_step = src_width * src_width;
		int d_c_step = dest_width * dest_width;
		double accumulator = 0.0;
		for (int c = 0; c < channel; c++) {
			for (int i = 0; i < dest_width; i += 2) {
				for (int j = 0; j < dest_width; j += 2) {
					dest[c*d_c_step + i * dest_width + j] = 0.25*src[c*s_c_step + (i / 2) * src_width + j / 2];
					dest[c*d_c_step + i * dest_width + (j + 1)] = 0.25*src[c*s_c_step + (i / 2)* src_width + j / 2];
					dest[c*d_c_step + (i + 1) * dest_width + j] = 0.25*src[c*s_c_step + (i / 2) * src_width + j / 2];
					dest[c*d_c_step + (i + 1) * dest_width + (j + 1)] = 0.25*src[c*s_c_step + (i / 2) * src_width + j / 2];
				}
			}
		}
	}
	else {
		int src_width = (input_width / 2) + 1;
		int s_c_step = src_width * src_width;
		int d_c_step = dest_width * dest_width;
		double accumulator = 0.0;
		for (int c = 0; c < channel; c++) {
			for (int i = 0; i < dest_width - 1; i += 2) {
				for (int j = 0; j < dest_width - 1; j += 2) {
					dest[c*d_c_step + i * dest_width + j] = 0.25*src[c*s_c_step + (i / 2) * src_width + j / 2];
					dest[c*d_c_step + i * dest_width + (j + 1)] = 0.25*src[c*s_c_step + (i / 2) * src_width + j / 2];
					dest[c*d_c_step + (i + 1) * dest_width + j] = 0.25*src[c*s_c_step + (i / 2) * src_width + j / 2];
					dest[c*d_c_step + (i + 1) * dest_width + (j + 1)] = 0.25*src[c*s_c_step + (i / 2)* src_width + j / 2];
				}
			}
			for (int i = 0; i < dest_width - 1; i += 2) {
				dest[c*d_c_step + i * dest_width + (dest_width - 1)] = 0.5*src[c*s_c_step + (i / 2)* src_width + (src_width - 1)];
				dest[c*d_c_step + (i + 1) * dest_width + (dest_width - 1)] = 0.5*src[c*s_c_step + (i / 2)* src_width + (src_width - 1)];

				dest[c*d_c_step + (dest_width - 1) * dest_width + i] = 0.5*src[c*s_c_step + (src_width - 1)* src_width + i / 2];
				dest[c*d_c_step + (dest_width - 1) * dest_width + (i + 1)] = 0.5*src[c*s_c_step + (src_width - 1)* src_width + i / 2];
			}
			dest[c*d_c_step + (dest_width - 1) * dest_width + (dest_width - 1)] = src[c*s_c_step + (src_width - 1)* src_width + (src_width - 1)];
		}
	}
}

MaxPooler::MaxPooler() {
	alloced = false;
	model = MAX_POOLER;
}
MaxPooler::MaxPooler(int c, int w) {
	input_width = w;
	channel = c;

	model = MAX_POOLER;
	alloced = false;
	contribute_table = new double[input_width*input_width*channel];
	alloced = true;

}
void MaxPooler::apply(double* dest, double* src) {

	for (int i = 0; i < channel*input_width*input_width; i++) {
		contribute_table[i] = 0.0;
	}
	double max;
	int h, w;
	int src_width = input_width;
	if (src_width % 2 == 0) {
		int dest_width = src_width / 2;
		int d_c_step = (src_width / 2)*(src_width / 2);
		int s_c_step = src_width * src_width;
		double accumulator = 0.0;
		for (int i = 0; i < channel*d_c_step; i++) {
			dest[i] = 0.0;
		}
		for (int c = 0; c < channel; c++) {
			for (int i = 0; i < src_width; i += 2) {
				for (int j = 0; j < src_width; j += 2) {

					max = src[c*s_c_step + i * src_width + j];
					h = 0;
					w = 0;
					if (max < src[c*s_c_step + i * src_width + j]) {
						h = 0;
						w = 0;
						max = src[c*s_c_step + i * src_width + j];
					}
					else if (max < src[c*s_c_step + i * src_width + j + 1]) {
						h = 0;
						w = 1;
						max = src[c*s_c_step + i * src_width + j + 1];
					}
					else if (max < src[c*s_c_step + (i + 1) * src_width + j]) {
						h = 1;
						w = 0;
						max = src[c*s_c_step + (i + 1) * src_width + j];
					}
					else if (max < src[c*s_c_step + (i + 1) * src_width + j + 1]) {
						h = 1;
						w = 1;
						max = src[c*s_c_step + (i + 1) * src_width + j + 1];
					}
					dest[c*d_c_step + (i / 2) * dest_width + j / 2] = max;
					contribute_table[c*s_c_step + (i + h) * src_width + j + w] = 1.0;
				}
			}
		}
	}
	else {
		int dest_width = (src_width / 2) + 1;
		int d_c_step = dest_width * dest_width;
		int s_c_step = src_width * src_width;
		double accumulator = 0.0;
		for (int i = 0; i < channel*d_c_step; i++) {
			dest[i] = 0.0;
		}
		for (int c = 0; c < channel; c++) {
			for (int i = 0; i < src_width - 1; i += 2) {
				for (int j = 0; j < src_width - 1; j += 2) {

					max = src[c*s_c_step + i * src_width + j];
					h = 0;
					w = 0;
					if (max < src[c*s_c_step + i * src_width + j]) {
						h = 0;
						w = 0;
						max = src[c*s_c_step + i * src_width + j];
					}
					else if (max < src[c*s_c_step + i * src_width + j + 1]) {
						h = 0;
						w = 1;
						max = src[c*s_c_step + i * src_width + j + 1];
					}
					else if (max < src[c*s_c_step + (i + 1) * src_width + j]) {
						h = 1;
						w = 0;
						max = src[c*s_c_step + (i + 1) * src_width + j];
					}
					else if (max < src[c*s_c_step + (i + 1) * src_width + j + 1]) {
						h = 1;
						w = 1;
						max = src[c*s_c_step + (i + 1) * src_width + j + 1];
					}
					dest[c*d_c_step + (i / 2) * dest_width + j / 2] = max;
					contribute_table[c*s_c_step + (i + h) * src_width + j + w] = 1.0;
				}
			}
			for (int i = 0; i < src_width - 1; i += 2) {
				if (src[c*s_c_step + (src_width - 1)*src_width + i] > src[c*s_c_step + (src_width - 1)*src_width + i + 1]) {
					dest[c*d_c_step + (dest_width - 1)*dest_width + i / 2] = src[c*s_c_step + (src_width - 1)*src_width + i];
					contribute_table[c*s_c_step + (src_width - 1) * src_width + i] = 1.0;
				}
				else {
					dest[c*d_c_step + (dest_width - 1)*dest_width + i / 2] = src[c*s_c_step + (src_width - 1)*src_width + (i + 1)];
					contribute_table[c*s_c_step + (src_width - 1) * src_width + i + 1] = 1.0;
				}
				if (src[c*s_c_step + i * src_width + (src_width - 1)] > src[c*s_c_step + (i + 1)*src_width + (src_width - 1)]) {
					dest[c*d_c_step + (i / 2)*dest_width + (dest_width - 1)] = src[c*s_c_step + i * src_width + (src_width - 1)];
					contribute_table[c*s_c_step + i * src_width + (src_width - 1)] = 1.0;
				}
				else {
					dest[c*d_c_step + (i / 2)*dest_width + (dest_width - 1)] = src[c*s_c_step + (i + 1) * src_width + (src_width - 1)];
					contribute_table[c*s_c_step + (i + 1) * src_width + (src_width - 1)] = 1.0;
				}
			}
			dest[c*d_c_step + (dest_width - 1)*dest_width + (dest_width - 1)] = src[c*s_c_step + (src_width - 1) * src_width + (src_width - 1)];
			contribute_table[c*s_c_step + (src_width - 1) * src_width + (src_width - 1)] = 1.0;
		}
	}
}
void MaxPooler::feedback(double* dest, double* src) {

	int dest_width = input_width;
	if (dest_width % 2 == 0) {
		int src_width = dest_width / 2;
		for (int i = 0; i < channel*input_width*input_width; i++) {
			dest[i] = 0;
		}
		int s_c_step = src_width * src_width;
		int d_c_step = dest_width * dest_width;
		for (int c = 0; c < channel; c++) {
			for (int i = 0; i < dest_width; i += 2) {
				for (int j = 0; j < dest_width; j += 2) {
					if (contribute_table[c*d_c_step + i * dest_width + j] == 1.0) {
						dest[c*d_c_step + i * dest_width + j] = src[c*s_c_step + (i / 2) * src_width + j / 2];
					}
					else if (contribute_table[c*d_c_step + i * dest_width + j + 1] == 1.0) {
						dest[c*d_c_step + i * dest_width + j + 1] = src[c*s_c_step + (i / 2) * src_width + j / 2];
					}
					else if (contribute_table[c*d_c_step + (i + 1) * dest_width + j] == 1.0) {
						dest[c*d_c_step + (i + 1) * dest_width + j] = src[c*s_c_step + (i / 2) * src_width + j / 2];
					}
					else if (contribute_table[c*d_c_step + (i + 1) * dest_width + j + 1] == 1.0) {
						dest[c*d_c_step + (i + 1) * dest_width + j + 1] = src[c*s_c_step + (i / 2) * src_width + j / 2];
					}
				}
			}
		}
	}

	else {
		int src_width = (dest_width / 2) + 1;
		for (int i = 0; i < channel*input_width*input_width; i++) {
			dest[i] = 0.0;
		}
		int s_c_step = src_width * src_width;
		int d_c_step = dest_width * dest_width;
		for (int c = 0; c < channel; c++) {
			for (int i = 0; i < dest_width - 1; i += 2) {
				for (int j = 0; j < dest_width - 1; j += 2) {
					if (contribute_table[c*d_c_step + i * dest_width + j] == 1.0) {
						dest[c*d_c_step + i * dest_width + j] = src[c*s_c_step + (i / 2) * src_width + j / 2];
					}
					else if (contribute_table[c*d_c_step + i * dest_width + j + 1] == 1.0) {
						dest[c*d_c_step + i * dest_width + j + 1] = src[c*s_c_step + (i / 2) * src_width + j / 2];
					}
					else if (contribute_table[c*d_c_step + (i + 1) * dest_width + j] == 1.0) {
						dest[c*d_c_step + (i + 1) * dest_width + j] = src[c*s_c_step + (i / 2) * src_width + j / 2];
					}
					else if (contribute_table[c*d_c_step + (i + 1) * dest_width + j + 1] == 1.0) {
						dest[c*d_c_step + (i + 1) * dest_width + j + 1] = src[c*s_c_step + (i / 2) * src_width + j / 2];
					}
				}
			}
			for (int i = 0; i < dest_width - 1; i += 2) {
				if (contribute_table[c*d_c_step + (dest_width - 1) * dest_width + i] == 1.0) {
					dest[c*d_c_step + (dest_width - 1) * dest_width + i] = src[c*s_c_step + (src_width - 1) * src_width + i / 2];
				}
				else {
					dest[c*d_c_step + (dest_width - 1) * dest_width + (i + 1)] = src[c*s_c_step + (src_width - 1) * src_width + i / 2];
				}

				if (contribute_table[c*d_c_step + i * dest_width + (dest_width - 1)] == 1.0) {
					dest[c*d_c_step + i * dest_width + (dest_width - 1)] = src[c*s_c_step + (i / 2) * src_width + (src_width - 1)];
				}
				else {
					dest[c*d_c_step + (i + 1) * dest_width + (dest_width - 1)] = src[c*s_c_step + (i / 2) * src_width + (src_width - 1)];
				}
			}
			dest[c*d_c_step + (dest_width - 1) * dest_width + (dest_width - 1)] = src[c*s_c_step + (src_width - 1) * src_width + (src_width - 1)];
		}
	}
}

