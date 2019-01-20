#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include "activation.h"
#include "pooler.h"
enum UNIT { PERCEPTRONS, RESIDUAL_BLOCKS, CONVOLUSIONS };

using namespace std;
typedef vector<int> Size;
const double DBM = std::numeric_limits<double>::max();
random_device rdev;
mt19937 rEngine(rdev());
uniform_real_distribution<> Weight_rand(-0.5, 0.5);
normal_distribution <> Gaussian(0, 0.3);
normal_distribution <> NOISE_MAKER(0, 0.1);

namespace km {
	void fill(double* arr, double val, int len) {
		for (int i = 0; i < len; i++) {
			arr[i] = val;
		}
	}
	void fill_noise(double* arr, int len) {
		for (int i = 0; i < len; i++) {
			arr[i] = NOISE_MAKER(rEngine);
		}
	}

	void fill_gaussian_dist(double* arr, int len) {
		for (int i = 0; i < len; i++) {
			arr[i] = Gaussian(rEngine);
		}
	}

	void plus(double* dest, double* src, int len) {
		for (int i = 0; i < len; i++) {
			dest[i] += src[i];
		}
	}
	
	void minus(double* dest, double* src, int len) {
		for (int i = 0; i < len; i++) {
			dest[i] -= src[i];
		}
	}
	
	void copy(double* dest, double* src, int len) {
		for (int i = 0; i < len; i++) {
			dest[i] = src[i];
		}
	}
	
	double* clone(double* src, int len) {
		double* dest = new double[len];
		for (int i = 0; i < len; i++) {
			dest[i] = src[i];
		}
		return dest;
	}
	double softmax_loss(double* pred, double* label, int len) {
		double loss = 0.0;
		for (int i = 0; i < len; i++) {
			loss += (-1)*log(pred[i])*label[i];
		}
		return loss;
	}
	void softmax_cross_entropy_derivative(double* dest, double* pred, double* label, int len) {
		for (int i = 0; i < len; i++) {
			dest[i] = pred[i] - label[i];
		}
	}

}

class unit {
public:
	unit() {};
	virtual void load_input(double* in) = 0;
	virtual void forward_propagation(double* forward) = 0;
	virtual void update_preparation() = 0;
	virtual void update(double learning_rate) = 0;
	virtual void backpropagation(double* delta) = 0;
	virtual void accumulate_GD() = 0;
	virtual void pass_delta() = 0;
	virtual void backward_pass(double* delta) = 0;
	UNIT u_flag;
	Activation* activator;
	int input_size;
	int output_size;
	double* Input_container;
	double* Output_container;
	double* forward_pass;
	double* back_pass;
	virtual ~unit() {}
};

class conv_size {
public:
	int width_and_height;
	int n_channel;
	int n_dimension;
	conv_size() {}
	conv_size(int n_channel, int width_and_height, int n_dimension) {
		this->width_and_height = width_and_height;
		this->n_channel = n_channel;
		this->n_dimension = n_dimension;
	}
	bool isgood(int output_length, int input_length, bool pooling) {
		bool isok = true;
		if (width_and_height*width_and_height*n_channel != input_length) {
			isok = false;
		}

		if (pooling == false) {
			if (width_and_height*width_and_height*n_dimension != output_length) {
				isok = false;
			}
		}
		else if (pooling == true) {
			int out_width = width_and_height / 2;
			if (width_and_height % 2 == 1) {
				out_width = out_width + 1;
			}

			if (out_width*out_width*n_dimension != output_length) {
				isok = false;
			}
		}

		return isok;
	}
};
class perceptrons : public unit {
public:
	double* weight;
	double* gradients;

	int tot_len;
	int width;
	int height;
	perceptrons() {}
	perceptrons(int output_sz, int input_sz, ACTIVATION f = IDENTITY) {
		this->create(output_sz, input_sz, f);
	}

	void create(int output_sz, int input_sz, ACTIVATION f = IDENTITY) {
		u_flag = PERCEPTRONS;
		if (f == SIGMOID) {
			activator = new Sigmoid();
		}
		else if (f == RELU) {
			activator = new Relu();
		}
		else if (f == TANH) {
			activator = new Tanh();
		}
		else if (f == SOFTMAX) {
			activator = new Softmax();
		}
		else if (f == IDENTITY) {
			activator = new Identity();
		}
		else if (f == SWISH) {
			activator = new Swish();
		}
		output_size = output_sz;
		input_size = input_sz;
		height = output_sz;
		width = input_sz + 1;

		tot_len = height * width;
		weight = new double[tot_len];
		gradients = new double[tot_len];

		for (int i = 0; i < height*width; i++) {
			weight[i] = Weight_rand(rEngine);
		}
		back_pass = new double[input_size];
		Input_container = new double[width];
		Output_container = new double[height];
		forward_pass = new double[output_size];
	}

	void load_input(double* in) {
		Input_container[0] = 1.0;
		for (int i = 1; i < width; i++) {
			Input_container[i] = in[i - 1];
		}
	}

	void update(double learning_rate) {
		for (int i = 0; i < tot_len; i++) {
			weight[i] -= gradients[i] * learning_rate;
		}
	}

	void accumulate_GD() {
		int i, j;
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				gradients[i*width + j] += Output_container[i] * Input_container[j];
			}
		}
	}

	void pass_delta() {
		int i, j;
		for (i = 1; i < width; i++) {
			back_pass[i - 1] = 0.0; //initialize the value for further accumulating 
			for (j = 0; j < height; j++) {
				back_pass[i - 1] += weight[j*width + i] * Output_container[j];
			}
		}
	}

	void forward_propagation(double* forward) {
		load_input(forward);
		int h_idx;

		int h, w;
		for (h = 0; h < height; h++) {
			Output_container[h] = 0.0;
			h_idx = h * width;
			for (w = 0; w < width; w++) {
				Output_container[h] += weight[h_idx + w] * Input_container[w];
			}
		}
		activator->apply(forward_pass, Output_container, output_size);
	}

	void backpropagation(double* delta) {

		activator->feedback(Output_container, output_size);
		for (int i = 0; i < output_size; i++) {
			Output_container[i] *= delta[i];
		}
		accumulate_GD();
		pass_delta();
	}

	void backward_pass(double* delta) {
		activator->feedback(Output_container, output_size);
		for (int i = 0; i < output_size; i++) {
			Output_container[i] *= delta[i];
		}
		pass_delta();
	}

	void update_preparation() {
		for (int i = 0; i < tot_len; i++) {
			gradients[i] = 0.0;
		}
	}

	~perceptrons() {
		delete[] weight;
		delete[] gradients;
		delete[] Input_container;
		delete[] Output_container;
		delete[] back_pass;
		delete[] forward_pass;
		delete activator;
	}

};

class convolutions : public unit {
public:
	double** filter;
	double** gradients;
	int filt_tot_len;
	int in_tot_len;
	int padded_tot_len;
	int n_channel;
	int input_width;
	int output_width;
	int n_dimension;
	int filter_width;
	int output_tot_len;
	Pooler *pooler;
	double* filtered_container;
	convolutions() {}
	convolutions(int output_length, int input_length, conv_size format, POOLER pr, ACTIVATION f) {

		this->create(output_length, input_length, format, pr, f);
	}
	void create(int output_length, int input_length, conv_size format, POOLER pr, ACTIVATION f) {

		u_flag = CONVOLUSIONS;
		if (pr == NO_POOLER) {
			if (!format.isgood(output_length, input_length, false)) {
				cout << "convolution_size_not_matched" << endl;
			}
		}
		else {
			if (!format.isgood(output_length, input_length, true)) {
				cout << "convolution_size_not_matched" << endl;
			}
		}
		output_size = output_length;
		input_size = input_length;

		if (f == SIGMOID) {
			activator = new Sigmoid();
		}
		else if (f == RELU) {
			activator = new Relu();
		}
		else if (f == TANH) {
			activator = new Tanh();
		}
		else if (f == SOFTMAX) {
			activator = new Softmax();
		}
		else if (f == IDENTITY) {
			activator = new Identity();
		}
		else if (f == SWISH) {
			activator = new Swish();
		}
		input_width = format.width_and_height;

		if (pr == NO_POOLER) {
			output_width = input_width;
		}

		else {
			output_width = input_width / 2;
			if (input_width % 2 == 1) {
				output_width++;
			}
		}

		if (pr == AVG_POOLER) {
			pooler = new AvgPooler(format.n_dimension, input_width);
		}
		else if (pr == MAX_POOLER) {
			pooler = new MaxPooler(format.n_dimension, input_width);
		}
		else if(pr == NO_POOLER){
			pooler = new NoPooler();
		}
		n_channel = format.n_channel;
		filter_width = 3;
		filt_tot_len = format.n_channel * filter_width * filter_width;
		n_dimension = format.n_dimension;
		in_tot_len = input_width * input_width*n_channel;
		output_tot_len = output_width * output_width*n_dimension;

		filter = new double*[n_dimension];
		gradients = new double*[n_dimension];
		filtered_container = new double[input_width*input_width*n_dimension];
		Input_container = new double[in_tot_len];
		Output_container = new double[output_tot_len];
		forward_pass = new double[output_tot_len];
		back_pass = new double[in_tot_len];
		for (int d = 0; d < n_dimension; d++) {
			filter[d] = new double[filt_tot_len];
			gradients[d] = new double[filt_tot_len];
			km::fill_gaussian_dist(filter[d], filt_tot_len);
		}
	}

	void load_input(double* in) {
		for (int i = 0; i < in_tot_len; i++) {
			Input_container[i] = in[i];
		}
	}

	void update_preparation() {
		for (int d = 0; d < n_dimension; d++) {
			for (int i = 0; i < filt_tot_len; i++) {
				gradients[d][i] = 0.0;
			}
		}
	}

	void forward_propagation(double* forward) {
		load_input(forward);
		km::fill(filtered_container, 0.0, input_width*input_width*n_dimension);
		for (int d = 0; d < n_dimension; d++) {

			for (int c = 0; c < n_channel; c++) {
				for (int i = 0; i < input_width; i++) {
					for (int j = 0; j < input_width; j++) {
						for (int p = 0; p < filter_width; p++) {
							if (i + p - 1 < 0 || i + p - 1 >= input_width) {
								continue;
							}
							for (int q = 0; q < filter_width; q++) {
								if (j + q - 1 < 0 || j + q - 1 >= input_width) {
									continue;
								}
								filtered_container[d* input_width* input_width + i * input_width + j] += filter[d][c*filter_width*filter_width + p * filter_width + q] *
									Input_container[input_width*input_width*c + (i + p - 1)*input_width + (j + q - 1)];
							}
						}
					}
				}
			}
		}

		if (pooler->model== NO_POOLER) {
			km::copy(Output_container, filtered_container, output_tot_len);

		}

		else {
			pooler->apply(Output_container, filtered_container);
		}

		activator->apply(forward_pass, Output_container, output_tot_len);
	}
	void accumulate_GD() {

		for (int d = 0; d < n_dimension; d++) {

			for (int c = 0; c < n_channel; c++) {
				for (int i = 0; i < input_width; i++) {
					for (int j = 0; j < input_width; j++) {
						for (int p = 0; p < filter_width; p++) {
							if (i + p - 1 < 0 || i + p - 1 >= input_width) {
								continue;
							}
							for (int q = 0; q < filter_width; q++) {
								if (j + q - 1 < 0 || j + q - 1 >= input_width) {
									continue;
								}
								gradients[d][c*filter_width*filter_width + p * filter_width + q] +=
									Input_container[input_width*input_width*c + (i + p - 1)*input_width + (j + q - 1)]
									* filtered_container[d* input_width* input_width + i * input_width + j];

							}
						}
					}
				}
			}
		}
	}

	void pass_delta() {
		km::fill(back_pass, 0.0, in_tot_len);
		for (int d = 0; d < n_dimension; d++) {
			for (int c = 0; c < n_channel; c++) {
				for (int i = 0; i < input_width; i++) {
					for (int j = 0; j < input_width; j++) {
						for (int p = 0; p < filter_width; p++) {
							if (i + p - 1 < 0 || i + p - 1 >= input_width) {
								continue;
							}
							for (int q = 0; q < filter_width; q++) {
								if (j + q - 1 < 0 || j + q - 1 >= input_width) {
									continue;
								}
								back_pass[input_width*input_width*c + (i + p - 1)*input_width + (j + q - 1)] +=
									filter[d][c*filter_width*filter_width + p * filter_width + q]
									* filtered_container[d* input_width* input_width + i * input_width + j];
							}
						}
					}
				}
			}
		}
	}

	void backpropagation(double* delta) {

		activator->feedback(Output_container, output_tot_len);
		for (int i = 0; i < output_tot_len; i++) {
			Output_container[i] *= delta[i];
		}

		if (pooler->model == NO_POOLER) {
			km::copy(filtered_container, Output_container, input_width*input_width*n_dimension);

		}

		else {
			pooler->feedback(filtered_container, Output_container);
		}

		accumulate_GD();
		pass_delta();
	}
	void backward_pass(double* delta) {

		activator->feedback(Output_container, output_tot_len);
		for (int i = 0; i < output_tot_len; i++) {
			Output_container[i] *= delta[i];
		}

		if (pooler->model == NO_POOLER) {
			km::copy(filtered_container, Output_container, input_width*input_width*n_dimension);

		}

		else {
			pooler->feedback(filtered_container, Output_container);
		}
		pass_delta();
	}
	void update(double learning_rate) {
		for (int d = 0; d < n_dimension; d++) {
			for (int i = 0; i < filt_tot_len; i++) {

				filter[d][i] -= gradients[d][i] * learning_rate;

			}
		}
	}

	~convolutions() {
		delete[] back_pass;
		delete[] forward_pass;
		delete[] Output_container;
		delete[] Input_container;
		delete[] filtered_container;
		for (int d = 0; d < n_dimension; d++) {
			delete[] filter[d];
			delete[] gradients[d];
		}
		delete activator;
		delete pooler;
	}
};


class DataSet {
public:
	int n_rows;
	int n_cols;
	int n_class;
	double** data;
	double** label_onehot_vector;
	DataSet() {}
	DataSet(string path) {
		ifstream fin(path);
		if (fin.fail()) {
			cout << "file opening failure" << endl;
		}
		else {
			n_rows = 0;
			n_cols = 0;
			n_class = 10;
			while (fin.good()) {
				string line;
				getline(fin, line);
				if (!fin.good()) {
					break;
				}

				if (n_rows == 0) {
					for (int i = 0; i < line.length(); i++) {
						if (line[i] == ',') {
							n_cols++;
						}
					}
				}
				n_rows++;
			}
			fin.close();
			fin.open(path);
			data = new double*[n_rows];
			label_onehot_vector = new double*[n_rows];

			int LINE = 0;
			while (fin.good()) {
				string line;
				getline(fin, line);
				if (!fin.good()) {
					break;
				}
				data[LINE] = new double[n_cols];
				label_onehot_vector[LINE] = new double[n_class];
				km::fill(label_onehot_vector[LINE], 0.0, n_class);
				string value_buffer;

				int count = -1;
				for (int i = 0; i <= (int)line.length(); i++) {
					if (line[i] == ',' || i == line.length()) {
						if (count == -1) {
							int idx = atoi(value_buffer.c_str());
							label_onehot_vector[LINE][idx] = 1.0;
							value_buffer.clear();
						}
						else {
							data[LINE][count] = atof(value_buffer.c_str());
							value_buffer.clear();
						}
						count++;
					}
					else {
						value_buffer.push_back(line[i]);
					}
				}
				LINE++;
			}
		}
		fin.close();
	}
	void scaling(ACTIVATION A) {
		if (A == SIGMOID) {
			for (int n = 0; n < n_rows; n++) {
				for (int i = 0; i < 28; i++) {
					for (int j = 0; j < 28; j++) {
						data[n][i * 28 + j] = (data[n][i * 28 + j] / 255);
					}
				}
			}
		}
		else if(A == TANH) {
			for (int n = 0; n < n_rows; n++) {
				for (int i = 0; i < 28; i++) {
					for (int j = 0; j < 28; j++) {
						data[n][i * 28 + j] = (data[n][i * 28 + j] / 128)-1.0;
					}
				}
			}
		}
	}
	void print() {
		for (int n = 0; n < n_rows; n++) {
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					if (data[n][i * 28 + j] > 0.1) {
						cout << "■";
					}
					else {
						cout << "□";
					}
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}

	~DataSet() {
		for (int i = 0; i < n_rows; i++) {
			delete[] data[i];
			delete[] label_onehot_vector[i];
		}
	}
};
void print_image(double* image, ACTIVATION A) {
	if (A == SIGMOID) {
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				if (image[i * 28 + j] > 0.05) {
					cout << "■";
				}
				else {
					cout << "□";
				}
			}
			cout << endl;
		}
		cout << endl;
	}
	else if(A == TANH){
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				if (image[i * 28 + j] > -0.9) {
					cout << "■";
				}
				else {
					cout << "□";
				}
			}
			cout << endl;
		}
		cout << endl;
	}
}

class multi_layer_net {
public:

	unit** layer;
	int n_layer;
	int end_layer;
	double* output;
	multi_layer_net() {
	}
	multi_layer_net(int N_layer) {
		n_layer = N_layer;
		end_layer = n_layer - 1;
		layer = new unit*[n_layer];
	}
	multi_layer_net(string path) {
		ifstream fin;
		fin.open(path);
		if (fin.fail()) {
			cout << "file " << path << "can not be opened" << endl;
		}
		else {
			string line;
			getline(fin, line);
			this->n_layer = atoi(line.c_str());
			end_layer = n_layer - 1;
			layer = new unit*[n_layer];
			for (int i = 0; i < n_layer; i++) {
				getline(fin, line);
				UNIT unit = (UNIT)atoi(line.c_str());
				if (unit == PERCEPTRONS) {
					getline(fin, line);
					ACTIVATION A = (ACTIVATION)atoi(line.c_str());
					getline(fin, line);
					int in_sz = atoi(line.c_str());
					getline(fin, line);
					int out_sz = atoi(line.c_str());
					this->layer[i] = new perceptrons(out_sz, in_sz, A);
					perceptrons* p_ptr = (perceptrons*)layer[i];
					for (int w = 0; w < p_ptr->height * p_ptr->width; w++) {
						getline(fin, line);
						p_ptr->weight[w] = atof(line.c_str());
					}
					p_ptr = NULL;
				}
				if (unit == CONVOLUSIONS) {
					getline(fin, line);
					ACTIVATION A = (ACTIVATION)atoi(line.c_str());
					getline(fin, line);
					POOLER P = (POOLER)atoi(line.c_str());
					getline(fin, line);
					int in_sz = atoi(line.c_str());
					getline(fin, line);
					int out_sz = atoi(line.c_str());
					getline(fin, line);
					int n_chan = atoi(line.c_str());
					getline(fin, line);
					int in_wd = atoi(line.c_str());
					getline(fin, line);
					int n_dim = atoi(line.c_str());
					layer[i] = new convolutions(out_sz, in_sz, conv_size(n_chan, in_wd, n_dim), P, A);
					convolutions* c_ptr = (convolutions*)layer[i];
					for (int d = 0; d < c_ptr->n_dimension; d++) {
						for (int w = 0; w < c_ptr->filter_width*c_ptr->filter_width*c_ptr->n_channel; w++) {
							getline(fin, line);
							c_ptr->filter[d][w] = atof(line.c_str());
						}
					}
					c_ptr = NULL;

				}
			}

		}
		fin.close();
	}
	void update(double learning_rate) {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->update(learning_rate);
		}
	}

	void predict(double* e) {
		layer[0]->forward_propagation(e);
		for (int i = 1; i < n_layer; i++) {
			layer[i]->forward_propagation(layer[i - 1]->forward_pass);
		}
		output = layer[end_layer]->forward_pass;
	}

	void update_preparation() {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->update_preparation();
		}
	}

	void learn(DataSet& instance, int epoch, double learning_rate) {
		for (int iter = 0; iter < epoch; iter++) {
			update_preparation();
			for (int m = 0; m < instance.n_rows; m++) {
				predict(instance.data[m]);
				km::softmax_cross_entropy_derivative(layer[end_layer]->Output_container, layer[end_layer]->forward_pass, instance.label_onehot_vector[m], layer[end_layer]->output_size);
				layer[end_layer]->accumulate_GD();
				layer[end_layer]->pass_delta();
				for (int i = n_layer - 2; i >= 0; i--) {
					layer[i]->backpropagation(layer[i + 1]->back_pass);
				}
			}
			learning_rate /= (double)instance.n_rows;
			update(learning_rate);
		}
	}
	void record_to_file(string path) {
		ofstream file_out;
		file_out.open(path);
		file_out << n_layer << endl;
		for (int i = 0; i < n_layer; i++) {
			file_out << layer[i]->u_flag << endl;
			if (layer[i]->u_flag == PERCEPTRONS) {
				perceptrons* p_ptr = (perceptrons*)layer[i];
				file_out << p_ptr->activator->function << endl;
				file_out << p_ptr->input_size << endl;
				file_out << p_ptr->output_size << endl;
				for (int w = 0; w < p_ptr->height * p_ptr->width; w++) {
					file_out << p_ptr->weight[w] << endl;
				}
				p_ptr = NULL;

			}
			else if (layer[i]->u_flag == CONVOLUSIONS) {
				convolutions* c_ptr = (convolutions*)layer[i];
				file_out << c_ptr->activator->function << endl;
				file_out << c_ptr->pooler->model << endl;
				file_out << c_ptr->input_size << endl;
				file_out << c_ptr->output_size << endl;
				file_out << c_ptr->n_channel << endl;
				file_out << c_ptr->input_width << endl;
				file_out << c_ptr->n_dimension << endl;
				for (int d = 0; d < c_ptr->n_dimension; d++) {
					for (int w = 0; w < c_ptr->filter_width*c_ptr->filter_width*c_ptr->n_channel; w++) {
						file_out << c_ptr->filter[d][w] << endl;
					}
				}
				c_ptr = NULL;
			}
		}
		cout << "recorded at " << path << endl;
		file_out.close();
	}
	void show_model_info() {
		cout << "total layers :" << n_layer << endl;
		for (int i = 0; i < n_layer; i++) {
			if (layer[i]->u_flag == PERCEPTRONS) {
				cout << "LAYER[" << i << "] = ";
				cout << "PERCEPTRONS(output_len:" << layer[i]->output_size;
				cout << ",input_len:" << layer[i]->input_size;
				cout << ",ACTIVATION:"; layer[i]->activator->print_what_it_is();
				cout << ")" << endl;
			}
			else if (layer[i]->u_flag == CONVOLUSIONS) {
				cout << "LAYER[" << i << "] = ";
				cout << "CONVOLUSIONS(output_len:" << layer[i]->output_size;
				cout << ",input_len:" << layer[i]->input_size;
				convolutions* p_ptr = (convolutions*)layer[i];
				cout << ",conv_size(" << p_ptr->n_channel << "," << p_ptr->input_width << "," << p_ptr->n_dimension << ")";
				cout << ",POOLER:"; p_ptr->pooler->print_what_it_is();
				cout << ",ACTIVATION:"; layer[i]->activator->print_what_it_is();
				cout << ")" << endl;
			}
		}
	}
	~multi_layer_net() {
		for (int i = 0; i < n_layer; i++) {
			delete layer[i];
		}
	}
};

/*
 *  created by kimin jeong, chungang university, south korea.
 *  2019-01-20
 */
int main() {
	DataSet image("C:\\AI\\MNIST_SET10.csv"); //MNIST file
	image.scaling(SIGMOID); //SIGMOID = (0 ~255) -> (0 ~ 1a), TANH = (0 ~ 255) -> (-1 ~ 1)
	
	multi_layer_net conv_net(6); // <- number of layers
	/*
	 *  convolution(output_len, input_len,conv_size(channel,input_width(=height),dimension),pooler,activation_function)
	 *  filter, stride : filter_width = 3, stride = 1 고정입니다.
	 *  pooler : avg pooler , max pooler , no pooler를 지원합니다. stride of pooler = 2 고정입니다.
	 *  pooler 적용 시 : input_width 가 짝수일때 output_width == input_width/2. 홀수일떄 output_width == (input_width/2)+1
	 *  pooler 미 적용시 : input_width = output_width
	 *  
	 */

	/*
	 * perceptron(output_len, input_len, activation_function)
	 * bias가 적용됩니다.
	 */

	conv_net.layer[0] = new convolutions(392,784, conv_size(1, 28, 2), MAX_POOLER, RELU); 
	conv_net.layer[1] = new convolutions(196, 392, conv_size(2, 14, 4), MAX_POOLER, RELU);
	conv_net.layer[2] = new convolutions(128, 196, conv_size(4, 7, 8), MAX_POOLER, RELU);
	conv_net.layer[3] = new convolutions(64, 128, conv_size(8, 4, 16), MAX_POOLER, RELU);
	conv_net.layer[4] = new convolutions(64, 64, conv_size(16, 2, 64), MAX_POOLER, RELU);
	conv_net.layer[5] = new perceptrons(10, 64, SOFTMAX);

	/*******저장된 모델을 불러올 경우*******/
	//multi_layer_net conv_net("conv.txt");
	/*************************************/

	conv_net.show_model_info();
	double mean = 0;
	double loss = 0;
	for (int i = 0; i < image.n_rows; i++) {
		conv_net.predict(image.data[i]);
		loss = km::softmax_loss(conv_net.output, image.label_onehot_vector[i], 10);
		mean += loss;
	}
	cout << "mean_loss : " << mean / (double)image.n_rows << endl;
	cout << "--------------------------------------" << endl;
	int c;
	for (int ITER = 0; ITER < 100; ITER++) {
		conv_net.learn(image, 100, 0.05);
		mean = 0;
		loss = 0;
		for (int i = 0; i < image.n_rows; i++) {
			conv_net.predict(image.data[i]);
			loss = km::softmax_loss(conv_net.output, image.label_onehot_vector[i], 10);
			mean += loss;
		}
		cout << "mean_loss : " << mean / (double)image.n_rows << endl;
		cout << "--------------------------------------" << endl;
		cout<<"quit : 0, store : 1, learns more : 2 >>";
		cin >> c;
		if (c == 0) {
			break;
		}
		else if (c == 1) {
			string fname;
			cout << "file name : ";
			cin >> fname;
			fname += ".txt";
			conv_net.record_to_file(fname);
			break;
		}
	}	
}
