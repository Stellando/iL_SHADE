#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <random>
#include <functional>
#include <string>

double ackley(const std::vector<double>& x);
std::vector<double> generateRandomIndividual(int D, double min, double max, std::mt19937& gen);

// CEC21 函數選擇器
std::function<double(const std::vector<double>&)> get_cec21_function(int cec21_type, int func_num);
std::string get_cec21_function_name(int cec21_type, int func_num);

// 輔助函數
double meanWL(const std::vector<double>& S, const std::vector<double>& F_delta);
int N_G(int N_init, int N_min, int MAX_NFE, int NFE);
double meanA(const std::vector<double>& S);
double meanL(const std::vector<double>& S);

// 基本測試函數 F1-F13
double F1(const std::vector<double>& x);
double F2(const std::vector<double>& x);
double F3(const std::vector<double>& x);
double F4(const std::vector<double>& x);
double F5(const std::vector<double>& x);
double F6(const std::vector<double>& x);
double F7(const std::vector<double>& x);
double F8(const std::vector<double>& x);
double F9(const std::vector<double>& x);
double F10(const std::vector<double>& x);
double F11(const std::vector<double>& x);
double F12(const std::vector<double>& x);
double F13(const std::vector<double>& x);

// CEC21 function declarations
extern "C" {
    void cec21_basic_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_bias_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_bias_rot_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_bias_shift_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_bias_shift_rot_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_rot_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_shift_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_shift_rot_func(double *x, double *f, int nx, int mx, int func_num);
}

#endif // FUNCTIONS_H
