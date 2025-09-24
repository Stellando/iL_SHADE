#include <vector>
#include <random>

void cec14_test_func(double *x, double *f, int nx, int mx, int func_num);
double ackley(const std::vector<double>& x);
double sphere_func (const std::vector<double>& x);
double rastrigin(const std::vector<double>& x);
double rosenbrock(const std::vector<double>& x);
double griewank(const std::vector<double>& x);
std::vector<double> generateRandomIndividual(int D, double min, double max, std::mt19937& gen);
// functions.h
double cec14_wrapper(const std::vector<double>& x, int func_num);
double meanWL(const std::vector<double>& S, const std::vector<double>& F_delta);
int N_G(int N_init, int N_min, int MAX_NFE, int NFE);
double meanA(const std::vector<double>& S);
double meanL(const std::vector<double>& S);
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
