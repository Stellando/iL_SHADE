#include "functions.h"
#include <cmath>
#include <algorithm>
#include <random> // 確保使用隨機數生成器
#include <iostream>
#define M_PI 3.14159265358979323846
using namespace std;

extern double *OShift, *M, *y, *z, *x_bound;
extern int ini_flag, n_flag, func_flag, *SS;
// functions.cpp
double cec14_wrapper(const vector<double>& x, int func_num) {
    int nx = x.size();
    double fx = 0.0;
    vector<double> x_copy = x; // 若 cec14_test_func 會改動 x
    cec14_test_func(x_copy.data(), &fx, nx, 1, func_num);
    return fx;
}

double ackley(const vector<double>& x) {
    const double a = 20.0, b = 0.2, c = 2 * M_PI;
    int D = x.size();
    double sum1 = 0.0, sum2 = 0.0;
    for (double val : x) {
        sum1 += val * val;
        sum2 += cos(c * val);
    }
    return -a * exp(-b * sqrt(sum1 / D)) - exp(sum2 / D) + a + exp(1.0);
}

vector<double> generateRandomIndividual(int D, double min, double max, mt19937& gen) {
    vector<double> individual(D);
    uniform_real_distribution<> dis(min, max);
    for (int j = 0; j < D; ++j)
        individual[j] = dis(gen);
    return individual;
}

//用來計算新的平均F,CR
double meanWL(const vector<double>& S, const vector<double>& F_delta) {
    if (S.size() != F_delta.size() || S.empty()) return 0.0;
    double sum_delta = 0.0;
    for (double d : F_delta) sum_delta += d;
    if (sum_delta == 0.0) return 0.0;

    double numerator = 0.0, denominator = 0.0;
    for (size_t k = 0; k < S.size(); ++k) {
        double wk = F_delta[k] / sum_delta;
        numerator += wk * S[k] * S[k];
        denominator += wk * S[k];
    }
    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}
int N_G(int N_init, int N_min, int MAX_NFE, int NFE) {
    double ratio = static_cast<double>(N_min - N_init) / MAX_NFE;
    double val = ratio * NFE + N_init;
    return static_cast<int>(round(val));
}
double meanA(const vector<double>& S) {
    if (S.empty()) return 0.0;
    double sum = 0.0;
    for (double val : S) sum += val;
    return sum / S.size();
}

double meanL(const vector<double>& S) {
    if (S.empty()) return 0.0;
    double sumF = 0.0, sumF2 = 0.0;
    for (double f : S) {
        sumF += f;
        sumF2 += f * f;
    }
    return sumF2 / sumF;
}
double F1(const vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
}
double F2(const vector<double> &position){
    double sum = 0.0, product = 1.0;
    for(double x : position){
        sum += fabs(x);
        product *= fabs(x);
    }
    double answer = sum + product;
    return answer;
}
//和JADE接近
double F3(const vector<double>& x) {
    double sum = 0.0;
    for(int i = 0; i < x.size(); ++i) {
        double term1 = 0.0;
        for (int j = 0; j < i; ++j) {
            term1 += x[j];
        }
        term1 *= term1;
        sum += term1;
    }
    return sum;
}
//和JADE接近
double F4(const vector<double> &position){
    double answer = fabs(position[0]);
    for(double x : position){
        if(fabs(x) > answer)
            answer = fabs(x);
    }
    return answer;
}
//0
double F5(const vector<double>& x) {
    double sum = 0.0;
    int D = x.size();
    for (int i = 0; i < D - 1; ++i) {
        double term1 = 100 * pow((x[i + 1] - x[i] * x[i]), 2);
        double term2 = pow((x[i] - 1), 2);
        sum += term1 + term2;
    }
    return sum;
}
double F6(const vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += pow(fabs(val + 0.5), 2);
    }
    return sum;
}
//7怪怪的 
double F7(const vector<double>& x) {
    double sum = 0.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> rand01(0.0, 1.0); // 隨機生成 [0, 1]

    for (int i = 0; i < x.size(); ++i) {
        sum += pow(x[i], 4) * (i + 1) ;
    }
    return sum   + rand01(gen);
}
double F8(const vector<double>& position) {
        double answer = 0.0;
        for(double x : position){
            answer += (-x * sin(sqrt(fabs(x))));
        }
        answer += (418.98288727243369 * position.size());
        return answer;
}
double F9(const vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += pow(val, 2) - 10 * cos(2 * M_PI * val) + 10;
    }
    return sum;
}
double F10(const vector<double>& x) {
    const double a = 20.0, b = 0.2, c = 2 * M_PI;
    int D = x.size();
    double sum1 = 0.0, sum2 = 0.0;
    for (double val : x) {
        sum1 += val * val;
        sum2 += cos(c * val);
    }
    return -a * exp(-b * sqrt(sum1 / D)) - exp(sum2 / D) + a + exp(1.0);
}
double F11(const vector<double> &position){
    double sum = 0.0, product = 1.0;
    for(size_t i = 0; i < position.size(); i++){
        sum += pow(position[i], 2);
        product *= cos(position[i] / sqrt(i + 1));
    }
    sum /= 4000.0;
    double answer = sum - product + 1;
    return answer;
}
double F12(const vector<double> &position){
    int D = position.size();
    int a = 10, k = 100, m = 4;
    vector<double> y(D);
    for(int i = 0; i < D; i++)
        y[i] = 1 + ((position[i] + 1) / 4);

    double answer = 10 * pow(sin(M_PI * y[0]), 2);
    for(int i = 0; i < D - 1; i++){
        answer += (pow(y[i] - 1, 2) * (1 + 10 * pow(sin(M_PI * y[i + 1]), 2)));
    }
    answer += pow(y[D - 1] - 1, 2);
    answer = ((answer * M_PI) / D);

    for(double x : position){
        if(x > a)
            answer += (k * pow((x - a), m));
        else if(x >= -a && x <= a)
            answer += 0;
        else
            answer += (k * pow((-x - a), m));
    }
    return answer;
}
double F13(const vector<double> &position){
    int D = position.size();
    int a = 5, k = 100, m = 4;
    double answer = pow(sin(3 * M_PI * position[0]), 2);
    for (int i = 0; i < D - 1; ++i){
        answer += pow((position[i] - 1), 2) * (1 + pow(sin(3 * M_PI * position[i + 1]), 2));
    }
    answer += (pow((position[D - 1] - 1), 2) * (1 + pow(sin(2 * M_PI * position[D - 1]), 2)));
    answer *= 0.1;
    for (double x : position) {
        if(x > a)
            answer += (k * pow((x - a), m));
        else if(x >= -a && x <= a)
            answer += 0;
        else
            answer += (k * pow((-x - a), m));
    }
    return answer;
}