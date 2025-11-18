#include <vector>
#include <functional>
#include <random>
#include <string>
#include <ctime>
using namespace std;
// 傳入目標函數（統一使用加權 Lehmer 更新 CR 與 F）
vector<double> differential_evolution(
    int D, int NP, int maxNFE, double p, double c, double minVal, double maxVal,
    function<double(const vector<double>&)> func,
    const string& logFilePath = "",
    bool verbose = false,
    mt19937::result_type seed = 0
);
void choose_pbest_and_r1r2(
    int i, int NP, int D, double p_i,
    const vector<double>& fitness,
    const vector<vector<double>>& P,
    const vector<vector<double>>& Archive,
    mt19937& gen,
    uniform_int_distribution<>& randNP,
    uniform_real_distribution<>& rand01,
    int& pBestIdx, int& r1, vector<double>& xr2
);
vector<double> mutation(const vector<vector<double>>&, double, int, int, int, const vector<double>&, double, double, int);
vector<double> crossover(const vector<double>&, const vector<double>&, double, int, mt19937&, uniform_real_distribution<>&);