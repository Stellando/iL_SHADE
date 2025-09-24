#include <iostream>
#include <vector>
#include "algorithm.h"
#include "functions.h"
using namespace std;

int main(int argc, char *argv[]) {

    int D = 10; // 維度
    int NP = 100; // 種群大小
    int G = 500; // 迭代次數
    double pb = 0.05; //取前幾%的個體
    double c = 0.1; //自適應參數
    int maxVal = 30; // 解空間範圍 [-maxVal, maxVal]
    int func_num = 5; // 選擇測試函數編號

    /*
    double *OShift,*M,*y,*z,*x_bound;
    int ini_flag=0,n_flag,func_flag,*SS;
    */
    
    cout << "Initializing parameters:\n";
    cout << "D: " << D << ", NP: " << NP << ", G: " << G << ", p: " << pb << ", c: " << c << "\n";
    string function_names[] = {"ackley", "sphere_func", "rastrigin", "rosenbrock", "griewank"};

    for (int  i = 0; i < sizeof(function_names) / sizeof(function_names[0]); i++)
    {
        algorithm alg;
        alg.RunALG(D, NP, G, pb, c, maxVal , i + 1);
        int idx;
        cout << "function: " << function_names[i] << endl;  
        cout << "Best fitness: " << alg.get_best_fitness(idx) << endl;
        cout << "Position: ";
        vector<double> best_pos = alg.get_best_position();
        for (double val : best_pos) {
            cout << val << " ";
        }
        cout << endl;
    }
    



    system("pause");
    return 0;
}