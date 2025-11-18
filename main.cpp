#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include "algorithm.h"
#include "functions.h"

using namespace std;

// ========== 主程式 ==========
int main(int argc, char *argv[]) {
    // ==================== 參數設定區 ====================
    // 這裡可以輕鬆調整所有參數
    
    // ========== 函數選擇開關 ==========
    // true: 使用 Ackley 函數
    // false: 使用 CEC21 函數
    bool use_ackley = false;
    // ==================================
    
    int D = 10;              // 維度(2 10 20)
    int NP = D * 18;         // 種群大小 (建議: 18*D)
    int maxNFE = D * 10000;  // 最大函數評估次數 (建議: D*10000)
    double pb = 0.11;        // p-best比例下界
    double c = 0.1;          // 參數適應率
    double maxVal = 100.0;  // Ackley 搜索空間上界 (標準範圍: [-32.768, 32.768])
    double minVal = -maxVal; // 搜索空間下界
    int runs = 1;           // 獨立運行次數
    bool verbose = false;    // 是否顯示詳細信息
    bool enableLog = true;   // 是否啟用收斂日誌
    bool enableOutputFiles = false;  // 是否生成輸出檔案 (txt 檔)
    
    // CEC21 函數設定 (當 use_ackley = false 時使用)
    // 1: basic, 2: bias, 3: bias_rot, 4: bias_shift, 
    // 5: bias_shift_rot, 6: rot, 7: shift, 8: shift_rot
    int cec21_type = 1;      // 選擇 CEC21 函數類型
    int func_start = 1;      // 起始函數編號 (1-10)
    int func_end = 5;       // 結束函數編號 (1-10)
    
    // ==================== 參數設定區結束 ====================
    
    // 遍歷多個 CEC21 函數
    for (int func_num = func_start; func_num <= func_end; ++func_num) {
        // 根據開關選擇函數
        function<double(const vector<double>&)> target_func;
        string func_name;
        
        if (use_ackley) {
            // 使用 Ackley 函數
            target_func = ackley;
            func_name = "Ackley";
        } else {
            // 使用 CEC21 函數
            target_func = get_cec21_function(cec21_type, func_num);
            func_name = get_cec21_function_name(cec21_type, func_num);
        }
        
        // 輸出配置信息
        cout << "\n========== iL-SHADE Algorithm Configuration ==========\n";
        cout << "Dimension (D)     : " << D << "\n";
        cout << "Population (NP)   : " << NP << "\n";
        cout << "Max NFE           : " << maxNFE << "\n";
        cout << "Search Range      : [" << minVal << ", " << maxVal << "]\n";
        cout << "p-best (pb)       : " << pb << "\n";
        cout << "Adapt rate (c)    : " << c << "\n";
        cout << "Runs              : " << runs << "\n";
        cout << "Function Type     : " << func_name << "\n";
        cout << "Logging           : " << (enableLog ? "enabled" : "disabled") << "\n";
        cout << "Output Files      : " << (enableOutputFiles ? "enabled" : "disabled") << "\n";
        cout << "======================================================\n\n";
        
        // 執行多次獨立運行
        vector<double> fitness_per_run;
        fitness_per_run.reserve(runs);
    
    vector<double> best_solution;
    double best_fitness = numeric_limits<double>::infinity();
    int best_run_index = -1;
    
    for (int run = 0; run < runs; ++run) {
        string log_path;
        if (enableLog && enableOutputFiles) {
            log_path = func_name;
            if (runs > 1) log_path += "_run" + to_string(run + 1);
            log_path += ".txt";
        }
        
        // 執行算法
        vector<double> solution = differential_evolution(
            D, NP, maxNFE, pb, c, minVal, maxVal,
            target_func, log_path, verbose, 0
        );
        
        double fitness = target_func(solution);
        fitness_per_run.push_back(fitness);
        
        cout << "Run " << (run + 1) << "/" << runs 
             << " => Fitness = " << scientific << setprecision(6) << fitness;
        if (!log_path.empty()) cout << " (log: " << log_path << ")";
        cout << endl;
        
        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_solution = solution;
            best_run_index = run;
        }
    }
    
    // 統計結果
    double mean_fitness = accumulate(fitness_per_run.begin(), fitness_per_run.end(), 0.0) / fitness_per_run.size();
    double stddev_fitness = 0.0;
    
    if (fitness_per_run.size() > 1) {
        for (double f : fitness_per_run) {
            double diff = f - mean_fitness;
            stddev_fitness += diff * diff;
        }
        stddev_fitness = sqrt(stddev_fitness / (fitness_per_run.size() - 1));
    }
    
    cout << "\n========== Results Summary ==========\n";
    cout << "Function          : " << func_name << "\n";
    cout << "Best Fitness      : " << scientific << setprecision(10) << best_fitness 
         << " (Run " << (best_run_index + 1) << ")\n";
    cout << "Mean Fitness      : " << scientific << setprecision(10) << mean_fitness << "\n";
    if (fitness_per_run.size() > 1) {
        cout << "Std. Deviation    : " << scientific << setprecision(10) << stddev_fitness << "\n";
    }
    cout << "=====================================\n";
    
    // 將結果寫入檔案
    if (enableOutputFiles) {
        ofstream result_file(func_name + "_summary.txt");
        if (result_file.is_open()) {
            result_file << "Function: " << func_name << "\n";
            result_file << "Dimension: " << D << "\n";
            result_file << "Runs: " << runs << "\n\n";
            result_file << "Best Fitness: " << scientific << setprecision(10) << best_fitness << " (Run " << (best_run_index + 1) << ")\n";
            result_file << "Mean Fitness: " << mean_fitness << "\n";
            if (fitness_per_run.size() > 1) {
                result_file << "Std. Deviation: " << stddev_fitness << "\n";
            }
            result_file << "\nAll Run Results:\n";
            for (size_t i = 0; i < fitness_per_run.size(); ++i) {
                result_file << "Run " << (i + 1) << ": " << fitness_per_run[i] << "\n";
            }
            result_file.close();
            cout << "Summary saved to: " << func_name << "_summary.txt\n";
        }
    }
    
    } // 結束 func_num 迴圈
    
    cout << "\n========== All Functions Completed ==========\n";
    system("pause");
    return 0;
}