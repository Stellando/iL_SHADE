#include "algorithm.h"
#include "functions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

using namespace std;
vector<double> differential_evolution(
    int D, int NP, int maxNFE, double pb, double c, double minVal, double maxVal,
    function<double(const vector<double>&)> f,
    const string& logFilePath,
    bool verbose,
    mt19937::result_type seed
) {
    // Initialization phase
    const int NP0 = NP;             // 保存初始種群大小供縮減與 Archive 限制使用
    const int Nmin = 4;             // 最小種群 (L-SHADE 標準)
    int H = min(NP, 100);           // iL-SHADE: H = min(NP, 100)
    vector<vector<double>> P(NP);   // Population
    vector<double> MCR(H, 0.9), MF(H, 0.9); // iL-SHADE 初始化為 0.9
    vector<vector<double>> Archive;        // 歷史存檔 (|A| <= NP0)
    int k = 0; // 0-based index counter for memory slots
    mt19937::result_type actualSeed = seed;
    if (actualSeed == 0) {
        std::random_device rd;
        const auto nowSeed = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        actualSeed = static_cast<mt19937::result_type>(nowSeed ^ (static_cast<std::uint64_t>(rd()) << 1));
    }
    std::mt19937 gen(actualSeed);
    std::uniform_real_distribution<> rand01(0.0, 1.0);
    std::uniform_int_distribution<> randH(0, H-1); // 0..H-1
    std::uniform_int_distribution<> randNP(0, NP-1);
    
    // 初始化族群
    vector<double> fitness(NP);
    for (int i = 0; i < NP; ++i) {
        P[i] = generateRandomIndividual(D, minVal, maxVal, gen);
        fitness[i] = f(P[i]);
    }
    
    // Main loop
    int MAX_NFE = (maxNFE > 0) ? maxNFE : D * 10000;
    int NFE = NP;  // 已經評估了 NP 個個體
    int Ng = NP; // initial Ng = Ninit = NP
    int G = 1; // G=1
    
    // 檔案輸出設置 - 每隔300次evaluation記錄一次
    bool logging = !logFilePath.empty();
    ofstream logFile;
    if (logging) {
        logFile.open(logFilePath, ios::trunc);
        if (!logFile) {
            cerr << "Warning: unable to open log file: " << logFilePath << "\n";
            logging = false;
        } else {
            logFile << "NFE\tBest_Fitness\n";
        }
    }
    
    int record_interval = 300; // 每300次evaluation記錄一次
    int next_record_nfe = record_interval;
    // （iL-SHADE）使用 NFE 作為主要終止條件
    while (NFE < MAX_NFE) {
        vector<double> CR(NP), F(NP);
        vector<double> S_CR, S_F, S_delta;
    vector<vector<double>> newP(NP);
    vector<double> newFitness(NP, numeric_limits<double>::infinity());
    int processed = 0;
        // 找到當前最佳 fitness
        double current_best_fitness = *min_element(fitness.begin(), fitness.end());
        int bestIdx_before = min_element(fitness.begin(), fitness.end()) - fitness.begin();
        
        // 記錄到檔案 (每隔300次NFE)
        if (logging && NFE >= next_record_nfe) {
            logFile << NFE << "\t" << current_best_fitness << "\n";
            next_record_nfe += record_interval;
        }
        
        if (verbose) {
            cout << "Generation " << G << " (NFE: " << NFE << ", Best: " << current_best_fitness << ")" << endl;
        }
        // mutate & crossover
        for (int i = 0; i < NP; ++i) {
            // 確保不超過最大評估次數
            if (NFE >= MAX_NFE) break;
            
            int r = randH(gen); // 0-based index into memory
            // iL-SHADE: use progress ratio for parameter shaping
            double progress = (double)NFE / (double)MAX_NFE; // 0~1
            
            //CRi
            normal_distribution<> randCR(MCR[r], 0.1); 
            if (MCR[r] == -1.0) {
                CR[i] = 0.0;
            } else {
                CR[i] = min(1.0, max(0.0, randCR(gen)));
            }
            // ==== BEGIN iL-SHADE CR gating (progress-based lower bounds) ====
            // ref: if g < 0.25*Gmax -> CR = max(CR, 0.5)
            //      else if g < 0.5*Gmax -> CR = max(CR, 0.25)
            if (progress < 0.25) {
                CR[i] = max(CR[i], 0.5);
            } else if (progress < 0.5) {
                CR[i] = max(CR[i], 0.25);
            }
            // ==== END iL-SHADE CR gating ====
            // Fi
            cauchy_distribution<> randF(MF[r], 0.1);
            double Fi;
            do {
                Fi = randF(gen);
            } while (Fi <= 0.0);
            if (Fi > 1.0) Fi = 1.0;
            F[i] = Fi;
            // ==== BEGIN iL-SHADE F gating (progress-based upper bounds) ====
            // ref: if g < 0.25*Gmax -> F = min(F, 0.7)
            //      else if g < 0.5*Gmax -> F = min(F, 0.8)
            //      else if g < 0.75*Gmax -> F = min(F, 0.9)
            if (progress < 0.25) {
                F[i] = min(F[i], 0.7);
            } else if (progress < 0.5) {
                F[i] = min(F[i], 0.8);
            } else if (progress < 0.75) {
                F[i] = min(F[i], 0.9);
            }
            // ==== END iL-SHADE F gating ====
            // 線性 p (L-SHADE 標準風格)：p 從 p_max 線性遞減到 p_min
            const double p_max = 0.20;
            const double p_min = max(0.02, min(pb, p_max));
            double p_i = p_min + (p_max - p_min) * (1.0 - progress); // 早期較大，後期趨近 p_min

            //選擇最佳個體pb和變異個體r1r2
            int pBestIdx, r1;
            vector<double> xr2;
            choose_pbest_and_r1r2(i, NP, D, p_i, fitness, P, Archive, gen, randNP, rand01, pBestIdx, r1, xr2);

            // mutation
            vector<double> vi = mutation(P, F[i], pBestIdx, i, r1, xr2, minVal, maxVal, D);
            // crossover
            vector<double> ui = crossover(P[i], vi, CR[i], D, gen, rand01);
            newP[i] = ui;
            newFitness[i] = f(ui);
            NFE++;
            processed = i + 1;
        }
    
        // selection
        for (int i = 0; i < processed; ++i) {
            if (newFitness[i] <= fitness[i]) {
                if (newFitness[i] < fitness[i]) {
                    Archive.push_back(P[i]);
                    double improvement = fitness[i] - newFitness[i];
                    if (improvement > 0) {
                        S_delta.push_back(improvement);
                        S_CR.push_back(CR[i]);
                        S_F.push_back(F[i]);
                    }
                }
                P[i] = newP[i];
                fitness[i] = newFitness[i];
            }
        }

        // 控制 archive 大小 - 隨機移除多餘的個體
        while (Archive.size() > (size_t)NP0) {
            if (Archive.empty()) break;
            uniform_int_distribution<> randArchive(0, Archive.size() - 1);
            int eraseIdx = randArchive(gen);
            Archive.erase(Archive.begin() + eraseIdx);
        }
        // 更新 MCR, MF
        if (!S_delta.empty()) {
            double sum_w = accumulate(S_delta.begin(), S_delta.end(), 0.0);

            double new_mcr = -1.0;
            if (sum_w > 0.0 && !S_CR.empty()) {
                double arithCR = 0.0;
                for (size_t idx = 0; idx < S_CR.size(); ++idx) {
                    arithCR += (S_delta[idx] / sum_w) * S_CR[idx];
                }
                if (arithCR > 0.0) new_mcr = arithCR;
            }

            if (new_mcr == -1.0) {
                MCR[k] = -1.0;
            } else if (MCR[k] == -1.0) {
                MCR[k] = new_mcr;
            } else {
                MCR[k] = (1.0 - c) * MCR[k] + c * new_mcr;
            }

            double numF = 0.0;
            double denF = 0.0;
            for (size_t idx = 0; idx < S_F.size(); ++idx) {
                numF += S_delta[idx] * S_F[idx] * S_F[idx];
                denF += S_delta[idx] * S_F[idx];
            }
            double lehmerF = (denF == 0.0 ? 0.0 : numF / denF);
            if (lehmerF > 0.0) {
                MF[k] = (1.0 - c) * MF[k] + c * lehmerF;
            }

            k = (k + 1) % H;
        }
        
        // iL-SHADE LPSR (Linear Population Size Reduction) strategy
        int Ng_next = N_G(NP0, Nmin, MAX_NFE, NFE); // 基於初始 NP0 線性縮減
        int delta_NG = Ng_next - Ng;
        
        if (delta_NG < 0 && Ng > Nmin) {
            // 只挑出最差的個體並刪除：使用 nth_element 而非全排序以節省時間
            int remove_count = abs(delta_NG);
            if (remove_count > Ng - Nmin) remove_count = Ng - Nmin; // 至少保留 Nmin 個
            if (remove_count > 0 && remove_count < Ng) {
                vector<size_t> idx(P.size());
                iota(idx.begin(), idx.end(), 0);
                // 將最差的 remove_count 個體放到 idx 的尾端區塊（不保證排序）
                std::nth_element(
                    idx.begin(),
                    idx.end() - remove_count,
                    idx.end(),
                    [&](size_t a, size_t b){ return fitness[a] < fitness[b]; }
                );
                vector<size_t> remove_idx(idx.end() - remove_count, idx.end());
                // 逆序排序以便安全 erase（高索引先刪除）
                sort(remove_idx.rbegin(), remove_idx.rend());
                for (size_t r_idx : remove_idx) {
                    P.erase(P.begin() + r_idx);
                    fitness.erase(fitness.begin() + r_idx);
                }
                // 更新 Ng, NP
                Ng = (int)P.size();
                NP = Ng; // 當前實際族群大小 (不影響 NP0)
                // 更新 randNP 分佈
                if (NP > 1) {
                    uniform_int_distribution<> new_randNP(0, NP-1);
                    randNP = new_randNP;
                }
            }
        }

        G++; 
    }
    
    // 最終記錄
    double final_best_fitness = *min_element(fitness.begin(), fitness.end());
    if (logging) {
        logFile << NFE << "\t" << final_best_fitness << "\n";
        logFile.close();
        if (verbose) {
            cout << "Convergence log saved to " << logFilePath << endl;
        }
    }

    // 回傳最佳解
    int bestIdx = min_element(fitness.begin(), fitness.end()) - fitness.begin();
    return P[bestIdx];
}


void choose_pbest_and_r1r2(
    int i, int NP, int D, double p_i,
    const vector<double>& fitness,
    const vector<vector<double>>& P,
    const vector<vector<double>>& Archive,
    mt19937& gen,
    uniform_int_distribution<>& randNP,
    uniform_real_distribution<>& rand01,
    int& pBestIdx, int& r1, vector<double>& xr2
) {
    // 排序以找到前 p% 的個體
    vector<int> sortedIdx(NP);
    iota(sortedIdx.begin(), sortedIdx.end(), 0);
    sort(sortedIdx.begin(), sortedIdx.end(), [&](int a, int b){ return fitness[a] < fitness[b]; });
    
    // 選擇前 p% 的個體中的一個作為 pbest
    int num_p = max(2, static_cast<int>(ceil(NP * p_i)));
    if (num_p > NP) num_p = NP;
    pBestIdx = sortedIdx[static_cast<int>(rand01(gen) * num_p)];
    
    // 選擇 r1 (不能等於 i)
    do { 
        r1 = randNP(gen); 
    } while (r1 == i);
    
    // 建立 r2 的候選者列表 (當前種群 + Archive)
    vector<int> candidates;
    for (int c = 0; c < NP; ++c) {
        if (c != i && c != r1) candidates.push_back(c);
    }
    for (int a = 0; a < static_cast<int>(Archive.size()); ++a) {
        candidates.push_back(NP + a);
    }
    
    // 隨機選擇 xr2
    if (!candidates.empty()) {
        int xr2_idx = candidates[static_cast<int>(rand01(gen) * candidates.size())];
        xr2.resize(D);
        if (xr2_idx < NP) {
            xr2 = P[xr2_idx];
        } else {
            xr2 = Archive[xr2_idx - NP];
        }
    } else {
        // 如果沒有候選者，使用隨機個體
        xr2 = P[r1];
    }
}
// mutation
vector<double> mutation(const vector<vector<double>>& P, double Fi, int pBestIdx, int i, int r1, const vector<double>& xr2, double minVal, double maxVal, int D) {
    vector<double> vi(D);
    for (int j = 0; j < D; ++j) {
        double trial = P[i][j]
            + Fi * (P[pBestIdx][j] - P[i][j])
            + Fi * (P[r1][j] - xr2[j]);
        if (trial < minVal) trial = minVal;
        else if (trial > maxVal) trial = maxVal;
        vi[j] = trial;
    }
    return vi;
}

// crossover
vector<double> crossover(const vector<double>& xi, const vector<double>& vi, double CRi, int D, mt19937& gen, uniform_real_distribution<>& rand01) {
    vector<double> ui(D);
    int jrand = static_cast<int>(rand01(gen) * D);
    for (int j = 0; j < D; ++j) {
        if (rand01(gen) < CRi || j == jrand) ui[j] = vi[j];
        else ui[j] = xi[j];
    }
    return ui;
}