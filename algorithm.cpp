#include <ctime>
#include <cstdlib>
#include "algorithm.h"
#include "functions.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
#include <cmath> // for fabs

using namespace std;

void algorithm::RunALG(int D, int NP, int G, double pb, int c, int maxVal, int func_num) {
    
    // Initialize parameters
    int minVal = -maxVal;
    int H = 6; // Standard for iL-SHADE
    vector<double> MF(H, 0.5), MCR(H, 0.8); // 設定縮放率與交叉率
    vector<Particle> Archive; // 歷史存檔
    int k = 1; // Index counter 從1開始
    const double terminal_value = -1.0; //終止值
    mt19937 gen(static_cast<unsigned>(time(nullptr)));
    uniform_real_distribution<> rand01(0.0, 1.0);

    int NP_init = NP; // Assume user sets NP to 12*D or similar
    int NP_min = 4;
    int current_NP = NP_init;
    int MAX_NFE = 10000 * D; // Standard for CEC benchmarks
    int NFE = 0; // Number_Of_Evaluations
    int Ng = current_NP; // Archive size = current NP
    int g = 1; // Generation=1

    // Initialize population
    population.resize(current_NP);
    for (int i = 0; i < current_NP; ++i) {
        Init(population[i], D, minVal, maxVal, gen);
        population[i].fitness = Evaluation(population[i], func_num);
        NFE++;
        cout << "Particle " << i << " fitness: " << population[i].fitness << endl;
        cout << "Position: ";
        for (int d = 0; d < D; d++) {
            cout << population[i].position[d] << " ";
        }
        cout << endl;
    }

    // Main loop
    vector<double> CR(current_NP), F(current_NP);
    while (NFE < MAX_NFE ) {
        int idx;
        //cout << "Generation: " << g  << endl;
        cout << "Number of Function Evaluations: " << NFE << endl;
        cout <<" Best fitness: " << get_best_fitness(idx) << endl;
        vector<Particle> v(current_NP);
        uniform_int_distribution<> randH(0, H - 1); // 0 to H-1
        vector<double> S_CR, S_F, S_delta;

        double fes_frac = static_cast<double>(NFE) / MAX_NFE;
        double p = 0.2 - 0.1 * fes_frac; // Decreases from 0.2 to 0.1

        uniform_int_distribution<> randNP(0, current_NP - 1);

        for (int i = 0; i < current_NP; ++i) {
            v[i].position.resize(D);
            v[i].fitness = numeric_limits<double>::max();
            uniform_int_distribution<> randD(1, D);
            int r = randH(gen);
            if (r == H - 1) { // 0-based, so H-1 for last entry
                MF[r] = 0.9;
                MCR[r] = 0.9;
            }
            double cr = 0.0;
            if (MCR[r] < 0) {
                cr = 0.0;
            } else {
                normal_distribution<> randCR(MCR[r], 0.1);
                cr = randCR(gen);
                cr = max(0.0, min(1.0, cr)); // Clip to [0,1]
            }
            CR[i] = cr;

            cauchy_distribution<> randF(MF[r], 0.1);
            double f = randF(gen);
            while (f <= 0.0) {
                f = randF(gen);
            }
            if (f > 1.0) f = 1.0;
            F[i] = f;

            // Phase-dependent adjustments
            if (fes_frac < 0.25) {
                CR[i] = max(CR[i], 0.5);
                F[i] = min(F[i], 0.7);
            } else if (fes_frac < 0.5) {
                CR[i] = max(CR[i], 0.25);
                F[i] = min(F[i], 0.8);
            } else if (fes_frac < 0.75) {
                F[i] = min(F[i], 0.9);
            }

            // Select p-best
            int pBestIdx = Select_p_best(population, p, current_NP, gen);

            int jrand = randD(gen);

            for (size_t j = 0; j < D; ++j) {
                // Mutation: current-to-pBest/1
                int r1 = randNP(gen);
                while (r1 == i || r1 == pBestIdx) r1 = randNP(gen);

                int total_for_r2 = current_NP + Archive.size();
                uniform_int_distribution<> randTotal(0, total_for_r2 - 1);
                int r2_idx = randTotal(gen);
                double r2_j;
                if (r2_idx < current_NP) {
                    r2_j = population[r2_idx].position[j];
                } else {
                    r2_j = Archive[r2_idx - current_NP].position[j];
                }
                double r1_j = population[r1].position[j];
                double diff_r1_r2 = r1_j - r2_j;

                // Ensure r2 != i, pBestIdx, r1 (approx, since r2_idx may coincide)
                while (r2_idx == i || r2_idx == pBestIdx || r2_idx == r1) {
                    r2_idx = randTotal(gen);
                    if (r2_idx < current_NP) {
                        r2_j = population[r2_idx].position[j];
                    } else {
                        r2_j = Archive[r2_idx - current_NP].position[j];
                    }
                    diff_r1_r2 = r1_j - r2_j;
                }

                double mutant = population[i].position[j] 
                              + F[i] * (population[pBestIdx].position[j] - population[i].position[j]) 
                              + F[i] * diff_r1_r2;

                // Crossover
                if (rand01(gen) <= CR[i] || (j + 1) == jrand) {
                    v[i].position[j] = mutant;
                } else {
                    v[i].position[j] = population[i].position[j];
                }

                // Boundary handling: (min + old)/2, if still out, random
                if (v[i].position[j] < minVal || v[i].position[j] > maxVal) {
                    v[i].position[j] = (minVal + population[i].position[j]) / 2.0;
                    if (v[i].position[j] < minVal || v[i].position[j] > maxVal) {
                        uniform_real_distribution<> dist(minVal, maxVal);
                        v[i].position[j] = dist(gen);
                    }
                }
            }
            v[i].fitness = Evaluation(v[i], func_num);
            NFE++;
        }

        for (int i = 0; i < current_NP; ++i) {
            Particle old_part = population[i];
            double old_fit = old_part.fitness;
            if (v[i].fitness <= old_fit) {
                population[i] = v[i];
            }
            if (v[i].fitness < old_fit) {
                Archive.push_back(old_part);
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
                S_delta.push_back(fabs(old_fit - v[i].fitness));
            }
            // Adjust Archive size
            while (Archive.size() > Ng) {
                uniform_int_distribution<> randA(0, Archive.size() - 1);
                int a = randA(gen);
                Archive.erase(Archive.begin() + a);
            }
        }

        // Update MF and MCR
        if (!S_CR.empty() && !S_F.empty()) {
            double sum_delta = 0.0;
            for (auto d : S_delta) sum_delta += d;

            double sum_w_cr = 0.0;
            double sum_w_f = 0.0;
            double sum_w_f2 = 0.0;
            for (size_t s = 0; s < S_CR.size(); ++s) {
                double w = S_delta[s] / sum_delta;
                sum_w_cr += w * S_CR[s];
                sum_w_f += w * S_F[s];
                sum_w_f2 += w * (S_F[s] * S_F[s]);
            }
            double mean_cr = sum_w_cr; // Weighted arithmetic mean
            double mean_f = (sum_w_f > 0.0) ? sum_w_f2 / sum_w_f : MF[k]; // Weighted Lehmer mean

            auto max_scr_it = max_element(S_CR.begin(), S_CR.end());
            double max_scr = (max_scr_it != S_CR.end()) ? *max_scr_it : 0.0;

            if (MCR[k - 1] == terminal_value || max_scr == 0.0) { // 1-based to 0-based
                MCR[k - 1] = terminal_value;
            } else {
                MCR[k - 1] = mean_cr;
            }
            MF[k - 1] = mean_f;
            k++;
            if (k > H) k = 1;
        }

        // Linear Population Size Reduction (LPSR)
        double next_NP_d = round(NP_init - (NP_init - NP_min) * (static_cast<double>(NFE) / MAX_NFE));
        int next_NP = static_cast<int>(next_NP_d);
        if (next_NP < NP_min) next_NP = NP_min;
        if (next_NP < current_NP) {
            // Sort population by fitness (ascending, best first)
            sort(population.begin(), population.end(), [](const Particle& a, const Particle& b) {
                return a.fitness < b.fitness;
            });
            // Remove worst (from end)
            population.resize(next_NP);
            current_NP = next_NP;
            Ng = current_NP;
        }

        g++;
    }
}

double algorithm::Evaluation(Particle &particle, int func_num) {
    //double fitness = cec14_wrapper(particle.position, func_num);
    double fitness= ackley(particle.position);
    //cout << "Evaluation " << fitness << endl;
    return fitness;
}

double algorithm::get_best_fitness(int& best_idx) const {
    double best_fitness = numeric_limits<double>::max();
    best_idx = -1;
    for (size_t i = 0; i < population.size(); ++i) {
        if (population[i].fitness < best_fitness) {
            best_fitness = population[i].fitness;
            best_idx = static_cast<int>(i);
        }
    }
    return best_fitness;
}
vector<double> algorithm::get_best_position() const {
    int best_idx;
    get_best_fitness(best_idx);
    if (best_idx >= 0 && best_idx < static_cast<int>(population.size())) {
        return population[best_idx].position;
    }
    return vector<double>(); // Return empty if no valid best index
}
void algorithm::Init(Particle &particle, int D, int minVal, int maxVal, mt19937& gen) {
    particle.position.resize(D);
    uniform_real_distribution<> dist(static_cast<double>(minVal), static_cast<double>(maxVal));
    for (int i = 0; i < D; ++i) {
        particle.position[i] = dist(gen); // 隨機浮點數初始化
    }
    particle.fitness = numeric_limits<double>::max();
}

int algorithm::Select_p_best(vector<Particle> &population, double pb, int NP, mt19937& gen) {
    // 收集所有 fitness
    vector<int> idx(NP);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return population[a].fitness < population[b].fitness;
    });

    // 計算前 pb% 的個體數量，至少一位
    int num_p = max(1, static_cast<int>(NP * pb));
    // 隨機選一位
    uniform_int_distribution<> dist(0, num_p - 1);
    int bestIdx = idx[dist(gen)];
    return bestIdx;
}

