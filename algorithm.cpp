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

void algorithm::RunALG(int D, int NP, int G, double pb, int c, int maxVal, int fun_num) {
    
    // Initialize parameters
    int minVal = -maxVal;
    int H = 6; // Standard for iL-SHADE
    vector<double> MF(H, 0.5), MCR(H, 0.8); // 設定縮放率與交叉率
    vector<Particle> Archive; // 歷史存檔
    int k = 1; // Index counter 從1開始
    const double terminal_value = -1.0; //終止值
    random_device rd;  // 硬體熵來源
    mt19937 gen(rd());  
    uniform_real_distribution<> rand01(0.0, 1.0);

    int NP_init = NP; // Assume user sets NP to 12*D or similar
    int NP_min = 4;
    int current_NP = NP_init;
    int MAX_NFE = 10000 * D; // Standard for CEC benchmarks
    int NFE = 0; // Number_Of_Evaluations
    int A_size = current_NP; // Archive size = current NP
    int g = 1; // Generation=1

    // Initialize population
    population.resize(current_NP);
    for (int i = 0; i < current_NP; ++i) {
        Init(population[i], D, minVal, maxVal, gen);
        population[i].fitness = Evaluation(population[i], fun_num);
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
        uniform_int_distribution<> randNP(0, current_NP - 1);
        vector<double> S_CR, S_F, F_delta(D,0.0);

        double fes_frac = static_cast<double>(NFE) / MAX_NFE;
        double p = (0.2 - 0.1) * fes_frac + 0.1; // Decreases from 0.2 to 0.1
        //cout << "p: " << p << endl;

        //將population與Archive合併為temp_pop
        vector<Particle> temp_pop = population; // Copy current population
        temp_pop.insert(temp_pop.end(), Archive.begin(), Archive.end()); // Add archive
        uniform_int_distribution<> randTotal(0, temp_pop.size() - 1);



        for (int i = 0; i < current_NP; ++i) {
            v[i].position.resize(D);
            v[i].fitness = numeric_limits<double>::max();
            uniform_int_distribution<> randD(1, D);
            //r ← select from [1, H] randomly
            int r = randH(gen);
            normal_distribution<> randCR(MCR[r], 0.1); 
            cauchy_distribution<> randF(MF[r], 0.1);
            if(r==H-1){
                MF[r]=0.9;
                MCR[r]=0.9;
            }
            if(MCR[r]<0){
                CR[i]=0;
            }else{
                CR[i]=randCR(gen); //normal distribution
            }
            if(g<0.25*G){
                CR[i]=max(CR[i],0.5);
            }else if(g<0.5*G){
                CR[i]=max(CR[i],0.25);
            }
            F[i]=randF(gen); //cauchy distribution
            if(g<0.25*G){
                F[i]=min(F[i],0.7);
            }else if(g<0.5*G){
                F[i]=min(F[i],0.8);
            }else if(g<0.75*G){
                F[i]=min(F[i],0.9);
            }

            // Select p-best
            int pBestIdx = Select_p_best(population, p, current_NP, gen);
            int jrand = randD(gen);

            for (size_t j = 0; j < D; ++j) {
                // Mutation: current-to-pBest/1
                int r1 = randTotal(gen);
                while (r1 == i || r1 == pBestIdx) r1 = randTotal(gen);
                int r2 = randTotal(gen);
                while (r2 == i || r2 == pBestIdx || r2 == r1) r2 = randTotal(gen);

                double mutant = population[i].position[j] 
                              + F[i] * (population[pBestIdx].position[j] - population[i].position[j]) 
                              + F[i] * (temp_pop[r1].position[j] - temp_pop[r2].position[j]);

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
                F_delta[j] = fabs(population[i].fitness - v[i].fitness);
            }
            v[i].fitness = Evaluation(v[i], fun_num);
            NFE++;
        }

        // Selection 
        for (int i = 0; i < current_NP; ++i) {
            if (v[i].fitness <= population[i].fitness) {
                population[i] = v[i];
            }else{
                // Do nothing, keep the old one
            }
            if (v[i].fitness < population[i].fitness) {
                Archive.push_back(population[i]);
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
                F_delta.push_back(fabs(population[i].fitness - v[i].fitness));
            }
            // Adjust Archive size
            while (Archive.size() > A_size) {
                uniform_int_distribution<> randA(0, Archive.size() - 1);
                int a = randA(gen);
                Archive.erase(Archive.begin() + a);
            }
        }

        //update MF and MCR
        bool hasPositiveSCR = any_of(S_CR.begin(), S_CR.end(), [](double v){ return v > 0; });
        if(!S_CR.empty() && !S_F.empty()){
            if(MCR[k]==terminal_value || !hasPositiveSCR){
                MCR[k]=terminal_value;
            }else{
                MCR[k]=(meanWL(S_CR, F_delta) + MCR[k])/2;
            }
            MF[k]= (meanWL(S_F, F_delta) + MF[k])/2;
            k++;
            if(k>H) k=1;
        }else{
            //MCR[k]不變
            //MF[k]不變
        }

        // Linear Population Size Reduction (LPSR)
        int next_NP = round((NP_min - NP_init) * (static_cast<double>(NFE) / MAX_NFE) + NP_init);
        if (next_NP < NP_min) next_NP = NP_min;
        if (next_NP < current_NP) {
            // Sort population by fitness (ascending, best first)
            sort(population.begin(), population.end(), [](const Particle& a, const Particle& b) {
                return a.fitness < b.fitness;
            });
            // Remove worst (from end)
            population.resize(next_NP);
            current_NP = next_NP;
            A_size = current_NP;
        }

        g++;
    }
}

double algorithm::Evaluation(Particle &particle, int fun_num) {
    //double fitness = cec14_wrapper(particle.position, func_num);
    /** */
    if(fun_num==1)
        return ackley(particle.position);
    else if(fun_num==2)
        return sphere_func(particle.position);
    else if(fun_num==3)
        return rastrigin(particle.position);
    else if(fun_num==4)
        return rosenbrock(particle.position);
    else if(fun_num==5)
        return griewank(particle.position);
    else
        return ackley(particle.position);
    
    //cout << "Evaluation " << fitness << endl;
    //return fitness;
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

