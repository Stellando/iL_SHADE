#include <ctime>
#include <cstdlib>
#include "algorithm.h"
#include "functions.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;
void algorithm::RunALG(int D, int NP, int G, double pb, int c, int maxVal, int func_num) {
    
    // Initialize parameters
    int minVal = -maxVal;
    int H = 10; // 固定 H 
    vector<double>  MF(H, 0.5) , MCR(H, 0.8); // 設定縮放率與交叉率
    vector<Particle> Archive; // 歷史存檔
    int k = 1; // Index counter 從1開始
    const double terminal_value = -1; //終止值⊥
    mt19937 gen(static_cast<unsigned>(time(nullptr)));
    uniform_real_distribution<> rand01(0.0, 1.0);
    uniform_int_distribution<> randNP(0, NP-1);
    //uniform_int_distribution<> randD(1, D);

    // Initialize population
    population.resize(NP);
    for (int i = 0; i < NP; ++i) {
        Init(population[i], D, minVal, maxVal , gen);
        population[i].fitness = Evaluation(population[i] , func_num);
        cout << "Particle " << i << " fitness: " << population[i].fitness << endl;
        cout<< "Position: ";
        for(int d=0;d<D;d++){
            cout<<population[i].position[d]<<" ";
        }
        cout<<endl;
    }

    // Main loop
    int MAX_NFE = D*1000;
    int NFE = 0 ; // Number_Of_Evaluations
    int Ng = NP; // initial Ng = Ninit = NP
    int g = 1; // Generation=1
    vector<double> CR(NP), F(NP);
    while ( NFE < MAX_NFE) {
        cout << "Generation: " << g << " Best fitness: " << get_best_fitness() << endl;
        //cout << "Evaluations: " << NFE << "/" << MAX_NFE << endl;
        vector<Particle> v(NP);
        uniform_int_distribution<> randH(0, H-1); // 0 to H-1
        vector<double> S_CR, S_F, F_delta(D, 0.0);

        //i=population size
        for (int i = 0; i < NP; ++i) {
            v[i].position.resize(D);
            v[i].fitness = numeric_limits<double>::max();
            uniform_int_distribution<> randD(1, D);
            //r ← select from [1, H] randomly
            int r = randH(gen);
            normal_distribution<> randCR(MCR[r], 0.1); 
            cauchy_distribution<> randF(MF[r], 0.1);
            if(r==H){
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
            int pBestIdx = Select_p_best(population, pb, NP);
            cout << "pBestIdx: " << pBestIdx << endl;

            //current-to-pBest/1/bin
            v.resize(NP);
            int jrand = randD(gen);
            for (size_t j = 0; j < v[i].position.size(); ++j) {
                // Mutation
                int r1 = randNP(gen);
                while (r1 == i || r1 == pBestIdx) r1 = randNP(gen);
                int r2 = randNP(gen);
                while (r2 == i || r2 == pBestIdx || r2 == r1) r2 = randNP(gen);

                v[i].position[j] = population[i].position[j] 
                + F[i]*(population[pBestIdx].position[j]-population[i].position[j]) 
                + F[i]*(population[r1].position[j]-population[r2].position[j]);
                // Crossover
                if(rand01(gen) <= CR[i] || j+1 == jrand){
                    //v[i].position[j] = v[i].position[j];
                }else{
                    v[i].position[j] = population[i].position[j];
                }
                //v[i].fitness = Evaluation(v[i] , func_num);
                F_delta[j]=(fabs(population[i].fitness - v[i].fitness));
            }
            v[i].fitness = Evaluation(v[i] , func_num);
            //F_delta[i]=(fabs(population[i].fitness - v[i].fitness));

        }
        for(int i=0;i<NP;i++){
            // Selection
            if(v[i].fitness <= population[i].fitness){
                population[i] = v[i];
                
            }else{
                //population[i]不變
            }
            if(v[i].fitness < population[i].fitness){
                //加入Archive
                Archive.push_back(population[i]);
                S_CR.push_back(CR[i]);
                S_F.push_back(F[i]);
            }
            //調整Archive的大小
            while (Archive.size() > Ng) {
                uniform_int_distribution<> randA(0, Archive.size()-1);
                int a = randA(gen);
                Archive.erase(Archive.begin()+a);
            }

            //update MF and MCR
            bool hasPositiveSCR = std::any_of(S_CR.begin(), S_CR.end(), [](double v){ return v > 0; });
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
        }
        g++;
        NFE += NP;
    }
}
double algorithm::get_best_fitness() const {
    // Implementation to return the best fitness found
    double best_fitness = numeric_limits<double>::max();
    for (const auto &particle : population) {
        if (particle.fitness < best_fitness) {
            best_fitness = particle.fitness;
        }
    }
    return best_fitness;
}
void algorithm::Init(Particle &particle, int D, int minVal, int maxVal, mt19937& gen) {
    particle.position.resize(D);
    uniform_real_distribution<> dist(minVal, maxVal);
    for (int i = 0; i < D; ++i) {
        particle.position[i] = dist(gen); // 隨機浮點數初始化
    }
    particle.fitness = numeric_limits<double>::max();
}

int algorithm::Select_p_best(vector<Particle> &population, double pb, int NP) {
    // 收集所有 fitness
    vector<int> idx(NP);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return population[a].fitness < population[b].fitness;
    });

    // 計算前 pb% 的個體數量，至少一位
    int num_p = max(1, static_cast<int>(NP * pb));
    // 隨機選一位
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, num_p - 1);
    int bestIdx = idx[dist(gen)];
    return bestIdx;
}

void algorithm::Mutation(Particle &particle, const Particle &p_best, const vector<Particle> &population, int g, int c) {
    // Implementation of mutation operation
}

void algorithm::Crossover(Particle &particle, const Particle &mutant, int g) {
    // Implementation of crossover operation
}

double algorithm::Evaluation(Particle &particle , int func_num) {
    double fitness = cec14_wrapper(particle.position, func_num);
    //cout << "Evaluation " << fitness << endl;
    return fitness;
}