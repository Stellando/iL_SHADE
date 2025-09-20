#include <vector>
#include <functional>
#include <random>
#include <ctime>
using namespace std;

#ifndef ALGORITHM_H
#define ALGORITHM_H

typedef struct Particle
{
    vector<double> position;
    double fitness;
    //將 fitness 初始化為最大值
    Particle():fitness(numeric_limits<double>::max()){} 
} Particle;

class algorithm 
{
    public:
        void RunALG(int, int, int, double, int,int, int);
        double get_best_fitness() const;
        vector<double> get_best_position() const;
    private:
        vector<Particle> population;
        void Init(Particle &particle,int,int,int , mt19937& gen);
        int Select_p_best(vector<Particle> &population  , double , int );
        void Mutation(Particle &particle, const Particle &p_best, const vector<Particle> &population, int, int);
        void Crossover(Particle &particle, const Particle &mutant, int);
        double Evaluation(Particle &particle , int);
        //double evaluate_fitness(const Particle &particle);
};


#endif // ALGORITHM_H