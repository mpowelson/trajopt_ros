#include <ros/ros.h>

// Pagmo
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>

// From UDP
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>  // needed for cereal registration macro
#include <pagmo/types.hpp>

namespace pagmo
{
struct squared
{
  squared() {};

  vector_double fitness(const vector_double& x) const
  {
    vector_double f(1, 0.);

    f[0] = x[0] * x[0] + 5;
    return f;
  }
  std::pair<vector_double, vector_double> get_bounds() const
  {
    vector_double lb(1, -500);
    vector_double ub(1, 500);
    return { lb, ub };
  }


};
}  // namespace pagmo

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pagmo");
  ros::NodeHandle nh;

  // 1 - Instantiate a pagmo problem constructing it from a UDP
  // (user defined problem).
  pagmo::squared tmp;
  pagmo::problem pagmo_prob(tmp);
  //  pagmo::problem pagmo_prob{pagmo::schwefel2(30)};

  // 2 - Instantiate a pagmo algorithm
  pagmo::algorithm algo{ pagmo::sade(100) };

  // 3 - Instantiate an archipelago with 16 islands having each 20 individuals
  pagmo::archipelago archi{ 16, algo, pagmo_prob, 20 };

  // 4 - Run the evolution in parallel on the 16 separate islands 10 times.
  archi.evolve(10);

  // 5 - Wait for the evolutions to be finished
  archi.wait_check();

  // 6 - Print the fitness of the best solution in each island
  for (const auto& isl : archi)
  {
    std::cout << isl.get_population().champion_f()[0] << '\n';
  }
}
