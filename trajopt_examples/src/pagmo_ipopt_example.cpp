// Pagmo
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/utils/gradients_and_hessians.hpp>

#define HAVE_CSTDDEF
#include <IpTNLP.hpp>
#undef HAVE_CSTDDEF

#include <pagmo/algorithms/ipopt.hpp>

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
  // This constructor is needed to when using the fork island because it appears the serialization library does not
  // support constructors with no arguments.
  squared(int unneeded){};

  squared(){};

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

  // Note that IPOPT requires gradient information. Because of this, we will approximate it numerically using PAGMO's
  // utilities
  bool has_gradient() const { return true; }
  vector_double gradient(const vector_double& x) const
  {
    using FTYPE = std::function<vector_double(const vector_double&)>;
    return pagmo::estimate_gradient<FTYPE>(std::bind(&squared::fitness, this, std::placeholders::_1), x);
  }

  template <typename Archive>
  void serialize(Archive& ar)
  {
    ar(123);
  }
};
}  // namespace pagmo

// This is also needed for the fork island
PAGMO_REGISTER_PROBLEM(pagmo::squared)

int main(int argc, char** argv)
{
  std::cout << "Beginning Optimization \n";

  // 1 - Instantiate a pagmo problem constructing it from a UDP
  pagmo::squared tmp;
  pagmo::problem pagmo_prob(tmp);

  // 2 - Create a User Defined Algorithm (UDA)
  pagmo::ipopt ipopt;

  // Set some of the IPOPT Wrapper settings.
  // Notice that IPOPT only optimizes one of the population per evolve, so we set how we choose which one optimize
  std::string type = "worst";
  ipopt.set_selection(type);
  //  ipopt.set_selection(0); // Can also select based on index as so
  ipopt.set_replacement(type);

  // Print the optimization results every 1 evaluation
  ipopt.set_verbosity(1);

  // 3 - Define a population to optimize
  pagmo::population pop(pagmo_prob, 1);

  // 4 - Evolve the population
  for (int i = 0; i < 1; i++)
  {
    pop = ipopt.evolve(pop);
    std::cout << "Champion x: " << pop.champion_x()[0] << "\n";
    std::cout << "Champion f: " << pop.champion_f()[0] << "\n";
  }

  // Now we solve using an archipelago.
  // For me this works about half of the time. This is not really all that beneficial in this case anyway because ipopt
  // is not threadsafe. Therefore it creates a fork island instead of a thread island.
  std::cout << "Now we solve the problem several times in parallel \n";
  std::cout << "If this does not return almost immediately, it is probably hung for some reason \n";
  ipopt.set_verbosity(0);

  // 2 - Instantiate a pagmo algorithm
  pagmo::algorithm algo{ ipopt };
  pagmo::problem test{ tmp };
  pagmo::population pop2(test, 1);

  // 3 - Instantiate an archipelago with 16 islands having each 20 individuals
  pagmo::archipelago archi{ 16, algo, pop2 };

  // 4 - Run the evolution in parallel on the 16 separate islands 10 times.
  archi.evolve(1);

  // 5 - Wait for the evolutions to be finished
  archi.wait_check();

  // 6 - Print the fitness of the best solution in each island
  for (const auto& isl : archi)
  {
    std::cout << "Champion x: " << isl.get_population().champion_x()[0]
              << "   Champion f: " << isl.get_population().champion_f()[0] << '\n';
  }
}
