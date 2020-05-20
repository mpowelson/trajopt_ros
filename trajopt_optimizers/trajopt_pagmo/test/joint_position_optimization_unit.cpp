#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <iostream>

#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>
#include <console_bridge/console.h>

#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/algorithms/ihs.hpp>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_pagmo/pagmo_solver.h>

#include <trajopt_ifopt/constraints/joint_position_constraint.h>
#include <trajopt_ifopt/variable_sets/joint_position_variable.h>

const bool DEBUG = false;

class JointPositionOptimization : public testing::TestWithParam<const char*>
{
public:
  // 1) Create the problem
  ifopt::Problem nlp_;

  void SetUp() override
  {
    if (DEBUG)
      console_bridge::setLogLevel(console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_DEBUG);
    else
      console_bridge::setLogLevel(console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_NONE);

    // 1) Create the problem
    ifopt::Problem nlp;

    // 2) Add Variables
    std::vector<trajopt::JointPosition::Ptr> vars;
    for (int ind = 0; ind < 2; ind++)
    {
      auto pos = Eigen::VectorXd::Zero(7);
      std::vector<std::string> joint_names(7, "name");
      auto var = std::make_shared<trajopt::JointPosition>(pos, joint_names, "Joint_Position_" + std::to_string(ind));
      vars.push_back(var);
      nlp_.AddVariableSet(var);
    }

    // 3) Add constraints
    Eigen::VectorXd start_pos = Eigen::VectorXd::Zero(7);
    std::vector<trajopt::JointPosition::Ptr> start;
    start.push_back(vars.front());

    auto start_constraint = std::make_shared<trajopt::JointPosConstraint>(start_pos, start, "StartPosition");
    nlp_.AddConstraintSet(start_constraint);

    Eigen::VectorXd end_pos = Eigen::VectorXd::Ones(7);
    std::vector<trajopt::JointPosition::Ptr> end;
    end.push_back(vars.back());
    auto end_constraint = std::make_shared<trajopt::JointPosConstraint>(end_pos, end, "EndPosition");
    nlp_.AddConstraintSet(end_constraint);

    if (DEBUG)
    {
      nlp_.PrintCurrent();
      std::cout << "Jacobian: \n" << nlp_.GetJacobianOfConstraints() << std::endl;
    }
  }
};

/**
 * @brief Applies a joint position constraint and solves the problem with Pagmo compass search
 */
TEST_F(JointPositionOptimization, joint_position_optimization_trajopt_pagmo_compass)
{
  ifopt::Problem nlp_trajopt_sqp(nlp_);

  pagmo::compass_search uda(5000, 0.1, 1e-4, 0.5);
  uda.set_verbosity(DEBUG);
  pagmo::algorithm algo{ uda };

  trajopt_pagmo::PagmoSolver solver{ algo };
  solver.Solve(nlp_trajopt_sqp);
  Eigen::VectorXd x = nlp_trajopt_sqp.GetOptVariables()->GetValues();

  for (Eigen::Index i = 0; i < 7; i++)
    EXPECT_NEAR(x[i], 0.0, 1e-3);
  for (Eigen::Index i = 7; i < 14; i++)
    EXPECT_NEAR(x[i], 1.0, 1e-3);

  if (DEBUG)
  {
    std::cout << "X: " << x.transpose() << std::endl;
    nlp_trajopt_sqp.PrintCurrent();
  }
}

/**
 * @brief Applies a joint position constraint and solves the problem with Pagmo Extended Ant Colony Optimization (GACO)
 */
TEST_F(JointPositionOptimization, joint_position_optimization_trajopt_pagmo_gaco)
{
  ifopt::Problem nlp_trajopt_sqp(nlp_);

  pagmo::gaco uda(100);
  uda.set_verbosity(DEBUG);
  pagmo::algorithm algo{ uda };

  trajopt_pagmo::PagmoSolver solver{ algo };
  // Apparently Extended Ant Colony Optimization (GACO) works best with a lot of ants
  solver.config_.population_size_ = 100;
  solver.Solve(nlp_trajopt_sqp);
  Eigen::VectorXd x = nlp_trajopt_sqp.GetOptVariables()->GetValues();

  // TODO: Tweak settings to change convergence criteria
  for (Eigen::Index i = 0; i < 7; i++)
    EXPECT_NEAR(x[i], 0.0, 0.05);
  for (Eigen::Index i = 7; i < 14; i++)
    EXPECT_NEAR(x[i], 1.0, 0.05);

  if (DEBUG)
  {
    std::cout << "X: " << x.transpose() << std::endl;
    nlp_trajopt_sqp.PrintCurrent();
  }
}

////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
