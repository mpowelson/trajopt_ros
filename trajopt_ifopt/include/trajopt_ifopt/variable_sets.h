#ifndef TRAJOPT_IFOPT_VARIABLE_SETS_H
#define TRAJOPT_IFOPT_VARIABLE_SETS_H

#include <ifopt/variable_set.h>
#include <ifopt/bounds.h>

#include <eigen3/Eigen/Dense>

namespace trajopt
{
/**
 * @brief Represents a single joint position in the optimization. Values are of dimension 1 x n_dof
 */
class JointPosition : public ifopt::VariableSet
{
public:
  using Ptr = std::shared_ptr<JointPosition>;
  using ConstPtr = std::shared_ptr<const JointPosition>;

  JointPosition(Eigen::VectorXd init_value, const std::string& name = "Joint_Position")
    : ifopt::VariableSet(static_cast<int>(init_value.size()), name)
  {
    // This needs to be set somehow
    ifopt::Bounds bounds(-M_PI, M_PI);
    bounds_ = std::vector<ifopt::Bounds>(static_cast<size_t>(init_value.size()), bounds);
    values_ = init_value;
  }

  // Here is where you can transform the Eigen::Vector into whatever
  // internal representation of your variables you have (here two doubles, but
  // can also be complex classes such as splines, etc..
  void SetVariables(const Eigen::VectorXd& x) override { values_ = x; }

  // Here is the reverse transformation from the internal representation to
  // to the Eigen::Vector
  Eigen::VectorXd GetValues() const override { return values_; }

  // Each variable has an upper and lower bound set here
  VecBound GetBounds() const override { return bounds_; }

  /**
   * @brief Sets the bounds for all of the variables in this
   * @param new_bounds
   */
  void SetBounds(VecBound& new_bounds) { bounds_ = new_bounds; }

private:
  VecBound bounds_;
  Eigen::VectorXd values_;
};

}  // namespace trajopt

#endif
