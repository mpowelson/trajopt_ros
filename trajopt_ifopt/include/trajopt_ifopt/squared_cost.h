#ifndef TRAJOPT_IFOPT_SQUARED_COST_H
#define TRAJOPT_IFOPT_SQUARED_COST_H

#include <ifopt/cost_term.h>

#include <Eigen/Eigen>

namespace trajopt
{
/**
 * @brief
 */
class SquaredCost : public ifopt::CostTerm
{
public:
  using Ptr = std::shared_ptr<SquaredCost>;
  using ConstPtr = std::shared_ptr<const SquaredCost>;

  SquaredCost(const ifopt::ConstraintSet::Ptr constraint);
  ~SquaredCost() override = default;

  double GetCost() const override;

  void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

private:
  ifopt::ConstraintSet::Ptr constraint_;

  Eigen::VectorXd weights_;
  Eigen::VectorXd targets_;
  long n_constraints_;
};

}  // namespace trajopt
#endif
