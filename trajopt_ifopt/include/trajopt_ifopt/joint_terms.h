#ifndef TRAJOPT_IFOPT_JOINT_TERMS_H
#define TRAJOPT_IFOPT_JOINT_TERMS_H

#include <ifopt/cost_term.h>
#include <ifopt/constraint_set.h>
#include <ifopt/composite.h>
#include <trajopt_ifopt/variable_sets.h>

#include <Eigen/Eigen>

namespace trajopt
{
/**
 * @brief This creates a joint position constraint. It applies a constraint that each of the passed variables must be
 * equal to the target vector that is passed in.
 *
 * TODO:
 * Convenient constructors for single positions
 * Capability for inequality constraints
 *
 */
class JointPosConstraint : public ifopt::ConstraintSet
{
public:
  using Ptr = std::shared_ptr<JointPosConstraint>;
  using ConstPtr = std::shared_ptr<const JointPosConstraint>;

  JointPosConstraint(const Eigen::VectorXd& targets,
                     const std::vector<JointPosition::ConstPtr> position_vars,
                     const std::string& name = "JointPos");

  ~JointPosConstraint() override = default;

  /**
   * @brief Returns the values associated with the constraint. In this case that is the joint values associated with
   * each of the joint positions
   *
   * Note: This is very sparse.
   * @return
   */
  Eigen::VectorXd GetValues() const override;

  // Set the limits on the constraint values (in this case just the targets)
  std::vector<ifopt::Bounds> GetBounds() const override;

  /**
   * @brief FillJacobianBlock
   *
   * @param var_set
   * @param jac_block
   */
  void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

private:
  long n_dof_;
  long n_vars_;
  /** @brief Vector of velocity targets */
  Eigen::VectorXd targets_;

  std::vector<JointPosition::ConstPtr> position_vars_;
};
};  // namespace trajopt

#endif
