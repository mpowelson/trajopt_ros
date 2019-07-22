#include <trajopt_ifopt/joint_terms.h>

#include <console_bridge/console.h>

namespace trajopt
{
JointPosConstraint::JointPosConstraint(const Eigen::VectorXd& targets,
                                       const std::vector<JointPosition::ConstPtr> position_vars,
                                       const std::string& name)
  : ifopt::ConstraintSet(static_cast<int>(targets.size()) * static_cast<int>(position_vars.size()), name)
  , targets_(targets)
  , position_vars_(position_vars)
{
  // Check and make sure the targets size aligns with the vars passed in
  for (auto& position_var : position_vars)
  {
    if (targets.size() != position_var->GetRows())
      CONSOLE_BRIDGE_logError("Targets size does not align with variables provided");
  }
  n_dof_ = targets_.size();
  n_vars_ = static_cast<long>(position_vars.size());
  assert(n_dof_ > 0);
  assert(n_vars_ > 0);
}

Eigen::VectorXd JointPosConstraint::GetValues() const
{
  // Get the correct variables
  Eigen::VectorXd values(static_cast<size_t>(n_dof_) * position_vars_.size());
  // TODO: I'm not sure if this is any different/better than just position_var->GetValues()
  for (auto& position_var : position_vars_)
    values << this->GetVariables()->GetComponent(position_var->GetName())->GetValues();

  return values;
}

// Set the limits on the constraint values (in this case just the targets)
std::vector<ifopt::Bounds> JointPosConstraint::GetBounds() const
{
  std::vector<ifopt::Bounds> bounds(static_cast<size_t>(GetRows()));

  // All of the positions should be exactly at their targets
  for (long j = 0; j < n_vars_; j++)
  {
    for (long i = 0; i < n_dof_; i++)
    {
      bounds[static_cast<size_t>(i + j * n_dof_)] = ifopt::Bounds(targets_[i], targets_[i]);
    }
  }
  return bounds;
}

void JointPosConstraint::FillJacobianBlock(std::string var_set, Jacobian& jac_block) const
{
  // Loop over all of the variables this constraint uses
  for (int i = 0; i < static_cast<int>(position_vars_.size()); i++)
  {
    // Only modify the jacobian if this constraint uses var_set
    if (var_set == position_vars_[static_cast<size_t>(i)]->GetName())
    {
      // Reserve enough room in the sparse matrix
      jac_block.reserve(Eigen::VectorXi::Constant(targets_.size(), 1));
      for (int j = 0; j < targets_.size(); i++)
      {
        // This cnt has one row per value in the var_set, so we must add i*n_dof to put it in the correct relative
        // spot for this var_set. The column is referenced to the beginning of the var_set, so that is not needed
        jac_block.coeffRef(i * n_dof_ + j, j) = 1.0;
      }
    }
  }
}

}  // namespace trajopt
