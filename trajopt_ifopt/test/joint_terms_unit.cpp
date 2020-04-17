#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <ctime>
#include <gtest/gtest.h>
TRAJOPT_IGNORE_WARNINGS_POP
#include <trajopt_ifopt/joint_terms.h>
#include <console_bridge/console.h>

using namespace trajopt;
using namespace std;

/**
 * @brief Tests the Joint Position Constraint
 */
TEST(JointTermsUnit, joint_pos_constraint_1)
{
  CONSOLE_BRIDGE_logDebug("JointTermsUnit, JointPosConstraint_1");

  std::vector<JointPosition::Ptr> position_vars;
  Eigen::VectorXd init1(10);
  init1 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  position_vars.push_back(std::make_shared<JointPosition>(init1, "test_var1"));

  Eigen::VectorXd init2(10);
  init2 << 10, 11, 12, 13, 14, 15, 16, 17, 18, 19;
  position_vars.push_back(std::make_shared<JointPosition>(init2, "test_var2"));

  Eigen::VectorXd targets;
  targets << 20, 21, 22, 23, 24, 25, 26, 27, 28, 29;

  std::string name("test_cnt");
  JointPosConstraint position_cnt(targets, position_vars, name);

//  EXPECT_EQ(position_cnt.GetRows(), targets.size() * position_vars.size());
//  EXPECT_EQ(position_cnt.GetName(), name);
//  EXPECT_EQ(position_cnt.GetBounds().size(), init1.size());

  Eigen::VectorXd combined(20);
  combined << init1, init2;
//  EXPECT_TRUE(combined.isApprox(position_cnt.GetValues()));

  // TODO: Test bounds set correctly

  // TODO: Test jacobians sets correctly

  // TODO: Test jacobian is not changed when not one we care about

}

////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
