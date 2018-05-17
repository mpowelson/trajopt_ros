#ifndef TESSERACT_ROS_PLANNING_CHAIN_OMPL_INTERFACE_H
#define TESSERACT_ROS_PLANNING_CHAIN_OMPL_INTERFACE_H

#include <ompl/geometric/SimpleSetup.h>
#include <tesseract_ros/kdl/kdl_env.h>

namespace tesseract_ros_planning
{

struct OmplPlanParameters
{
  double planning_time = 5.0;
  bool simplify = true;
};

class ChainOmplInterface
{
public:
  ChainOmplInterface(tesseract::tesseract_ros::ROSBasicEnvConstPtr environment, const std::string& manipulator_name);

  boost::optional<ompl::geometric::PathGeometric> plan(ompl::base::PlannerPtr planner, const std::vector<double>& from,
                                                       const std::vector<double>& to, const OmplPlanParameters& params);

  ompl::base::SpaceInformationPtr spaceInformation();


private:
  bool isStateValid(const ompl::base::State* state) const;

  bool isContactAllowed(const std::string& a, const std::string& b) const;

private:
  ompl::geometric::SimpleSetupPtr ss_;
  tesseract::tesseract_ros::ROSBasicEnvConstPtr env_;
  tesseract::IsContactAllowedFn contact_fn_;
  std::vector<std::string> joint_names_;
  std::vector<std::string> link_names_;
};

}

#endif
