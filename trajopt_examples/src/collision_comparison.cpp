#include <tesseract_core/basic_types.h>
#include <tesseract_ros/kdl/kdl_env.h>
#include <tesseract_ros/ros_basic_plotting.h>
#include <tesseract_planning/trajopt/trajopt_planner.h>

#include <urdf_parser/urdf_parser.h>

#include <trajopt/file_write_callback.hpp>
#include <trajopt/plot_callback.hpp>
#include <trajopt_utils/logging.hpp>

int main(int argc, char** argv)
{
  //////////////////////
  /// INITIALIZATION ///
  //////////////////////

  ros::init(argc, argv, "pick_and_place_plan");
  ros::NodeHandle nh, pnh("~");

  int steps_per_phase;
  bool plotting_cb, file_write_cb;
  pnh.param<int>("steps_per_phase", steps_per_phase, 10);
  nh.param<bool>("/pick_and_place_node/plotting", plotting_cb, false);
  nh.param<bool>("/pick_and_place_node/file_write_cb", file_write_cb, false);

  // Set Log Level
  util::gLogLevel = util::LevelInfo;

  /////////////
  /// SETUP ///
  /////////////

  // Pull ROS params
  std::string urdf_xml_string, srdf_xml_string, box_parent_link;
  double box_side, box_x, box_y;
  nh.getParam("robot_description", urdf_xml_string);
  nh.getParam("robot_description_semantic", srdf_xml_string);
  nh.getParam("box_side", box_side);
  nh.getParam("box_x", box_x);
  nh.getParam("box_y", box_y);
  nh.getParam("box_parent_link", box_parent_link);

  // Initialize the environment
  urdf::ModelInterfaceSharedPtr urdf_model = urdf::parseURDF(urdf_xml_string);
  srdf::ModelSharedPtr srdf_model = srdf::ModelSharedPtr(new srdf::Model);
  srdf_model->initString(*urdf_model, srdf_xml_string);
  tesseract::tesseract_ros::KDLEnvPtr env(new tesseract::tesseract_ros::KDLEnv);
  assert(urdf_model != nullptr);
  assert(env != nullptr);
  bool success = env->init(urdf_model, srdf_model);
  assert(success);

  // Set the initial state of the robot
  std::unordered_map<std::string, double> joint_states;
  joint_states["iiwa_joint_1"] = 0.0;
  joint_states["iiwa_joint_2"] = 0.0;
  joint_states["iiwa_joint_3"] = 0.0;
  joint_states["iiwa_joint_4"] = -1.57;
  joint_states["iiwa_joint_5"] = 0.0;
  joint_states["iiwa_joint_6"] = 0.0;
  joint_states["iiwa_joint_7"] = 0.0;
  env->setState(joint_states);


  // Create the planner and the responses that will store the results
  tesseract::tesseract_planning::TrajOptPlanner planner;
  tesseract::tesseract_planning::PlannerResponse planning_response;

  // Choose the manipulator and end effector link
  std::string manip = "Manipulator";
  std::string end_effector = "iiwa_link_ee";

  // Create the problem construction info
  trajopt::ProblemConstructionInfo pci(env);

  pci.kin = env->getManipulator(manip);

  pci.basic_info.n_steps = steps_per_phase * 2;
  pci.basic_info.manip = manip;
  pci.basic_info.dt_lower_lim = 2;    // 1/most time
  pci.basic_info.dt_upper_lim = 100;  // 1/least time
  pci.basic_info.start_fixed = true;
  pci.basic_info.use_time = false;

  pci.init_info.type = trajopt::InitInfo::STATIONARY;
  pci.init_info.dt = 0.5;

  // Add a collision cost
  if (true)
  {
    std::shared_ptr<trajopt::CollisionTermInfo> collision(new trajopt::CollisionTermInfo);
    collision->name = "collision";
    collision->term_type = trajopt::TT_COST;
    collision->continuous = true;
    collision->first_step = 0;
    collision->last_step = pci.basic_info.n_steps - 1;
    collision->gap = 1;
    collision->info = trajopt::createSafetyMarginDataVector(pci.basic_info.n_steps, 0.025, 40);
    pci.cost_infos.push_back(collision);
  }

//      CollisionCost(prob.GetKin(),
//                                                          prob.GetEnv(),
//                                                          info[static_cast<size_t>(i - first_step)],
//                                                          prob.GetVarRow(i, 0, n_dof),
//                                                          prob.GetVarRow(i + gap, 0, n_dof))));

  // Create the pick problem
  trajopt::TrajOptProbPtr pick_prob = ConstructProblem(pci);

  // Set the optimization parameters (Most are being left as defaults)
  tesseract::tesseract_planning::TrajOptPlannerConfig config(pick_prob);
  config.params.max_iter = 50;



  // Solve problem. Results are stored in the response
  planner.solve(planning_response, config);



  ROS_INFO("Done");
  ros::spin();
}
