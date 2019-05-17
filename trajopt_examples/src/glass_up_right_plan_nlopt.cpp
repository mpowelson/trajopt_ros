/**
 * @file glass_up_right_plan.cpp
 * @brief Example using Trajopt for constrained free space planning
 *
 * @author Levi Armstrong
 * @date Dec 18, 2017
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2017, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <jsoncpp/json/json.h>
#include <ros/ros.h>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <tesseract_ros/kdl/kdl_chain_kin.h>
#include <tesseract_ros/kdl/kdl_env.h>
#include <tesseract_ros/ros_basic_plotting.h>
#include <trajopt/plot_callback.hpp>
#include <trajopt/file_write_callback.hpp>
#include <trajopt/problem_description.hpp>
#include <trajopt_utils/config.hpp>
#include <trajopt_utils/logging.hpp>

// Pagmo
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>

#define HAVE_CSTDDEF
#include <IpTNLP.hpp>
#undef HAVE_CSTDDEF

#include <pagmo/algorithms/nlopt.hpp>

// From UDP
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>  // needed for cereal registration macro
#include <pagmo/types.hpp>
#include <pagmo/utils/gradients_and_hessians.hpp>

namespace pagmo
{
struct trajopt_udp
{
  trajopt_udp(){};

  vector_double fitness(const vector_double& decision_vec) const
  {
    // These are all added to the objective function. Note that there is no manual inflating of constraints like done in
    // the sco solver. In practice I think the constraints would just have a very large coefficient and a steep slope
    std::vector<double> objective;
    for (const sco::CostPtr& cost : costs_)
    {
      objective.push_back(cost->value(decision_vec));
    }
    //    for (const sco::ConstraintPtr& constraint : constraints_)
    //    {
    //      objective.push_back(constraint->violation(decision_vec));
    //    }
    // Sum them all for now to convert to single objective
    vector_double f(1, 0.);
    for (size_t ind = 0; ind < objective.size(); ind++)
      f[0] += objective[ind];

    // Concatenate equality constraints first
    for (const sco::ConstraintPtr& constraint : constraints_)
    {
      if (constraint->type() == sco::ConstraintType::EQ)
        f.push_back(constraint->violation(decision_vec));
    }
    // Then concatenate the inequality constraints
    for (const sco::ConstraintPtr& constraint : constraints_)
    {
      if (constraint->type() == sco::ConstraintType::INEQ)
        f.push_back(constraint->violation(decision_vec));
    }

    return f;
  }

  std::pair<vector_double, vector_double> get_bounds() const { return { lower_, upper_ }; }

  // This method is necessary or else n_obj will be assumed 1. For now it is set to 1 because not many solvers can
  // handle multi objective
  vector_double::size_type get_nobj() const { return 1; }
  // This is the number of equality constraints
  vector_double::size_type get_nec() const
  {
    vector_double::size_type nec = 0;
    for (const sco::ConstraintPtr& constraint : constraints_)
    {
      if (constraint->type() == sco::ConstraintType::EQ)
        nec++;
    }
    return nec;
  }
  // This is the number of inequality constraints
  vector_double::size_type get_nic() const
  {
    vector_double::size_type nic = 0;
    for (const sco::ConstraintPtr& constraint : constraints_)
    {
      if (constraint->type() == sco::ConstraintType::INEQ)
        nic++;
    }
    return nic;
  }

  bool has_gradient() const { return true; }
  vector_double gradient(const vector_double& x) const
  {
    const double epsilon = 1E-5;
    using FTYPE = std::function<vector_double(const vector_double&)>;
    return pagmo::estimate_gradient<FTYPE>(std::bind(&trajopt_udp::fitness, this, std::placeholders::_1), x, epsilon);
    //    return pagmo::estimate_gradient(&trajopt_udp::fitness, x);
  }

  trajopt::TrajOptProbPtr prob_;
  std::vector<sco::ConstraintPtr> constraints_;
  std::vector<sco::CostPtr> costs_;
  std::vector<double> lower_;
  std::vector<double> upper_;

  void setProb(trajopt::TrajOptProbPtr prob)
  {
    prob_ = prob;
    lower_ = prob->getLowerBounds();
    upper_ = prob->getUpperBounds();
    constraints_ = prob->getConstraints();
    costs_ = prob->getCosts();
  }
};

}  // namespace pagmo

using namespace trajopt;
using namespace tesseract;

const std::string ROBOT_DESCRIPTION_PARAM = "robot_description"; /**< Default ROS parameter for robot description */
const std::string ROBOT_SEMANTIC_PARAM = "robot_description_semantic"; /**< Default ROS parameter for robot
                                                                          description */
const std::string TRAJOPT_DESCRIPTION_PARAM =
    "trajopt_description"; /**< Default ROS parameter for trajopt description */

static bool plotting_ = false;
static bool write_to_file_ = false;
static int steps_ = 5;
static std::string method_ = "cpp";
static urdf::ModelInterfaceSharedPtr urdf_model_; /**< URDF Model */
static srdf::ModelSharedPtr srdf_model_;          /**< SRDF Model */
static tesseract_ros::KDLEnvPtr env_;             /**< Trajopt Basic Environment */

TrajOptProbPtr jsonMethod()
{
  ros::NodeHandle nh;
  std::string trajopt_config;

  nh.getParam(TRAJOPT_DESCRIPTION_PARAM, trajopt_config);

  Json::Value root;
  Json::Reader reader;
  bool parse_success = reader.parse(trajopt_config.c_str(), root);
  if (!parse_success)
  {
    ROS_FATAL("Failed to load trajopt json file from ros parameter");
  }

  return ConstructProblem(root, env_);
}

TrajOptProbPtr cppMethod()
{
  ProblemConstructionInfo pci(env_);

  // Populate Basic Info
  pci.basic_info.n_steps = steps_;
  pci.basic_info.manip = "manipulator";
  pci.basic_info.start_fixed = false;
  pci.basic_info.use_time = false;

  // Create Kinematic Object
  pci.kin = pci.env->getManipulator(pci.basic_info.manip);

  // Populate Init Info
  Eigen::VectorXd start_pos = pci.env->getCurrentJointValues(pci.kin->getName());
  Eigen::VectorXd end_pos;
  end_pos.resize(pci.kin->numJoints());
  end_pos << 0.4, 0.2762, 0.0, -1.3348, 0.0, 1.4959, 0.0;

  pci.init_info.type = InitInfo::GIVEN_TRAJ;
  pci.init_info.data = TrajArray(steps_, pci.kin->numJoints());
  for (unsigned idof = 0; idof < pci.kin->numJoints(); ++idof)
  {
    pci.init_info.data.col(idof) = Eigen::VectorXd::LinSpaced(steps_, start_pos[idof], end_pos[idof]);
  }

  // Populate Cost Info
  std::shared_ptr<JointVelTermInfo> jv = std::shared_ptr<JointVelTermInfo>(new JointVelTermInfo);
  jv->coeffs = std::vector<double>(7, 1.0);
  jv->targets = std::vector<double>(7, 0.0);
  jv->first_step = 0;
  jv->last_step = pci.basic_info.n_steps - 1;
  jv->name = "joint_vel";
  jv->term_type = TT_COST;
  pci.cost_infos.push_back(jv);

  std::shared_ptr<CollisionTermInfo> collision = std::shared_ptr<CollisionTermInfo>(new CollisionTermInfo);
  collision->name = "collision";
  collision->term_type = TT_COST;
  collision->continuous = true;
  collision->first_step = 0;
  collision->last_step = pci.basic_info.n_steps - 1;
  collision->gap = 1;
  collision->info = createSafetyMarginDataVector(pci.basic_info.n_steps, 0.025, 20);
  for (auto& info : collision->info)
  {
    info->SetPairSafetyMarginData("base_link", "link_5", 0.05, 10);
    info->SetPairSafetyMarginData("link_3", "link_5", 0.01, 10);
    info->SetPairSafetyMarginData("link_3", "link_6", 0.01, 10);
  }
  pci.cost_infos.push_back(collision);

  // Populate Constraints
  double delta = 0.5 / pci.basic_info.n_steps;
  for (auto i = 0; i < pci.basic_info.n_steps; ++i)
  {
    std::shared_ptr<CartPoseTermInfo> pose = std::shared_ptr<CartPoseTermInfo>(new CartPoseTermInfo);
    pose->term_type = TT_CNT;
    pose->name = "waypoint_cart_" + std::to_string(i);
    pose->link = "tool0";
    pose->timestep = i;
    pose->xyz = Eigen::Vector3d(0.5, -0.2 + delta * i, 0.62);
    pose->wxyz = Eigen::Vector4d(0.0, 0.0, 1.0, 0.0);
    if (i == (pci.basic_info.n_steps - 1) || i == 0)
    {
      pose->pos_coeffs = Eigen::Vector3d(100, 100, 100);
      pose->rot_coeffs = Eigen::Vector3d(100, 100, 100);
    }
    else
    {
      pose->pos_coeffs = Eigen::Vector3d(0, 0, 0);
      pose->rot_coeffs = Eigen::Vector3d(10, 10, 0);
    }
    pci.cnt_infos.push_back(pose);
  }

  return ConstructProblem(pci);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "glass_up_right_plan");
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;

  // Initial setup
  std::string urdf_xml_string, srdf_xml_string;
  nh.getParam(ROBOT_DESCRIPTION_PARAM, urdf_xml_string);
  nh.getParam(ROBOT_SEMANTIC_PARAM, srdf_xml_string);
  urdf_model_ = urdf::parseURDF(urdf_xml_string);

  srdf_model_ = srdf::ModelSharedPtr(new srdf::Model);
  srdf_model_->initString(*urdf_model_, srdf_xml_string);
  env_ = tesseract_ros::KDLEnvPtr(new tesseract_ros::KDLEnv);
  assert(urdf_model_ != nullptr);
  assert(env_ != nullptr);

  bool success = env_->init(urdf_model_, srdf_model_);
  assert(success);

  // Create Plotting tool
  tesseract_ros::ROSBasicPlottingPtr plotter(new tesseract_ros::ROSBasicPlotting(env_));

  // Add sphere
  AttachableObjectPtr obj(new AttachableObject());
  std::shared_ptr<shapes::Sphere> sphere(new shapes::Sphere());
  Eigen::Isometry3d sphere_pose;

  sphere->radius = 0.15;

  sphere_pose.setIdentity();
  sphere_pose.translation() = Eigen::Vector3d(0.5, 0, 0.55);

  obj->name = "sphere_attached";
  obj->visual.shapes.push_back(sphere);
  obj->visual.shape_poses.push_back(sphere_pose);
  obj->collision.shapes.push_back(sphere);
  obj->collision.shape_poses.push_back(sphere_pose);
  obj->collision.collision_object_types.push_back(CollisionObjectType::UseShapeType);

  env_->addAttachableObject(obj);

  AttachedBodyInfo attached_body;
  attached_body.object_name = "sphere_attached";
  attached_body.parent_link_name = "base_link";
  attached_body.transform.setIdentity();
  //  attached_body.touch_links = {}; // This element enables the attached body
  //  to collide with other links

  env_->attachBody(attached_body);

  // Get ROS Parameters
  pnh.param("plotting", plotting_, plotting_);
  pnh.param("write_to_file", write_to_file_, write_to_file_);
  pnh.param<std::string>("method", method_, method_);
  pnh.param<int>("steps", steps_, steps_);

  // Set the robot initial state
  std::unordered_map<std::string, double> ipos;
  ipos["joint_a1"] = -0.4;
  ipos["joint_a2"] = 0.2762;
  ipos["joint_a3"] = 0.0;
  ipos["joint_a4"] = -1.3348;
  ipos["joint_a5"] = 0.0;
  ipos["joint_a6"] = 1.4959;
  ipos["joint_a7"] = 0.0;
  env_->setState(ipos);

  plotter->plotScene();

  // Set Log Level
  util::gLogLevel = util::LevelInfo;

  // Setup Problem
  TrajOptProbPtr prob;
  if (method_ == "cpp")
    prob = cppMethod();
  else
    prob = jsonMethod();

  // Solve Trajectory
  ROS_INFO("glass upright plan example");

  // Solve with Pagmo

  // 1 - Instantiate a pagmo problem constructing it from a UDP (user defined problem).
  pagmo::trajopt_udp temp;
  temp.setProb(prob);
  ROS_INFO("n_obj: %d", temp.get_nobj());
  ROS_INFO("n_ec: %d", temp.get_nec());
  ROS_INFO("n_ic: %d", temp.get_nic());
  pagmo::problem pagmo_prob(temp);
  pagmo_prob.set_c_tol(0.1);

  // 2 - Instantiate a pagmo algorithm
  // NLOPT SLSQP - It appears that when using this algorithm there is no benefit to multiple evolve calls, the algorithm
  // restarts on the second iteration anyway
  pagmo::nlopt nlopt = pagmo::nlopt("auglag");
  nlopt.set_verbosity(1);
  //  nlopt.set_xtol_rel(0.0001);

  // Stopping criteria are specified in the local optimizer
  pagmo::nlopt local_optimizer = pagmo::nlopt("slsqp");
  local_optimizer.set_xtol_rel(0.000001);
  local_optimizer.set_xtol_abs(0.0001);

  nlopt.set_local_optimizer(local_optimizer);

  std::string type = "worst";
  nlopt.set_selection(type);
  nlopt.set_replacement(type);
  pagmo::algorithm algo{ nlopt };

  // Create a population
  pagmo::population population(pagmo_prob, 0);
  trajopt::TrajArray init_traj = prob->GetInitTraj();
  Eigen::VectorXd v = Eigen::Map<const Eigen::VectorXd>(init_traj.data(), init_traj.size());
  std::vector<double> initial_vec(v.data(), v.data() + v.rows() * v.cols());
  population.push_back(initial_vec);

  // 3 - Instantiate an archipelago
  pagmo::archipelago archi{ 1, algo, population };

  for (int i = 0; i < 100; i++)
  {
    ROS_ERROR("Press enter to continue");
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    // 4 - Run the evolution
    archi.evolve(1);

    // 5 - Wait for the evolutions to be finished
    archi.wait_check();

    // 6 - Print the fitness of the best solution in each island
    int best_ind = 0;
    int ind = 0;
    double best_err = std::numeric_limits<double>::max();
    for (const auto& isl : archi)
    {
      std::cout << "f: " << isl.get_population().champion_f()[0] << "\n";
      if (isl.get_population().champion_f()[0] < best_err)
      {
        best_ind = ind;
        best_err = isl.get_population().champion_f()[0];
      }
      ind++;
    }

    std::cout << "Best Joint Values: \n";
    Eigen::MatrixXd traj = trajopt::getTraj(archi[best_ind].get_population().champion_x(), prob->GetVars());
    std::cout << traj << "\n";

    // plot the trajectory in Rviz
    plotter->plotTrajectory(env_->getJointNames(), traj);
  }
}
