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
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>

// From UDP
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <limits>
#include <algorithm>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>  // needed for cereal registration macro
#include <pagmo/types.hpp>

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
    for (const sco::ConstraintPtr& constraint : constraints_)
    {
      objective.push_back(constraint->violation(decision_vec));
    }
    // Sum them all for now to convert to single objective
    vector_double f(1, 0.);
    for (size_t ind = 0; ind < objective.size(); ind++)
      f[0] += objective[ind];
    return f;
  }

  std::pair<vector_double, vector_double> get_bounds() const { return { lower_, upper_ }; }

  // This method is necessary or else n_obj will be assumed 1. For now it is set to 1 because not many solvers can
  // handle multi objective
  vector_double::size_type get_nobj() const
  {
    //      return constraints_.size() + costs_.size();
    return 1;
  };

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

static urdf::ModelInterfaceSharedPtr urdf_model_; /**< URDF Model */
static srdf::ModelSharedPtr srdf_model_;          /**< SRDF Model */
static tesseract_ros::KDLEnvPtr env_;             /**< Trajopt Basic Environment */
static int steps_ = 5;


int main(int argc, char** argv)
{
  ROS_ERROR("Press enter to continue");
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  ros::init(argc, argv, "pagmo_plan");
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

  // Set Log Level
  util::gLogLevel = util::LevelInfo;

  ProblemConstructionInfo pci(env_);

  // Populate Basic Info
  pci.basic_info.n_steps = steps_;
  pci.basic_info.manip = "manipulator";
  pci.basic_info.start_fixed = false;
  pci.basic_info.use_time = false;

  // Create Kinematic Object
  pci.kin = pci.env->getManipulator(pci.basic_info.manip);

  // Populate Init Info
  Eigen::VectorXd start_pos;
  Eigen::VectorXd end_pos;
  start_pos.resize(pci.kin->numJoints());
  start_pos << -0.4, 0.2762, 0.0, -1.3348, 0.0, 1.4959, 0.0;
  end_pos.resize(pci.kin->numJoints());
  end_pos << 0.4, 0.2762, 0.0, -1.3348, 0.0, 1.4959, 0.0;

  // This does nothing for Pagmo at the moment
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

  std::shared_ptr<JointPosTermInfo> jp1 = std::shared_ptr<JointPosTermInfo>(new JointPosTermInfo);
  jp1->coeffs = std::vector<double>(7, 100.0);
  jp1->targets = std::vector<double>(7) = {-0.4, 0.2762, 0.0, -1.3348, 0.0, 1.4959, 0.0};
  jp1->first_step = 0;
  jp1->last_step = 0;
  jp1->name = "start_pos";
  jp1->term_type = TT_COST;
  pci.cost_infos.push_back(jp1);

  std::shared_ptr<JointPosTermInfo> jp2 = std::shared_ptr<JointPosTermInfo>(new JointPosTermInfo);
  jp2->coeffs = std::vector<double>(7, 100.0);
  jp2->targets = std::vector<double>(7) = {0.4, 0.2762, 0.0, -1.3348, 0.0, 1.4959, 0.0};
  jp2->first_step = pci.basic_info.n_steps - 1;
  jp2->last_step = pci.basic_info.n_steps - 1;
  jp2->name = "end_pos";
  jp2->term_type = TT_CNT;
  pci.cnt_infos.push_back(jp2);

   TrajOptProbPtr prob = ConstructProblem(pci);

  // Solve with Pagmo
  // 1 - Instantiate a pagmo problem constructing it from a UDP
  // (user defined problem).
  //  pagmo::trajopt_UDP temp2(prob);
  pagmo::trajopt_udp temp;
  temp.setProb(prob);
  pagmo::problem pagmo_prob(temp);

  // 2 - Instantiate a pagmo algorithm
  pagmo::algorithm algo{ pagmo::sade(100) };

  // 3 - Instantiate an archipelago with 16 islands having each 20 individuals
  pagmo::archipelago archi{ 16, algo, pagmo_prob, 20 };

  // 4 - Run the evolution in parallel on the 16 separate islands 10 times.
  archi.evolve(100);

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


}
