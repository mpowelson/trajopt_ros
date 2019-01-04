#include <tesseract_core/basic_types.h>
#include <tesseract_ros/kdl/kdl_env.h>
#include <tesseract_ros/ros_basic_plotting.h>
#include <tesseract_planning/trajopt/trajopt_planner.h>

#include <urdf_parser/urdf_parser.h>
#include <fstream>

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
  double box_side;  //, box_x, box_y;
  nh.getParam("robot_description", urdf_xml_string);
  nh.getParam("robot_description_semantic", srdf_xml_string);
  nh.getParam("box_side", box_side);
  nh.getParam("box_parent_link", box_parent_link);

  // Define Benchmark parameters
  std::vector<int> num_steps{ 10, 20, 40, 80 };
  std::vector<double> pick_points_x{ -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15 };
  std::vector<double> pick_points_y{ 0.35, 0.40, 0.45, 0.50, 0.55 };
  std::vector<double> place_points{ -0.148856,  /*-0.11577689,*/ -0.08269778, /* -0.04961867,*/ -0.01653956,
                                    0.01653956, /*0.04961867,*/ 0.08269778,   /*0.11577689,*/ 0.148856 };

  unsigned long num_iter = pick_points_x.size() * pick_points_y.size() * place_points.size() * num_steps.size();
  std::string solver_string = "Unspecified";
  sco::ModelType solver = sco::ModelType::BPMPD;
  std::cout << "Processing " << num_iter << " iterations";

  std::ofstream ofs;
  std::string path = ros::package::getPath("trajopt_examples") + "/output.csv";
  ofs.open(path, std::ofstream::out | std::ofstream::trunc);
  ofs << "Iteration,"
      << "Total Iterations,"
      << "Pick point x,"
      << "Pick point y,"
      << "Place point,"
      << "steps_per_phase,"
      << "solver,"
      << "pick_time,"
      << "place_time,"
      << "pick_collision,"
      << "place_collision\n";

  // Define outputs
  double pick_time;
  double place_time;
  bool pick_collision;
  bool place_collision;

  unsigned long ind = 0;
  for (int num_step : num_steps)
  {
    for (double box_x : pick_points_x)
    {
      for (double box_y : pick_points_y)
      {
        for (double place_point : place_points)
        {
          ind += 1;
          steps_per_phase = num_step;

          // Initialize the environment
          urdf::ModelInterfaceSharedPtr urdf_model = urdf::parseURDF(urdf_xml_string);
          srdf::ModelSharedPtr srdf_model = srdf::ModelSharedPtr(new srdf::Model);
          srdf_model->initString(*urdf_model, srdf_xml_string);
          tesseract::tesseract_ros::KDLEnvPtr env(new tesseract::tesseract_ros::KDLEnv);
          assert(urdf_model != nullptr);
          assert(env != nullptr);

          bool success = env->init(urdf_model, srdf_model);
          assert(success);

          ROS_ERROR("Press enter to continue");
          std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

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

          // Attach the simulated box
          tesseract::AttachableObjectPtr obj(new tesseract::AttachableObject());
          std::shared_ptr<shapes::Box> box(new shapes::Box());
          Eigen::Isometry3d box_pose = Eigen::Isometry3d::Identity();

          box->size[0] = box_side;
          box->size[1] = box_side;
          box->size[2] = box_side;

          obj->name = "box";
          obj->visual.shapes.push_back(box);
          obj->visual.shape_poses.push_back(box_pose);
          obj->collision.shapes.push_back(box);
          obj->collision.shape_poses.push_back(box_pose);
          obj->collision.collision_object_types.push_back(tesseract::CollisionObjectType::UseShapeType);

          env->addAttachableObject(obj);

          // Move box to correct location
          tesseract::AttachedBodyInfo attached_body;
          Eigen::Isometry3d object_pose = Eigen::Isometry3d::Identity();
          object_pose.translation() += Eigen::Vector3d(box_x, box_y, box_side / 2.0);
          attached_body.object_name = "box";
          attached_body.parent_link_name = box_parent_link;
          attached_body.transform = object_pose;

          env->attachBody(attached_body);

          // Send the initial trajectory for plotting
          tesseract::tesseract_ros::ROSBasicPlotting plotter(env);
          Eigen::RowVectorXd init_pos = env->getCurrentJointValues();
          //                  plotter.plotTrajectory(env->getJointNames(),
          //                  init_pos.leftCols(env->getJointNames().size()));

          ////////////
          /// PICK ///
          ////////////

          //  ROS_ERROR("Press enter to continue");
          //  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

          // Create the planner and the responses that will store the results
          tesseract::tesseract_planning::TrajOptPlanner planner;
          tesseract::tesseract_planning::PlannerResponse planning_response;
          tesseract::tesseract_planning::PlannerResponse planning_response_place;

          // Choose the manipulator and end effector link
          std::string manip = "Manipulator";
          std::string end_effector = "iiwa_link_ee";

          // Define the final pose (on top of the box)
          Eigen::Isometry3d final_pose;
          Eigen::Quaterniond orientation(0.0, 0.0, 1.0, 0.0);
          final_pose.translation() = Eigen::Vector3d(0.0, 0.0, 0.0);
          final_pose.linear() = orientation.matrix();
          final_pose.translation() += Eigen::Vector3d(box_x, box_y, box_side + 0.77153);  // Offset for the table

          // Define the approach pose
          Eigen::Isometry3d approach_pose = final_pose;
          approach_pose.translation() += Eigen::Vector3d(0.0, 0.0, 0.15);

          // Create the problem construction info
          trajopt::ProblemConstructionInfo pci(env);

          pci.kin = env->getManipulator(manip);

          pci.basic_info.n_steps = steps_per_phase * 2;
          pci.basic_info.manip = manip;
          pci.basic_info.dt_lower_lim = 0.2;
          pci.basic_info.dt_upper_lim = .5;
          pci.basic_info.start_fixed = true;
          pci.basic_info.use_time = false;
          pci.basic_info.convex_solver = solver;

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
            collision->last_step = pci.basic_info.n_steps - 2;
            collision->gap = 1;
            collision->info = trajopt::createSafetyMarginDataVector(pci.basic_info.n_steps, 0.025, 40);
            pci.cost_infos.push_back(collision);
          }

          // Add a velocity cost
          if (true)
          {
            std::shared_ptr<trajopt::JointVelTermInfo> jv(new trajopt::JointVelTermInfo);

            // Taken from iiwa documentation (radians/s) and scaled by 0.8
            std::vector<double> vel_lower_lim{ 1.71 * -0.8, 1.71 * -0.8, 1.75 * -0.8, 2.27 * -0.8,
                                               2.44 * -0.8, 3.14 * -0.8, 3.14 * -0.8 };
            std::vector<double> vel_upper_lim{ 1.71 * 0.8, 1.71 * 0.8, 1.75 * 0.8, 2.27 * 0.8,
                                               2.44 * 0.8, 3.14 * 0.8, 3.14 * 0.8 };

            jv->targets = std::vector<double>(7, 0.0);
            jv->coeffs = std::vector<double>(7, 5.0);
            //  jv->lower_tols = vel_lower_lim;
            //  jv->upper_tols = vel_upper_lim;
            jv->term_type = trajopt::TT_COST;
            jv->first_step = 0;
            jv->last_step = pci.basic_info.n_steps - 1;
            jv->name = "joint_velocity_cost";
            pci.cost_infos.push_back(jv);
          }

          // Add cartesian pose cnt at the approach point
          if (true)
          {
            Eigen::Quaterniond rotation(approach_pose.linear());
            std::shared_ptr<trajopt::CartPoseTermInfo> pose_constraint =
                std::shared_ptr<trajopt::CartPoseTermInfo>(new trajopt::CartPoseTermInfo);
            pose_constraint->term_type = trajopt::TT_CNT;
            pose_constraint->link = end_effector;
            pose_constraint->timestep = steps_per_phase;
            pose_constraint->xyz = approach_pose.translation();

            pose_constraint->wxyz = Eigen::Vector4d(rotation.w(), rotation.x(), rotation.y(), rotation.z());
            pose_constraint->pos_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->rot_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->name = "pose_" + std::to_string(steps_per_phase);
            pci.cnt_infos.push_back(pose_constraint);
          }

          // Add cartesian pose cnt at the final point
          if (true)
          {
            Eigen::Quaterniond rotation(final_pose.linear());
            std::shared_ptr<trajopt::CartPoseTermInfo> pose_constraint =
                std::shared_ptr<trajopt::CartPoseTermInfo>(new trajopt::CartPoseTermInfo);
            pose_constraint->term_type = trajopt::TT_CNT;
            pose_constraint->link = end_effector;
            pose_constraint->timestep = 2 * steps_per_phase - 1;
            pose_constraint->xyz = final_pose.translation();

            pose_constraint->wxyz = Eigen::Vector4d(rotation.w(), rotation.x(), rotation.y(), rotation.z());
            pose_constraint->pos_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->rot_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->name = "pose_" + std::to_string(2 * steps_per_phase - 1);
            pci.cnt_infos.push_back(pose_constraint);
          }

          // Create the pick problem
          trajopt::TrajOptProbPtr pick_prob = ConstructProblem(pci);

          // Set the optimization parameters (Most are being left as defaults)
          tesseract::tesseract_planning::TrajOptPlannerConfig config(pick_prob);
          config.params.max_iter = 100;

          // Solve problem. Results are stored in the response
          ros::Time tStart1 = ros::Time::now();
          planner.solve(planning_response, config);
          pick_time = (ros::Time::now() - tStart1).toSec();

          if (true)
          {
            // Check and report collisions
            std::vector<tesseract::ContactResultMap> collisions;
            tesseract::ContinuousContactManagerBasePtr manager = config.prob->GetEnv()->getContinuousContactManager();
            manager->setActiveCollisionObjects(config.prob->GetKin()->getLinkNames());
            manager->setContactDistanceThreshold(0);
            collisions.clear();
            pick_collision = tesseract::continuousCollisionCheckTrajectory(
                *manager, *config.prob->GetEnv(), *config.prob->GetKin(), planning_response.trajectory, collisions);
          }
          plotter.plotTrajectory(env->getJointNames(),
                                 planning_response.trajectory.leftCols(env->getJointNames().size()));

          /////////////
          /// PLACE ///
          /////////////

          ROS_ERROR("Press enter to continue");
          std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

          // Detach the simulated box from the world and attach to the end effector
          env->detachBody("box");
          attached_body.parent_link_name = end_effector;
          attached_body.transform.translation() = Eigen::Vector3d(0, 0, box_side / 2.0);
          attached_body.touch_links = { "iiwa_link_ee", end_effector };   // allow the box to contact the end effector
          attached_body.touch_links = { "workcell_base", end_effector };  // allow the box to contact the table (since
                                                                          // it's sitting on it)

          env->attachBody(attached_body);

          // Set the current state to the last state of the pick trajectory
          env->setState(env->getJointNames(), planning_response.trajectory.bottomRows(1).transpose());

          // Retreat to the approach pose
          Eigen::Isometry3d retreat_pose = approach_pose;

          // Define some place locations.
          Eigen::Isometry3d middle_right_shelf, middle_left_shelf, middle_shelf;
          middle_right_shelf.linear() = Eigen::Quaterniond(0, 0, 0.7071068, 0.7071068).matrix();
          middle_right_shelf.translation() = Eigen::Vector3d(0.148856, 0.73085, 1.16);
          middle_left_shelf.linear() = Eigen::Quaterniond(0, 0, 0.7071068, 0.7071068).matrix();
          middle_left_shelf.translation() = Eigen::Vector3d(-0.148856, 0.73085, 1.16);

          middle_shelf.linear() = Eigen::Quaterniond(0, 0, 0.7071068, 0.7071068).matrix();
          middle_shelf.translation() = Eigen::Vector3d(place_point, 0.70085, 1.16);

          // Set the target pose to middle_left_shelf
          final_pose = middle_shelf;

          // Setup approach for place move
          approach_pose = final_pose;
          approach_pose.translation() += Eigen::Vector3d(0.0, -0.2, 0);

          // Create the problem construction info
          trajopt::ProblemConstructionInfo pci_place(env);

          pci_place.kin = env->getManipulator(manip);

          pci_place.basic_info.n_steps = steps_per_phase * 3;
          pci_place.basic_info.manip = manip;
          pci_place.basic_info.dt_lower_lim = 0.2;
          pci_place.basic_info.dt_upper_lim = .5;
          pci_place.basic_info.start_fixed = true;
          pci_place.basic_info.use_time = false;
          pci_place.basic_info.convex_solver = solver;

          pci_place.init_info.type = trajopt::InitInfo::STATIONARY;
          pci_place.init_info.dt = 0.5;

          // Add a collision cost
          if (true)
          {
            std::shared_ptr<trajopt::CollisionTermInfo> collision(new trajopt::CollisionTermInfo);
            collision->name = "collision";
            collision->term_type = trajopt::TT_COST;
            collision->continuous = true;
            collision->first_step = 0;
            collision->last_step = pci_place.basic_info.n_steps - 1;
            collision->gap = 1;
            collision->info = trajopt::createSafetyMarginDataVector(pci_place.basic_info.n_steps, 0.075, 40);
            pci_place.cost_infos.push_back(collision);
          }

          // Add a velocity cost
          if (true)
          {
            std::shared_ptr<trajopt::JointVelTermInfo> jv(new trajopt::JointVelTermInfo);

            // Taken from iiwa documentation (radians/s) and scaled by 0.8
            std::vector<double> vel_lower_lim{ 1.71 * -0.8, 1.71 * -0.8, 1.75 * -0.8, 2.27 * -0.8,
                                               2.44 * -0.8, 3.14 * -0.8, 3.14 * -0.8 };
            std::vector<double> vel_upper_lim{ 1.71 * 0.8, 1.71 * 0.8, 1.75 * 0.8, 2.27 * 0.8,
                                               2.44 * 0.8, 3.14 * 0.8, 3.14 * 0.8 };

            jv->targets = std::vector<double>(7, 0.0);
            jv->coeffs = std::vector<double>(7, 5.0);
            //  jv->lower_tols = vel_lower_lim;
            //  jv->upper_tols = vel_upper_lim;
            jv->term_type = trajopt::TT_COST;
            jv->first_step = 0;
            jv->last_step = pci_place.basic_info.n_steps - 1;
            jv->name = "joint_velocity_cost";
            pci_place.cost_infos.push_back(jv);
          }

          // Add cartesian pose cnt at the retreat point
          if (true)
          {
            Eigen::Quaterniond rotation(retreat_pose.linear());
            std::shared_ptr<trajopt::CartPoseTermInfo> pose_constraint =
                std::shared_ptr<trajopt::CartPoseTermInfo>(new trajopt::CartPoseTermInfo);
            pose_constraint->term_type = trajopt::TT_CNT;
            pose_constraint->link = end_effector;
            pose_constraint->timestep = steps_per_phase - 1;
            pose_constraint->xyz = retreat_pose.translation();

            pose_constraint->wxyz = Eigen::Vector4d(rotation.w(), rotation.x(), rotation.y(), rotation.z());
            pose_constraint->pos_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->rot_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->name = "pose_" + std::to_string(steps_per_phase - 1);
            pci_place.cnt_infos.push_back(pose_constraint);
          }

          // Add cartesian pose cnt at the approach point
          if (true)
          {
            Eigen::Quaterniond rotation(approach_pose.linear());
            std::shared_ptr<trajopt::CartPoseTermInfo> pose_constraint =
                std::shared_ptr<trajopt::CartPoseTermInfo>(new trajopt::CartPoseTermInfo);
            pose_constraint->term_type = trajopt::TT_CNT;
            pose_constraint->link = end_effector;
            pose_constraint->timestep = 2 * steps_per_phase - 1;
            pose_constraint->xyz = approach_pose.translation();

            pose_constraint->wxyz = Eigen::Vector4d(rotation.w(), rotation.x(), rotation.y(), rotation.z());
            pose_constraint->pos_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->rot_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->name = "pose_" + std::to_string(2 * steps_per_phase - 1);
            pci_place.cnt_infos.push_back(pose_constraint);
          }

          // Add cartesian pose cnt at the final point
          if (true)
          {
            Eigen::Quaterniond rotation(final_pose.linear());
            std::shared_ptr<trajopt::CartPoseTermInfo> pose_constraint =
                std::shared_ptr<trajopt::CartPoseTermInfo>(new trajopt::CartPoseTermInfo);
            pose_constraint->term_type = trajopt::TT_CNT;
            pose_constraint->link = end_effector;
            pose_constraint->timestep = 3 * steps_per_phase - 1;
            pose_constraint->xyz = final_pose.translation();

            pose_constraint->wxyz = Eigen::Vector4d(rotation.w(), rotation.x(), rotation.y(), rotation.z());
            pose_constraint->pos_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->rot_coeffs = Eigen::Vector3d(10.0, 10.0, 10.0);
            pose_constraint->name = "pose_" + std::to_string(3 * steps_per_phase - 1);
            pci_place.cnt_infos.push_back(pose_constraint);
          }

          // Create the place problem
          trajopt::TrajOptProbPtr place_prob = ConstructProblem(pci_place);

          // Set the optimization parameters
          tesseract::tesseract_planning::TrajOptPlannerConfig config_place(place_prob);
          config_place.params.max_iter = 100;

          // Solve problem
          ros::Time tStart2 = ros::Time::now();
          planner.solve(planning_response_place, config_place);
          place_time = (ros::Time::now() - tStart2).toSec();
          std::cout << planning_response_place.status_code << "  " << planning_response_place.status_description
                    << '\n';

          if (true)
          {
            // Check and report collisions
            std::vector<tesseract::ContactResultMap> collisions;
            tesseract::ContinuousContactManagerBasePtr manager =
                config_place.prob->GetEnv()->getContinuousContactManager();
            manager->setActiveCollisionObjects(config_place.prob->GetKin()->getLinkNames());
            manager->setContactDistanceThreshold(0);
            collisions.clear();
            place_collision = tesseract::continuousCollisionCheckTrajectory(*manager,
                                                                            *config_place.prob->GetEnv(),
                                                                            *config_place.prob->GetKin(),
                                                                            planning_response_place.trajectory,
                                                                            collisions);
          }

          plotter.plotTrajectory(env->getJointNames(),
                                 planning_response_place.trajectory.leftCols(env->getJointNames().size()));

          // Print results to terminal
          std::cout << "Iter: " << ind << "/" << num_iter << " Pick point: " << box_x << ", " << box_y
                    << "  Place point: " << place_point << " steps_per_phase: " << steps_per_phase << '\n';

          // Save results to file
          ofs << ind << ',' << num_iter << ',' << box_x << ',' << box_y << ',' << place_point << ',' << steps_per_phase
              << ',' << solver << ',' << pick_time << ',' << place_time << ',' << pick_collision << ','
              << place_collision << '\n'
              << std::flush;

        }  // place_point
      }    // pick point
    }      // pick point
  }        // num_steps

  ofs.close();
  ROS_INFO("Done");
  ros::spin();
}
