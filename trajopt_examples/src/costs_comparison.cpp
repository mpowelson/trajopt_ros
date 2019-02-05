#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <ctime>
#include <ros/package.h>
#include <ros/ros.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/trajectory_costs.hpp>
#include <trajopt/typedefs.hpp>  // should only need this one

#include <trajopt_sco/bpmpd_interface.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>

// No idea which of these I need
#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <jsoncpp/json/json.h>
#include <ros/ros.h>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>
#include <octomap_ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <tesseract_ros/kdl/kdl_chain_kin.h>
#include <tesseract_ros/kdl/kdl_env.h>
#include <tesseract_ros/ros_basic_plotting.h>
#include <trajopt/plot_callback.hpp>
#include <trajopt/problem_description.hpp>
#include <trajopt_utils/config.hpp>
#include <trajopt_utils/logging.hpp>

using namespace trajopt;

class CostsComparisonUtils
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ros::NodeHandle nh_;

  CostsComparisonUtils()
  {
    // Create all the inputs upfront
    num_rows = 10;
    num_cols = 7;
    coeffs = Eigen::VectorXd::Random(num_cols);
    targets = Eigen::VectorXd::Random(num_cols);
    upper_limits = Eigen::VectorXd::Random(num_cols);
    lower_limis = upper_limits / 2.;
    first_step = 0;
    last_step = (num_rows)-1;
    dblvec = createRandomDblVec(num_rows * num_cols);
    model = std::make_shared<sco::BPMPDModel>();
    traj = createVarArray(model, num_rows, num_cols);
  }

  /**
   * @brief Creates a DblVec full of random numbers
   * @param size Length of the vector to create
   * @return
   */
  DblVec createRandomDblVec(const int size)
  {
    DblVec output;
    for (int ind; ind < size; ind++)
    {
      output.push_back(static_cast<double>(rand() / RAND_MAX));
    }
    return output;
  }

  /**
   * @brief Creates a VarArray based on the model passed in
   * @param model - Model that will store the varReps for the vars in the VarArray
   * @param num_rows - Number of rows in the VarArray
   * @param num_cols - Number of columns in the VarArray
   * @return
   */
  trajopt::VarArray createVarArray(std::shared_ptr<sco::BPMPDModel> model, const int& num_rows, const int& num_cols)
  {
    trajopt::VarArray output;

    for (int ind = 0; ind < num_rows; ind++)
    {
      for (int ind2 = 0; ind2 < num_cols; ind2++)
      {
        std::string name = std::to_string(ind) + "_" + std::to_string(ind2);
        model->addVar(name);

        // Copy in VarVector
        output.m_data = model->getVars();
        // Set rows and columns
        output.m_nRow = num_rows;
        output.m_nCol = num_cols;
      }
    }
    return output;
  }

  // Create all the inputs upfront
  int num_rows;
  int num_cols;
  Eigen::VectorXd coeffs;
  Eigen::VectorXd targets;
  Eigen::VectorXd upper_limits;
  Eigen::VectorXd lower_limis;
  int first_step;
  int last_step;
  trajopt::DblVec dblvec;
  std::shared_ptr<sco::BPMPDModel> model;
  trajopt::VarArray traj;
};

//////

int main(int argc, char** argv)
{
  ROS_ERROR("Press enter to continue");
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  ros::init(argc, argv, "trajopt_costs_comparison_unit");
  ros::NodeHandle pnh("~");

  //---------------------------------------------
  //----------- Initialize ----------------------
  //---------------------------------------------
  std::ofstream ofs;
  std::string path = ros::package::getPath("trajopt_examples") + "/cost_comparison_output.csv";
  ofs.open(path, std::ofstream::out | std::ofstream::trunc);
  ofs << "Cost/Cnt,"
      << "Construction Time,"
      << "Value Time,"
      << "Convex Time";

  //---------------------------------------------
  //----------- JointPosEqCost  -----------------
  //---------------------------------------------
  for (int ind = 0; ind < 100; ind++)
  {
    // Create cost comparison utility that contains all of the parameters
    CostsComparisonUtils util;

    // Define outputs
    double construct_time, value_time, convex_time;
    ros::Time tStart;

    // Time construction
    tStart = ros::Time::now();
    JointPosEqCost object(util.traj, util.coeffs, util.targets, util.first_step, util.last_step);
    construct_time = (ros::Time::now() - tStart).toSec();

    // Time the evaluation
    tStart = ros::Time::now();
    object.value(util.dblvec);
    value_time = (ros::Time::now() - tStart).toSec();

    // Time convex fnc
    tStart = ros::Time::now();
    object.convex(util.dblvec, util.model.get());
    convex_time = (ros::Time::now() - tStart).toSec();

    // Display to terminal and print to file
    std::string cost_name = "JointPosEqCost";
    std::cout << cost_name << "   construct_time: " << construct_time << "    value_time: " << value_time
              << "    convex time: " << convex_time << "\n";
    ofs << cost_name << "," << construct_time << "," << value_time  << "," << convex_time << "\n" << std::flush;
  }
  //---------------------------------------------
  //----------- JointVelEqCost ------------------
  //---------------------------------------------
  for (int ind = 0; ind < 100; ind++)
  {
    // Create cost comparison utility that contains all of the parameters
    CostsComparisonUtils util;

    // Define outputs
    double construct_time, value_time, convex_time;
    ros::Time tStart;

    // Time construction
    tStart = ros::Time::now();
    JointVelEqCost object(util.traj, util.coeffs, util.targets, util.first_step, util.last_step);
    construct_time = (ros::Time::now() - tStart).toSec();

    // Time the evaluation
    tStart = ros::Time::now();
    object.value(util.dblvec);
    value_time = (ros::Time::now() - tStart).toSec();

    // Time convex fnc
    tStart = ros::Time::now();
    object.convex(util.dblvec, util.model.get());
    convex_time = (ros::Time::now() - tStart).toSec();

    // Display to terminal and print to file
    std::string cost_name = "JointVelEqCost";
    std::cout << cost_name << "   construct_time: " << construct_time << "    value_time: " << value_time
              << "    convex time: " << convex_time << "\n";
    ofs << cost_name << "," << construct_time << "," << value_time  << "," << convex_time << "\n" << std::flush;
  }
  //---------------------------------------------
  //-----------  JointAccEqCost  ----------------
  //---------------------------------------------
  for (int ind = 0; ind < 100; ind++)
  {
    // Create cost comparison utility that contains all of the parameters
    CostsComparisonUtils util;

    // Define outputs
    double construct_time, value_time, convex_time;
    ros::Time tStart;

    // Time construction
    tStart = ros::Time::now();
    JointAccEqCost object(util.traj, util.coeffs, util.targets, util.first_step, util.last_step);
    construct_time = (ros::Time::now() - tStart).toSec();

    // Time the evaluation
    tStart = ros::Time::now();
    object.value(util.dblvec);
    value_time = (ros::Time::now() - tStart).toSec();

    // Time convex fnc
    tStart = ros::Time::now();
    object.convex(util.dblvec, util.model.get());
    convex_time = (ros::Time::now() - tStart).toSec();

    // Display to terminal and print to file
    std::string cost_name = "JointAccEqCost";
    std::cout << cost_name << "   construct_time: " << construct_time << "    value_time: " << value_time
              << "    convex time: " << convex_time << "\n";
    ofs << cost_name << "," << construct_time << "," << value_time  << "," << convex_time << "\n" << std::flush;
  }
  //---------------------------------------------
  //-----------  JointJerkEqCost ----------------
  //---------------------------------------------
  for (int ind = 0; ind < 100; ind++)
  {
    // Create cost comparison utility that contains all of the parameters
    CostsComparisonUtils util;

    // Define outputs
    double construct_time, value_time, convex_time;
    ros::Time tStart;

    // Time construction
    tStart = ros::Time::now();
    JointJerkEqCost object(util.traj, util.coeffs, util.targets, util.first_step, util.last_step);
    construct_time = (ros::Time::now() - tStart).toSec();

    // Time the evaluation
    tStart = ros::Time::now();
    object.value(util.dblvec);
    value_time = (ros::Time::now() - tStart).toSec();

    // Time convex fnc
    tStart = ros::Time::now();
    object.convex(util.dblvec, util.model.get());
    convex_time = (ros::Time::now() - tStart).toSec();

    // Display to terminal and print to file
    std::string cost_name = "JointJerkEqCost";
    std::cout << cost_name << "   construct_time: " << construct_time << "    value_time: " << value_time
              << "    convex time: " << convex_time << "\n";
    ofs << cost_name << "," << construct_time << "," << value_time  << "," << convex_time << "\n" << std::flush;
  }


  //---------------------------------------------
  //----------- Close  ----------------------
  //---------------------------------------------
  ofs.close();

  return 0;
}
