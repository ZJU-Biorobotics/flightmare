#include "flightros/pilot/flight_pilot.hpp"

namespace flightros {

FlightPilot::FlightPilot(const ros::NodeHandle &nh, const ros::NodeHandle &pnh)
  : nh_(nh),
    pnh_(pnh),
    scene_id_(UnityScene::WAREHOUSE),
    unity_ready_(false),
    unity_render_(false),
    receive_id_(0),
    main_loop_freq_(50.0) {
  // load parameters
  if (!loadParams()) {
    ROS_WARN("[%s] Could not load all parameters.",
             pnh_.getNamespace().c_str());
  } else {
    ROS_INFO("[%s] Loaded all parameters.", pnh_.getNamespace().c_str());
  }

  // quad initialization
  quad_ptr_ = std::make_shared<Quadrotor>();

  // add mono camera
  rgb_camera_ = std::make_shared<RGBCamera>();
  Vector<3> B_r_BC(0.3, 0.0, 0.3);
  Matrix<3, 3> R_BC = Quaternion(0.707, 0.0, 0.0, -0.707).toRotationMatrix();
  std::cout << R_BC << std::endl;
  image_transport::ImageTransport it(pnh);

  depth_pub = it.advertise("/depth", 1);
  rgb_camera_->setFOV(90);
  rgb_camera_->setWidth(256);
  rgb_camera_->setHeight(192);
  rgb_camera_->setRelPose(B_r_BC, R_BC);
  rgb_camera_->setPostProcesscing(std::vector<bool>{
    true, false, false});  // depth, segmentation, optical flow
  quad_ptr_->addRGBCamera(rgb_camera_);

  // initialization
  quad_state_.setZero();
  quad_ptr_->reset(quad_state_);

  // Initialize gates
  std::string object_id = "gate";
  std::string prefab_id = "rpg_gate";
  gate0 = std::make_shared<StaticObject>(object_id, prefab_id);
  gate0->setPosition(Eigen::Vector3f(0, 10, 2.5));
  gate0->setQuaternion(
    Quaternion(std::cos(1 * M_PI_4), 0.0, 0.0, std::sin(1 * M_PI_4)));

  // std::string object_id_2 = "unity_gate_2";
  // gate_2 = std::make_shared<StaticGate>(object_id_2);
  // gate_2->setPosition(Eigen::Vector3f(0, -10, 2.5));
  // gate_2->setQuaternion(
  //   Quaternion(std::cos(1 * M_PI_4), 0.0, 0.0, std::sin(1 * M_PI_4)));

  // std::string object_id_3 = "moving_gate";
  // gate_3 = std::make_shared<StaticGate>(object_id_3);
  // gate_3->setPosition(Eigen::Vector3f(5, 0, 2.5));
  // gate_3->setQuaternion(Quaternion(0.0, 0.0, 0.0, 1.0));

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      std::string id = "gate" + std::to_string(i * 7 + j);
      std::shared_ptr<StaticObject> gate =
        std::make_shared<StaticObject>(id, "+++");
      gate->setPosition(
        Eigen::Vector3f(5.0 * (i - 3), 5.0 * (j - 3) - 15, 2.5));
      // gate->setQuaternion(
      //   Quaternion(std::cos(M_PI_2 * (7 * i + j) / 49 - M_PI_4), 0.0, 0.0,
      //              std::sin(M_PI_2 * (i * 7 + j) / 49 - M_PI_4)));
      gate->setQuaternion(
        Quaternion(std::cos(1 * 0), 0.0, 0.0, std::sin(1 * 0)));
      gate_vec.push_back(gate);
      std::cout << &gate << std::endl;
    }
  }


  // initialize subscriber call backs
  sub_state_est_ = nh_.subscribe("flight_pilot/state_estimate", 1,
                                 &FlightPilot::poseCallback, this);

  timer_main_loop_ = nh_.createTimer(ros::Rate(main_loop_freq_),
                                     &FlightPilot::mainLoopCallback, this);


  // wait until the gazebo and unity are loaded
  ros::Duration(5.0).sleep();

  // connect unity
  setUnity(unity_render_);
  connectUnity();
}

FlightPilot::~FlightPilot() {}

void FlightPilot::poseCallback(const nav_msgs::Odometry::ConstPtr &msg) {
  quad_state_.x[QS::POSX] = (Scalar)msg->pose.pose.position.x;
  quad_state_.x[QS::POSY] = (Scalar)msg->pose.pose.position.y - 35;
  quad_state_.x[QS::POSZ] = (Scalar)msg->pose.pose.position.z;
  quad_state_.x[QS::ATTW] = (Scalar)msg->pose.pose.orientation.w;
  quad_state_.x[QS::ATTX] = (Scalar)msg->pose.pose.orientation.x;
  quad_state_.x[QS::ATTY] = (Scalar)msg->pose.pose.orientation.y;
  quad_state_.x[QS::ATTZ] = (Scalar)msg->pose.pose.orientation.z;
  //
  quad_ptr_->setState(quad_state_);

  cv::Mat img;
  ros::Time timestamp = ros::Time::now();
  rgb_camera_->getDepthMap(img);
  sensor_msgs::ImagePtr depth_msg =
    cv_bridge::CvImage(std_msgs::Header(), "32FC1", img).toImageMsg();
  depth_msg->header.stamp = timestamp;
  depth_pub.publish(depth_msg);

  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(0);
    unity_bridge_ptr_->handleOutput();

    if (quad_ptr_->getCollision()) {
      // collision happened
      ROS_INFO("COLLISION");
    }
  }
}

void FlightPilot::mainLoopCallback(const ros::TimerEvent &event) {
  // empty
}

bool FlightPilot::setUnity(const bool render) {
  unity_render_ = render;
  if (unity_render_ && unity_bridge_ptr_ == nullptr) {
    // create unity bridge
    unity_bridge_ptr_ = UnityBridge::getInstance();
    unity_bridge_ptr_->addQuadrotor(quad_ptr_);
    // unity_bridge_ptr_->addStaticObject(gate_1);
    // unity_bridge_ptr_->addStaticObject(gate_2);
    // unity_bridge_ptr_->addStaticObject(gate_3);
    for (int i = 0; i < gate_vec.size(); i++) {
      unity_bridge_ptr_->addStaticObject(gate_vec[i]);
    }
    ROS_INFO("[%s] Unity Bridge is created.", pnh_.getNamespace().c_str());
  }
  return true;
}

bool FlightPilot::connectUnity() {
  if (!unity_render_ || unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}

bool FlightPilot::loadParams(void) {
  // load parameters
  quadrotor_common::getParam("main_loop_freq", main_loop_freq_, pnh_);
  quadrotor_common::getParam("unity_render", unity_render_, pnh_);

  return true;
}

}  // namespace flightros