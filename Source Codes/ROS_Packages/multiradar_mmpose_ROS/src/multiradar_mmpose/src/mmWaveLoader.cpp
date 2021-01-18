#include "ros/ros.h"
#include "nodelet/loader.h"

int main(int argc, char **argv)
{

  ros::init(argc, argv, "mmWave_Manager");
  
  nodelet::Loader manager(true);
  
  nodelet::M_string remap(ros::names::getRemappings());
  
  nodelet::V_string nargv;
  
//  manager.load("mmWaveCommSrv", "multiradar_mmpose/mmWaveCommSrv", remap, nargv);
	std::string myNodeID = ros::this_node::getName();
	ROS_INFO("Node for Radar_%s", myNodeID.c_str());
	ROS_INFO("Last char is %c", myNodeID[myNodeID.length()-1]);
	std::string path = "/mmWaveCommSrv_" + myNodeID.substr(myNodeID.length()-1);
	ROS_INFO("manager.load mmWaveCommSrv is %s", path.c_str());
	manager.load(path.c_str(), "multiradar_mmpose/mmWaveCommSrv", remap, nargv);
  
//  manager.load("mmWaveDataHdl", "multiradar_mmpose/mmWaveDataHdl", remap, nargv);
	path = "/mmWaveDataHdl_" + myNodeID.substr(myNodeID.length()-1);
	ROS_INFO("manager.load mmWaveDataHdl is %s", path.c_str());
	manager.load(path.c_str(), "multiradar_mmpose/mmWaveDataHdl", remap, nargv);
  
  ros::spin();
  
  return 0;
  }
