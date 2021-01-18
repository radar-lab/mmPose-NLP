#include "mmWaveDataHdl.hpp"
#include "DataHandlerClass.h"

namespace multiradar_mmpose
{

PLUGINLIB_EXPORT_CLASS(multiradar_mmpose::mmWaveDataHdl, nodelet::Nodelet);

mmWaveDataHdl::mmWaveDataHdl() {}

void mmWaveDataHdl::onInit()
{
   ros::NodeHandle private_nh = getPrivateNodeHandle();
   
   std::string mySerialPort;
   int myBaudRate;
   int myMaxAllowedElevationAngleDeg;
   int myMaxAllowedAzimuthAngleDeg;
   /*
   private_nh.getParam("/multiradar_mmpose/data_port", mySerialPort);
   
   private_nh.getParam("/multiradar_mmpose/data_rate", myBaudRate);
   
   if (!(private_nh.getParam("/multiradar_mmpose/max_allowed_elevation_angle_deg", myMaxAllowedElevationAngleDeg)))
   {
      myMaxAllowedElevationAngleDeg = 90;  // Use max angle if none specified
   }

   if (!(private_nh.getParam("/multiradar_mmpose/max_allowed_azimuth_angle_deg", myMaxAllowedAzimuthAngleDeg)))
   {
      myMaxAllowedAzimuthAngleDeg = 90;  // Use max angle if none specified
   }
   */
   std::string myNodeID = ros::this_node::getName();
	ROS_INFO("Node for Radar_%s", myNodeID.c_str());
	std::string path = myNodeID + "/data_port";
//	ROS_INFO("Temp is %s", path.c_str());
	private_nh.getParam(path.c_str(), mySerialPort);
   
	path = myNodeID + "/data_rate";
//	ROS_INFO("Temp is %s", path.c_str());
	private_nh.getParam(path.c_str(), myBaudRate);
   
	path = myNodeID + "/max_allowed_elevation_angle_deg";
//	ROS_INFO("Temp is %s", path.c_str());
	if (!(private_nh.getParam(path.c_str(), myMaxAllowedElevationAngleDeg)))
   {
      myMaxAllowedElevationAngleDeg = 90;  // Use max angle if none specified
   }

//   if (!(private_nh.getParam("/mmWave_Manager/max_allowed_azimuth_angle_deg", myMaxAllowedAzimuthAngleDeg)))
	path = myNodeID + "/max_allowed_azimuth_angle_deg";
//	ROS_INFO("Temp is %s", path.c_str());
	if (!(private_nh.getParam(path.c_str(), myMaxAllowedAzimuthAngleDeg)))
   {
      myMaxAllowedAzimuthAngleDeg = 90;  // Use max angle if none specified
   }
   ROS_INFO("mmWaveDataHdl: data_port = %s", mySerialPort.c_str());
   ROS_INFO("mmWaveDataHdl: data_rate = %d", myBaudRate);
   ROS_INFO("mmWaveDataHdl: max_allowed_elevation_angle_deg = %d", myMaxAllowedElevationAngleDeg);
   ROS_INFO("mmWaveDataHdl: max_allowed_azimuth_angle_deg = %d", myMaxAllowedAzimuthAngleDeg);
   
   DataUARTHandler DataHandler(&private_nh);
   DataHandler.setUARTPort( (char*) mySerialPort.c_str() );
   DataHandler.setBaudRate( myBaudRate );
   DataHandler.setMaxAllowedElevationAngleDeg( myMaxAllowedElevationAngleDeg );
   DataHandler.setMaxAllowedAzimuthAngleDeg( myMaxAllowedAzimuthAngleDeg );
   DataHandler.start();
   
   NODELET_DEBUG("mmWaveDataHdl: Finished onInit function");
}

}



