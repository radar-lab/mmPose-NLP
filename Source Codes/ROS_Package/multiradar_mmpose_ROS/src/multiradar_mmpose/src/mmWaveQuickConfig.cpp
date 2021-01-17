#include "ros/ros.h"
#include "multiradar_mmpose/mmWaveCLI.h"
#include <cstdlib>
#include <fstream>
#include <stdio.h>
#include <regex>
#include "parameter_parser.h"

std::string idx_str(int idx, std::string s)
{
	std::regex words_regex("[^\\s]+");
	auto words_begin = std::sregex_iterator(s.begin(), s.end(), words_regex);
	for (int i = 0; i < idx; i++, words_begin++);
	std::smatch tmp = *words_begin;
	std::string tmp_src = tmp.str();
//	std::cout << tmp_src << std::endl;

	return tmp_src;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "mmWaveQuickConfig");
  //std::string reset_str("reset");

  if (argc != 2) {
    ROS_INFO("mmWaveQuickConfig: usage: mmWaveQuickConfig /file_directory/params.cfg");
    return 1;
  } 
  else
  { 
    ROS_INFO("mmWaveQuickConfig: Configuring mmWave device using config file: %s", argv[1]);
  }
  ros::NodeHandle nh;
  std::string myNodeID = ros::this_node::getName();
	ROS_INFO("Node for Radar_%s", myNodeID.c_str());
	ROS_INFO("Last char is %c", myNodeID[myNodeID.length()-1]);
  std::string path = "/mmWaveCommSrv_" + myNodeID.substr(myNodeID.length()-1) + "/mmWaveCLI";
	ROS_INFO("ServiceClient is %s", path.c_str());
  
  ros::ServiceClient client = nh.serviceClient<multiradar_mmpose::mmWaveCLI>(path.c_str());
  multiradar_mmpose::mmWaveCLI srv;  
  std::ifstream myParams;
  multiradar_mmpose::parameter_parser p;
  //wait for service to become available
  ros::service::waitForService(path.c_str(), 100000); 
  
  myParams.open(argv[1]);
  
  if (myParams.is_open()) {
    while( std::getline(myParams, srv.request.comm)) {
      // Remove Windows carriage-return if present
      srv.request.comm.erase(std::remove(srv.request.comm.begin(), srv.request.comm.end(), '\r'), srv.request.comm.end());
      // Ignore comment lines (first non-space char is '%') or blank lines
      if (!(std::regex_match(srv.request.comm, std::regex("^\\s*%.*")) || std::regex_match(srv.request.comm, std::regex("^\\s*")))) {
        // ROS_INFO("mmWaveQuickConfig: Sending command: '%s'", srv.request.comm.c_str() );
        if (client.call(srv)) {
          if (std::regex_search(srv.response.resp, std::regex("Done"))) {
            // ROS_INFO("mmWaveQuickConfig: Command successful (mmWave sensor responded with 'Done')");            
            p.params_parser(srv, nh);
          } /*else {
            if(reset_str.compare(srv.request.comm.c_str()) != 0 )
            {
              ROS_ERROR("mmWaveQuickConfig: Command failed (mmWave sensor did not respond with 'Done')");
              ROS_ERROR("mmWaveQuickConfig: Response: '%s'", srv.response.resp.c_str() );
              return 1;
            }
          }*/
        } else {
          ROS_ERROR("mmWaveQuickConfig: Failed to call service mmWaveCLI");
          ROS_ERROR("%s", srv.request.comm.c_str() );
          return 1;
        }
      }
    }
    p.cal_params(nh);
    myParams.close();
  } else {
    ROS_ERROR("mmWaveQuickConfig: Failed to open File %s", argv[1]);
    return 1;
  }
  ROS_INFO("mmWaveQuickConfig: mmWaveQuickConfig will now terminate. Done configuring mmWave device using config file: %s", argv[1]);
  return 0;
}