#include "parameter_parser.h"

namespace multiradar_mmpose {

PLUGINLIB_EXPORT_CLASS(multiradar_mmpose::parameter_parser, nodelet::Nodelet);

parameter_parser::parameter_parser() {}

void parameter_parser::onInit() {}

void parameter_parser::params_parser(multiradar_mmpose::mmWaveCLI &srv, ros::NodeHandle &nh) {

//   ROS_ERROR("%s",srv.request.comm.c_str());
//   ROS_ERROR("%s",srv.response.resp.c_str());
  std::string myNodeID = ros::this_node::getName();
  std::vector <std::string> v;
  std::string s = srv.request.comm.c_str(); 
  std::istringstream ss(s);
  std::string token;
  std::string req;
  int i = 0;
  while (std::getline(ss, token, ' ')) {
    v.push_back(token);
    if (i > 0) {
      if (!req.compare("profileCfg")) {
        switch (i) {
          case 2:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/startFreq", std::stof(token)); break;
          case 3:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/idleTime", std::stof(token)); break;
          case 4:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/adcStartTime", std::stof(token)); break;
          case 5:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/rampEndTime", std::stof(token)); break;
          case 8:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/freqSlopeConst", std::stof(token)); break;
          case 10:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/numAdcSamples", std::stoi(token)); break;
          case 11:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/digOutSampleRate", std::stof(token)); break;
          case 14:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/rxGain", std::stof(token)); break;
        }
      } else if (!req.compare("frameCfg")) {
        switch (i) {
          case 1:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/chirpStartIdx", std::stoi(token)); break;
          case 2:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/chirpEndIdx", std::stoi(token)); break;
          case 3:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/numLoops", std::stoi(token)); break;
          case 4:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/numFrames", std::stoi(token)); break;
          case 5:
            nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/framePeriodicity", std::stof(token)); break;
        }
      }
    } else req = token;
    i++;
  }
}

void parameter_parser::cal_params(ros::NodeHandle &nh) {

  std::string myNodeID = ros::this_node::getName();
  float c0 = 299792458;
  int chirpStartIdx;
  int chirpEndIdx;
  int numLoops;
  float framePeriodicity;
  float startFreq;
  float idleTime;
  float adcStartTime;
  float rampEndTime;
  float digOutSampleRate;
  float freqSlopeConst;
  float numAdcSamples;

  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/startFreq", startFreq);
  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/idleTime", idleTime);
  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/adcStartTime", adcStartTime);
  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/rampEndTime", rampEndTime);
  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/digOutSampleRate", digOutSampleRate);
  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/freqSlopeConst", freqSlopeConst);
  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/numAdcSamples", numAdcSamples);

  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/chirpStartIdx", chirpStartIdx);
  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/chirpEndIdx", chirpEndIdx);
  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/numLoops", numLoops);
  nh.getParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/framePeriodicity", framePeriodicity);

  int ntx = chirpEndIdx - chirpStartIdx + 1;
  int nd = numLoops;
  int nr = numAdcSamples;
  float tfr = framePeriodicity * 1e-3;
  float fs = digOutSampleRate * 1e3;
  float kf = freqSlopeConst * 1e12;
  float adc_duration = nr / fs;
  float BW = adc_duration * kf;
  float PRI = (idleTime + rampEndTime) * 1e-6;
  float fc = startFreq * 1e9 + kf * (adcStartTime * 1e-6 + adc_duration / 2); 
  float fc_chirp = startFreq * 1e9 + BW / 2; 

  float vrange = c0 / (2 * BW);
  float max_range = nr * vrange;
  float max_vel = c0 / (2 * fc * PRI) / ntx;
  float vvel = max_vel / nd;

  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/num_TX", ntx);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/f_s", fs);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/f_c", fc);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/fc_chirp", fc_chirp);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/BW", BW);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/PRI", PRI);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/t_fr", tfr);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/max_range", max_range);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/range_resolution", vrange);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/max_doppler_vel", max_vel);
  nh.setParam("/multiradar_mmpose/radar_" + myNodeID.substr(myNodeID.length()-1) + "/doppler_vel_resolution", vvel);
}

}