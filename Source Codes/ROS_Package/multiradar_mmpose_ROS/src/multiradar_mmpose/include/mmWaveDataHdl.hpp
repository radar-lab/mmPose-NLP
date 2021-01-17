/*
 * mmWaveDataHdl.hpp
 *
 * This file defines a ROS nodelet which will open up a serial port provided by the user
 * at a certain baud rate (also provided by user) that will interface with the 1443EVM mmwDemo
 * Data UART to be used for board configuration.
 *
 *                       
 * Copyright (C) 2017 Texas Instruments Incorporated - http://www.ti.com/ 
 * 
 * 
 *  Redistribution and use in source and binary forms, with or without 
 *  modification, are permitted provided that the following conditions 
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the 
 *    documentation and/or other materials provided with the   
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
*/
#ifndef MMWAVE_DATA_HDL_H
#define MMWAVE_DATA_HDL_H

/*Include ROS specific headers*/
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "serial/serial.h"  
#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>

/*Include standard C/C++ headers*/
#include <iostream>
#include <cstdio>
#include <sstream>

/*mmWave Driver Headers*/
#include "DataHandlerClass.h"

namespace multiradar_mmpose
{

class mmWaveDataHdl : public nodelet::Nodelet
{  
   public:
   
   mmWaveDataHdl();
   
   private:
   
   virtual void onInit();
   
   //char* mySerialPort;
   
   //int myBaudRate;
   
}; //Class mmWaveDataHdl 

} //namespace multiradar_mmpose 

#endif
