#ifndef _SDF_JOINT_CONTROLLER_HPP_
#define _SDF_JOINT_CONTROLLER_HPP_

#include <memory>
#include <thread>
#include <vector>
#include <iostream>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "ros/callback_queue.h"
#include "std_msgs/Float32MultiArray.h"
#include "geometry_msgs/Vector3.h"

namespace gazebo
{
    class SDFJointController : public ModelPlugin
    {
    public:
        SDFJointController() {}

        virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
        {
            if (_model->GetJointCount() <= 0)
            {
                std::cerr << "Invalid joint count, ROS_SDF plugin not loaded\n";
                return;
            }
            else
                std::cout << "\nCustom joint controller plugin is attached to model ---> " << _model->GetName() << " \n";

            m_model = _model;
            m_joints = m_model->GetJoints();
          
        
            // initial PID values
            n.getParam("gazebo_controller_init/propor",propor);
            n.getParam("gazebo_controller_init/integr",integr);
            n.getParam("gazebo_controller_init/deriv",deriv);
            init_pid = common::PID(propor, integr, deriv); 

            // Set initial PID values
            auto joints_it_1 = std::begin(m_joints);
            for (int i = 0; i < 6; i++)
                m_model->GetJointController()->SetPositionPID((*joints_it_1++)->GetScopedName(), init_pid);

            std::cout << "\n The initial PID values of all joints: P: " << propor << ", I: "<<integr << ", and D: "<< deriv <<std::endl;

            // set  6 legs' initial lengths
            n.getParam("gazebo_controller_init/init_legs_length",init_legs_length);
     
            auto joints_it = std::begin(m_joints);
            for (int i = 0; i < 6; i++)
                m_model->GetJointController()->SetPositionTarget((*joints_it++)->GetScopedName(), init_legs_length);
            

            // ros node initialization
            int argc = 0;
            ros::init(argc, nullptr, "gazebo_client", ros::init_options::NoSigintHandler);
            m_nh.reset(new ros::NodeHandle("gazebo_client"));

            // Subscribe to /model_name/legs_position_cmd topic (you publish to this to set legs' positions)
            ros::SubscribeOptions so = ros::SubscribeOptions::create<std_msgs::Float32MultiArray>(
                "/" + m_model->GetName() + "/legs_position_cmd", 
                100, 
                boost::bind(&SDFJointController::setPosition, this, _1),
                ros::VoidPtr(),
                &m_ros_queue);
            m_ros_sub = m_nh->subscribe(so);

            // Subscribe to /model_name/pid_cmd topic (you publish to this to set the new pid valuse)
            ros::SubscribeOptions ops = ros::SubscribeOptions::create<geometry_msgs::Vector3>(
                "/"+  m_model->GetName() + "/pid_cmd",
                100,
                boost::bind(&SDFJointController::SetPID, this, _1),
                ros::VoidPtr(),
                &m_ros_queue);
            m_ros_pid_sub = m_nh->subscribe(ops);

            // Set up a handler so we don't block here
            m_ros_queue_thread = std::thread(std::bind(&SDFJointController::queueThread, this));
        }

        void setPosition(const std_msgs::Float32MultiArray::ConstPtr& msg)
        {
            auto joints_it = std::begin(m_joints);
            for (auto data_it = std::begin(msg->data); data_it != std::end(msg->data); ++data_it)
                m_model->GetJointController()->SetPositionTarget((*joints_it++)->GetScopedName(), *data_it);
        }

       void SetPID(const geometry_msgs::Vector3::ConstPtr& msg)
        {
        new_pid = common::PID(msg->x, msg->y, msg->z);
        
        // Set the joint's pid values.
        for (const auto& joint: m_joints)
                m_model->GetJointController()->SetPositionPID(joint->GetScopedName(), new_pid);

        }


    private:
        void queueThread()
        {
            static const double timeout = 0.01;
            while (m_nh->ok())
                m_ros_queue.callAvailable(ros::WallDuration(timeout));
        }

        physics::ModelPtr m_model;
        common::PID init_pid;
        common::PID new_pid;

        ros::CallbackQueue m_ros_queue;
        ros::Subscriber m_ros_sub;
        ros::Subscriber m_ros_pid_sub;
        ros::NodeHandle n;
 
        std::thread m_ros_queue_thread;
        std::unique_ptr<ros::NodeHandle> m_nh;
        std::vector<physics::JointPtr> m_joints;
        double propor;
        double integr;
        double deriv;
        double init_legs_length;

    };


    GZ_REGISTER_MODEL_PLUGIN(SDFJointController)
}



#endif // _SDF_JOINT_CONTROLLER_HPP_