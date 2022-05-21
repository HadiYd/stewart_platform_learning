#include <cmath>

#include "eigen3/Eigen/Core"

#include "geometry_msgs/Twist.h"
#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"

class IK
{
public: 
    IK(int argc, char **argv)
    {
        height = 2.0625;
        // Attachment points position in the base platform ( we get them from the SDF file)
        b << 1.5588457268119897, 0.8999999999999999, 0.25, 1,
              1.1021821192326179e-16, 1.8, 0.25, 1,
              -1.5588457268119897, 0.8999999999999999, 0.25, 1,
              -1.5588457268119895, -0.9000000000000002, 0.25, 1,
             -3.3065463576978537e-16, -1.8, 0.25, 1,
             1.558845726811989, -0.9000000000000008, 0.25, 1;

            
        // Attachment points position in the moving platform  (we get them from the SDF file)
        p << 0.9545941546018393, 0.9545941546018392, -0.05, 1,
              0.34940571088840333, 1.303999865490242, -0.05, 1,
              -1.303999865490242, 0.3494057108884034, -0.05, 1,
              -1.3039998654902425, -0.3494057108884025, -0.05, 1,
             0.34940571088840355, -1.303999865490242, -0.05, 1,
             0.9545941546018399, -0.9545941546018385, -0.05, 1;


        for (int i = 0; i < 6; i++)
        {
            f32ma_msg.data.push_back(0);
        }

        ros::init(argc, argv, "ik");
        ros::NodeHandle nh;
        pub = nh.advertise<std_msgs::Float32MultiArray>("/stewart/legs_position_cmd", 100);
        sub = nh.subscribe("stewart/platform_pose", 100, &IK::callback, this);
    }

    void run()
    {
        ros::spin();
    }

private: 
    void callback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        float x = msg->linear.x ;
        float y = msg->linear.y;
        float z = msg->linear.z;
        float roll = msg->angular.x;
        float pitch = msg->angular.y;
        float yaw = msg->angular.z;

        Eigen::Matrix<float, 4, 4> T = transformation_matrix(x, y, z + height, roll, pitch, yaw);

        for (size_t i = 0; i < 6; i++)
        {
            Eigen::Matrix<float, 4, 1> length = T*p.row(i).transpose() - b.row(i).transpose();
            f32ma_msg.data[i] = sqrt(pow(length(0), 2) + pow(length(1), 2) + pow(length(2), 2)) - height;
        }
        pub.publish(f32ma_msg);
    }

    Eigen::Matrix<float, 4, 4> transformation_matrix(float x , float y, float z, float r, float p, float yaw)
    {
        Eigen::Matrix<float, 4, 4> T;
        T << cos(yaw)*cos(p), -sin(yaw)*cos(r) + cos(yaw)*sin(p)*sin(r),  sin(yaw)*sin(r)+cos(yaw)*sin(p)*cos(r), x,
             sin(yaw)*cos(p),  cos(yaw)*cos(r) + sin(yaw)*sin(p)*sin(r), -cos(yaw)*sin(r)+sin(yaw)*sin(p)*cos(r), y,
                     -sin(p),                             cos(p)*sin(r),                           cos(p)*cos(r), z,
                           0,                                         0,                                       0, 1;
        return T;
    }
// changed cos(yaw) to cos(r) in 3x3 location!
    float height;

    Eigen::Matrix<float, 6, 4> b, p;

    ros::Publisher pub;
    ros::Subscriber sub;
    std_msgs::Float32MultiArray f32ma_msg;
};

int main(int argc, char **argv)
{
    IK ik(argc, argv);
    ik.run();

    return 0;
}