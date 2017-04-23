#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;
  H_laser_ << 1,0,0,0,
              0,1,0,0;
  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  //create a 4D state vector, we don't know yet the values of the x state
  	ekf_.x_ = VectorXd(4);

  	//state covariance matrix P
  	ekf_.P_ = MatrixXd(4, 4);
  	ekf_.P_ << 1, 0, 0, 0,
  			  0, 1, 0, 0,
  			  0, 0, 1000, 0,
  			  0, 0, 0, 1000;

  	//measurement matrix
  	ekf_.H_ = H_laser_;

  	//the initial transition matrix F_
  	ekf_.F_ = MatrixXd(4, 4);
  	ekf_.F_ << 1, 0, 1, 0,
  			  0, 1, 0, 1,
  			  0, 0, 1, 0,
  			  0, 0, 0, 1;

  	//set the acceleration noise components
  	noise_ax = 12;
  	noise_ay = 12;


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
	if (!is_initialized_)
	{
		cout << "Kalman Filter Initialization " << endl;

		if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
		{
			//set the state with the initial location and zero velocity
			ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
		}
		else
			ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1],
			measurement_pack.raw_measurements_[2], 0;


		previous_timestamp_ = measurement_pack.timestamp_;
		is_initialized_ = true;
		return;
	}

	//compute the time elapsed between the current and previous measurements
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = measurement_pack.timestamp_;

	float dt_2 = dt * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;

	//Modify the F matrix so that the time is integrated
	ekf_.F_(0, 0) = 1;
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 1) = 1;
	ekf_.F_(1, 3) = dt;
	ekf_.F_(2, 2) = 1;
	ekf_.F_(3, 3) = 1;

	//set the process covariance matrix Q
	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
			   0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
			   dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
			   0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

	//predict
	ekf_.Predict();

	if(measurement_pack.sensor_type_ == MeasurementPackage::LASER)
	{
		ekf_.R_ = R_laser_;
		ekf_.H_ = H_laser_;
	}
	else
	{
		ekf_.R_ = R_radar_;
		ekf_.H_ = Tools::CalculateJacobian(ekf_.x_);
	}
	//measurement update
	ekf_.Update(measurement_pack.raw_measurements_,measurement_pack.sensor_type_);

	std::cout << "x_= " << ekf_.x_ << std::endl;
	std::cout << "P_= " << ekf_.P_ << std::endl;

}
