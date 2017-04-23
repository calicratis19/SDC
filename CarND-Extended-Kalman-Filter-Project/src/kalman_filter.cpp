#include "kalman_filter.h"
#include <iostream>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::max;

#define PI 3.1415926535897

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z, MeasurementPackage::SensorType sensorType)
{
	VectorXd y;
	if(sensorType == MeasurementPackage::RADAR)
	{
		VectorXd hx = VectorXd(3);
		double px = x_[0];
		double py = x_[1];
		double vx = x_[2];
		double vy = x_[3];
		hx[0] = sqrt(px*px + py*py);
		hx[1] = atan2(py,px);
		hx[2] = (px*vx + py*vy)/max(.0001,hx[0]);
		y = z - hx;

		while(y[1] > PI || y[1] < -PI)
		{
			if(y[1] > PI)
				y[1]-=PI;
			else y[1]+=PI;
		}
	}
	else
	{
		y = z - H_ * x_;
	}

	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

