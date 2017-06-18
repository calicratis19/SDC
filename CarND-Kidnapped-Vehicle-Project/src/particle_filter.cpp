/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

#define PARTICLE_NUMBER 100

using namespace std;

normal_distribution<double> zeroOneDistribution(0, 1);
default_random_engine generator;

inline double bivariate_normal(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
	return exp(-((x - mu_x)*(x - mu_x) / (2 * sig_x*sig_x) + (y - mu_y)*(y - mu_y) / (2 * sig_y*sig_y))) / (2.0*3.14159*sig_x*sig_y);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	
	for (int i = 0;i < PARTICLE_NUMBER;i++)
	{
		Particle _particle;
		_particle.x = x + std[0] * zeroOneDistribution(generator);
		_particle.y = y + std[1] * zeroOneDistribution(generator);
		_particle.theta = theta + std[2] * zeroOneDistribution(generator);
		_particle.weight = 1;
		particles.push_back(_particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


	for (int i = 0;i < particles.size(); i++)
	{
		Particle &_particle = particles[i];

		double _theta = _particle.theta;
		double _angle = _theta + yaw_rate*delta_t;
		if (yaw_rate > .00001)
		{
			_particle.x += velocity / yaw_rate * (sin(_angle) - sin(_theta)) + std_pos[0] * zeroOneDistribution(generator);
			_particle.y += velocity / yaw_rate * (cos(_theta) - cos(_angle)) + std_pos[1] * zeroOneDistribution(generator);;			
		}	
		else
		{
			_particle.x += velocity * cos(_theta)* delta_t + std_pos[0] * zeroOneDistribution(generator);
			_particle.y += velocity * sin(_theta) * delta_t + std_pos[1] * zeroOneDistribution(generator);;			
		}
		_particle.theta = _angle + std_pos[2] * zeroOneDistribution(generator);		
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.



}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm

	vector<LandmarkObs> _groundTruthObs;
	LandmarkObs _obs;

	for (Map::single_landmark_s &_landMark : map_landmarks.landmark_list)
	{		
		_obs.x = _landMark.x_f;
		_obs.y = _landMark.y_f;
		_obs.id = _landMark.id_i;

		_groundTruthObs.push_back(_obs);
	}

	double _weightSum = 0;
	for (Particle &_particle: particles)
	{
		//convert observations from vehicle to map coordinate with respect to particles

		vector<LandmarkObs> vehicleToMapObs;

		double _theta = -_particle.theta;
		double _weight = 1;
		for (LandmarkObs &_landMarkObs : observations)
		{
			if (dist(_landMarkObs.x, _landMarkObs.y, 0, 0) > sensor_range)
				continue;

			LandmarkObs _mapCoordinate;
			_mapCoordinate.x = _landMarkObs.x * cos(_theta) + _landMarkObs.y * sin(_theta) + _particle.x;
			_mapCoordinate.y = _landMarkObs.y * cos(_theta) - _landMarkObs.x * sin(_theta) + _particle.y;

			vehicleToMapObs.push_back(_mapCoordinate);

			int _indx = 0;
			int _minDist;

			for (int i = 0; i < _groundTruthObs.size(); i++)
			{
				double _distance = dist(_groundTruthObs[i].x, _groundTruthObs[i].y, _mapCoordinate.x, _mapCoordinate.y);

				if (!i)
					_minDist = _distance;
				else if( _minDist > _distance)
				{
					_minDist = _distance;
					_indx = i;
				}
			}
			_weight *= bivariate_normal(_mapCoordinate.x, 
										_mapCoordinate.y,
									    _groundTruthObs[_indx].x,
										_groundTruthObs[_indx].y,
									    std_landmark[0],
										std_landmark[1]);

		}
		_particle.weight = _weight;
		_weightSum += _weight;
	}

	for (Particle &_particle : particles)
	{
		_particle.weight /= _weightSum;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine _generator;
	vector<double>weights;
	for (Particle _particle : particles)
		weights.push_back(_particle.weight);

	discrete_distribution<> _weightDistribution(weights.begin(), weights.end());

	vector<Particle> _newParticle;

	for (int i = 0;i < weights.size(); i++)
	{
		_newParticle.push_back(particles[_weightDistribution(_generator)]);
	}
	particles = _newParticle;
	for (Particle &_particle : particles)
	{
		_particle.weight = 1.0;
	}
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
