/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  if (this->is_initialized)
  {
    return;
  }

  num_particles = 100;  // TODO: Set the number of particles

  this->particles = std::move(std::vector<Particle>(num_particles));
  this->weights = std::move(std::vector<double>(this->num_particles));

  int counter = 0;

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0.0, std[0]);
	std::normal_distribution<double> dist_y(0.0, std[1]);
	std::normal_distribution<double> dist_theta(0.0, std[2]);
  for (auto& particle : this->particles)
  {

    particle.id = counter;
    this->weights[counter] = 1.0;

    particle.x = x + dist_x(gen);
    particle.y = y + dist_y(gen);
    particle.theta = theta + dist_theta(gen);

    counter++;
  }

  this->is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
	std::normal_distribution<double> dist_y(0.0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

  auto delta_theta = yaw_rate * delta_t;
  auto velocity_ratio = velocity / yaw_rate;

  for (auto &particle : this->particles)
  {
    if (fabs(yaw_rate) < 0.001)
    {
      particle.x += (velocity * cos(particle.theta) * delta_t);
	    particle.y += (velocity * sin(particle.theta) * delta_t);
    } else
    {
      particle.x += (velocity_ratio *
	      	(sin(particle.theta + delta_theta) - sin(particle.theta)));
      particle.y += (velocity_ratio *
	      	(cos(particle.theta) - cos(particle.theta + delta_theta)));
    }
    particle.theta += delta_theta;
    // Add noise
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
  }

}

int ParticleFilter::dataAssociation(LandmarkObs& observation,
                                    const Map& map_landmarks,
                                    const double& sensor_range)
{
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  int lm_index = -1;
  auto min_dist = std::sqrt(sensor_range * sensor_range);

  for (const auto& landmark : map_landmarks.landmark_list)
  {
    auto landmark_x = landmark.x_f;
    auto landmark_y = landmark.y_f;

    double distance = dist(observation.x, observation.y, landmark_x, landmark_y);
    if (distance < min_dist)
    {
      min_dist = distance;
      lm_index = landmark.id_i - 1;
    }
  }
  observation.id = lm_index;
  return lm_index;
}

void ParticleFilter::updateWeights(const double& sensor_range,const double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  unsigned int counter = 0;
  for (auto &particle : this->particles)
  {
        auto associations = std::make_unique<std::vector<int>>();
        auto sense_x = std::make_unique<std::vector<double>>();
        auto sense_y = std::make_unique<std::vector<double>>();

        auto t_observations = observations;

        particle.weight = 1.0; // reset weight

        for (auto& observation : t_observations)
        {
            homogenousTransformation<double>(particle.x, particle.y, particle.theta, observation.x, observation.y);
            auto min_index = dataAssociation(observation, map_landmarks, sensor_range);

            auto x = observation.x;
            auto y = observation.y;

            auto mean_x = double (map_landmarks.landmark_list[min_index].x_f);
            auto mean_y = double (map_landmarks.landmark_list[min_index].y_f);

            auto std_x = std_landmark[0];
            auto std_y = std_landmark[1];

            auto multiplier = multivariate_gaussian_2D<double>(x, y, mean_x, mean_y, std_x, std_y);

            if (multiplier > 0.0)
                particle.weight *= multiplier;

            associations->push_back(min_index+1);
            sense_x->push_back(x);
            sense_y->push_back(y);

        }

        this->weights[counter] = particle.weight;
        SetAssociations(particle, *std::move(associations), *std::move(sense_x), *std::move(sense_y));
        counter++;
    }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  auto new_particles = std::make_unique<decltype(this->particles)>(this->particles.size());

  std::default_random_engine generator;
  std::discrete_distribution<int> sampling_distribution (this->weights.begin(), this->weights.end());

  for (auto i = 0; i < this->particles.size(); i++)
  {
    auto number = sampling_distribution(generator);
    (*new_particles)[i] = this->particles[number];
  }

  this->particles = std::move(*new_particles);
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
