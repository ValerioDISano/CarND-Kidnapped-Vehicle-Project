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

  num_particles = 30;  // TODO: Set the number of particles

  this->particles = std::move(std::vector<Particle>(num_particles));

  //auto add_noise = [&std](auto x, const auto& idx){return x + GaussianNoise<decltype(x)>(x, std[idx]).getSample(); };
  this->weights = std::move(std::vector<double>(this->num_particles));

  int counter = 0;

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0.0, std[0]);
	std::normal_distribution<double> dist_y(0.0, std[1]);
	std::normal_distribution<double> dist_theta(0.0, std[2]);
  for (auto& particle : this->particles)
  {

    particle.id = counter;
    //particle.weight = 1.0;
    this->weights[counter] = 1.0;
    //particle.x = add_noise(x, 0);
    //particle.y = add_noise(y, 1);
    //particle.theta = add_noise(theta, 2);

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
    if (fabs(yaw_rate) < 0.00001)
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
    //auto add_noise = [&std_pos](const auto& x, const auto& idx){return x + GaussianNoise<double>(x, std_pos[idx]).getSample(); };
    //particle.x = add_noise(particle.x, 0);
    //particle.y = add_noise(particle.y, 1);
    //particle.theta = add_noise(particle.theta, 2);
    //particle.x += dist_x(gen);
    //particle.y += dist_y(gen);
    //particle.theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(const Map& map,
                                     LandmarkObs& observation,
                                     const double& sensor_range) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  double current_dist (0.0);
  double min_dist (std::numeric_limits<double>::max());

  observation.id = -1;

  for (auto &landmark : map.landmark_list)
  {
    current_dist = dist(landmark.x_f, landmark.y_f, observation.x, observation.y);
    if (current_dist > sensor_range) continue;
    if (current_dist < min_dist)
    {
      min_dist = current_dist;
      observation.id = landmark.id_i - 1;
    }
  }
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
  Map::single_landmark_s associated_landmark;
  vector<LandmarkObs> observations_copy = observations;

  double weights_norm = 0.0;

  for (auto& particle : this->particles)
  {
    bool update = false;
    particle.weight = 1.0; // reset particle weight

    for (auto& observation : observations_copy)
    {
      homogenousTransformation(particle.x, particle.y, particle.theta, observation.x, observation.y);
      this->dataAssociation(map_landmarks, observation, sensor_range);
      if (observation.id < 0)
      {
        continue;
      }
      associated_landmark = map_landmarks.landmark_list.at(observation.id);
      auto current_weight = multivariate_gaussian_2D<double>(observation.x,
                                                              observation.y,
                                                              associated_landmark.x_f,
                                                              associated_landmark.y_f,
                                                              std_landmark[0],
                                                              std_landmark[1]
                                                            );
      if (current_weight > 0.0)
      {
        particle.weight *= current_weight;
        update = true;
      }
    }

    if (!update)
    {
      particle.weight = 0.0;
    }
    weights_norm += particle.weight;
  }
  int counter = 0;
  for (auto& particle : this->particles)
  {
    if (weights_norm > 0.0) particle.weight /= weights_norm;
    this->weights[counter] = particle.weight;
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

  std::vector<Particle> new_particles;
  new_particles.resize(this->particles.size());

  std::default_random_engine generator;
  //std::random_device rd;
  //std::mt19937 generator(rd());
  std::discrete_distribution<int> sampling_distribution (this->weights.begin(), this->weights.end());

  for (auto i = 0; i < this->particles.size(); i++)
  {
    auto number = sampling_distribution(generator);
    new_particles[i] = this->particles[number];
  }

  // remove duplicates
  //std::sort(new_particles.begin(), new_particles.end());
  //auto last = std::unique( new_particles.begin(), new_particles.end());
  //new_particles.erase(last, new_particles.end());

  this->particles = new_particles;
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
