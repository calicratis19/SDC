# PID controller
The main goal of the project is to implement PID controller to steer the car around a track successfully.

## About PID controller

A PID consists of three types of controllers,

P - proportional component which directly reduces cross track error. Only using P controller is not enough for effective car driving because the car will oscillate and will drive off track very quickly. It produces a control output which influences the steering angle by the form of -Kp*cte where Kp is a tunable hyper parameter.

D - derivative controller reduces the oscillation that we see in P component. It produces a control output which influences the steering angle by the form of -Kd* d/dt(cte) where Kd is another tunable hyper parameter. Using PD component only doesn't reduce the oscillation problem completely. It still

I - integral controller is use to eliminate system bias problem in PD controller. It sums up the CTE overtime and contributes to the steering by the form of -Ki * sum(cte) where Ki is another tunable hyper parameter.

## How it was tuned

We used to following values for hypter parameterrs,
```
Kp = 0.2
Ki = 0.0001
Kd = 5
```
We have manually tuned the parameters.

We have tested the solution using P, PD and PID controller and captured the output.

1. With only P component the car drives off the track after few seconds even with very low throttle(.3). The output video can be found [here](https://youtu.be/hZlh5kZDBdQ).

2. With PD component the car can drive the track with continuous oscillation. The video output is [here](https://youtu.be/uQBJ4Rx7Ous).

3. With PID component the car oscillates less then before. But to me it seems the output is very close to the PD controller. Though the video is captured with throttle value 5 for both PD and PID, the PID controller could successfully drive for throttle value 0.6 which PD could not. The output is [here](https://youtu.be/A8yA0yt0ETw).

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets) == 0.13, but the master branch will probably work just fine
  * Follow the instructions in the [uWebSockets README](https://github.com/uWebSockets/uWebSockets/blob/master/README.md) to get setup for your platform. You can download the zip of the appropriate version from the [releases page](https://github.com/uWebSockets/uWebSockets/releases). Here's a link to the [v0.13 zip](https://github.com/uWebSockets/uWebSockets/archive/v0.13.0.zip).
  * If you run OSX and have homebrew installed you can just run the ./install-mac.sh script to install this
* Simulator. You can download these from the [project intro page](https://github.com/udacity/CarND-PID-Control-Project/releases) in the classroom.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`.

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.
