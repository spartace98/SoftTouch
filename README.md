# SoftTouch: A Sensor-Placement Framework for Soft Robotic Hands
Repository containing contributed data and models described in Conference Paper Submission

## Abstract
Sensor placement for grasping tasks in conventional robotic hands has been studied, with goals including sensorizing essential contact areas or determining the effect of number of sensors on performance. However, with the new generation of dexterous soft robotic hands that deform to the shape of the object, the frameworks used to study these problems may not be sufficient.  In particular, we find that real-world experiments are essential for determining the value of different sensors and the effect of different sensor placements due to the complex interactions between the deformable robot body, sensor material properties, and sensor and task performance.  In this paper, we propose a sensor-placement framework for dexterous soft robotic hands that is easily reconfigurable to different hand designs using off-the-shelf sensors. Our three-step framework selects and evaluates candidate sensor configurations to determine the effectiveness of sensors in each configuration for estimating qualitative and quantitative manipulation metrics. We tested our framework on a soft robotic hand to select the optimum sensor placement for a given set of manipulation patterns using force and inertial sensors, which we discuss further in our submitted conference paper. 

## Folder Hierarchy
- data: Sensor readings + AprilTag readings for all the configurations
  - config1 (A1)
  - config2 (B1)
  - config3 (A2)
  - config4 (B2)
- Prediction_Models: Classifier and Regression Models to process sensor readings
  - Classifier
    - "classifier.py"
  - Regression
    - "regression.py"

## Raw Regression Results
<img width="592" alt="image" src="https://user-images.githubusercontent.com/36890067/197688155-5e2a4129-d3d9-4b18-93d4-09ce1a1de3fc.png">
More details on the calculations will be updated before the conference

## Notes
1. During our experiments, we ran two manipulations for a single set. Post processing was done to separate the two manipulations. Files in data are post-processed. 
2. Video showing the manipulation runs will be uploaded soon.  

## Acknowledgement
- Professor Nancy: for her mentorship for this project
- Carnegie Mellon University: for providing undergraduate funding (SURF, SURG) for this project

For additional information, please contact Charlie at chaoli.charlie@gmail.com
