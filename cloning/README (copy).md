# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## Project discription

Implement a PID controler.


The simulater has very sharp turns therefore the P element which defines  steering should be Proportional to the cross-track error (CTE).
Doing hard steeering will cause oscillations. The D element will fix this by steering in porportion to the CTE dirivative.
The Integral element is for fixing the bias that accumulates over time.

At first I reckoned that the highest value should be assigned to P. After seeing the extreme osccilations, I gradually lowered the P and highened the the D elements. The I element I ignored comletely as I could not see any bias. The car followed the track for more than a complete track round. 

After getting feddback on the project that the car had lefet the track I did some more manual parameter tuning and added a very small Integral magnitude.

| Element | Magnitude|
| ------ |:------:|
| P      |  -0.1 |
| I      |   -0.001 |
| D |  -0.8 |

To make sure the I got the numbers right, I let the simulator run a good few minutes.
