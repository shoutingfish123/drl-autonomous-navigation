#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan

def callback(message):
    rospy.loginfo(f"I see {len(message.ranges)} laser points!")

def spy():
    rospy.init_node('spy', anonymous=True)
    rospy.Subscriber("/scan",LaserScan, callback)
    rospy.spin()

if __name__=="__main__":
    try:
        spy()
    except rospy.ROSInterruptException:
        print("node terminated")




'''import rospy
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

print(f"Keras backend: {keras.backend.backend()}")

import keras
import numpy as np

# Create test data using NumPy
tensor_a_np = np.array([1.0, 2.0, 3.0], dtype="float32")
tensor_b_np = np.array([4.0, 5.0, 6.0], dtype="float32")

# Perform an operation using keras.ops
tensor_c = keras.ops.add(tensor_a_np, tensor_b_np)

# Print the result and its type
print(f"Result tensor: {tensor_c}")
print(f"Result type: {type(tensor_c)}")
'''