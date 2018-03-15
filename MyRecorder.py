#coding=utf-8
#!/usr/bin/env python

import rospy
import baxter_interface

class MyRecorder(object):
    def __init__(self, filename, rate):
        self._filename = filename
        self._raw_rate = rate
        self._rate = rospy.Rate(rate)
        self._start_time = rospy.get_time()
        self._done = False

        self._limb_left = baxter_interface.Limb("left")
        self._limb_right = baxter_interface.Limb("right")

    def _time_stamp(self):
        return rospy.get_time() - self._start_time

    def stop(self):
        """
        Stop recording.
        """
        self._done = True

    def done(self):
        """
        Return whether or not recording is done.
        """
        if rospy.is_shutdown():
            self.stop()
        return self._done

    def recorder(self):
        if self._filename:
            pass
