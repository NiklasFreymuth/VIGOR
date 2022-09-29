import numpy as np
import pinocchio
import os


if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RobotModelFromPinochio:
    def __init__(self, obj_urdf="environments/util/panda_pinocchio.urdf"):
        self.pin_model = pinocchio.buildModelFromUrdf(obj_urdf)
        self.pin_data = self.pin_model.createData()

        self.pin_end_effector_frame_id = self.pin_model.getFrameId("panda_grasptarget")
        self.pin_rod_frame_id = self.pin_model.getFrameId("panda_rod")

        self.pin_q = np.zeros(self.pin_model.nv)
        self.pin_qd = np.zeros(self.pin_model.nv)

    def get_forward_kinematics(self, q):
        # account for additional joints (e.g. finger)
        self.pin_q[:7] = q
        pinocchio.framesForwardKinematics(self.pin_model, self.pin_data, self.pin_q)

        current_c_pos = np.array(
            self.pin_data.oMf[self.pin_end_effector_frame_id].translation
        )

        quat_pin = pinocchio.Quaternion(
            self.pin_data.oMf[self.pin_end_effector_frame_id].rotation
        ).coeffs()  # [ x, y, z, w]
        current_c_quat = np.zeros(4)
        current_c_quat[1:] = quat_pin[:3]
        current_c_quat[0] = quat_pin[-1]

        return current_c_pos, current_c_quat

    def get_forward_kinematic_position(self, joint_position: np.array, rod=False) -> np.array:
        # account for additional joints (e.g. finger)
        shaped_joint_positions = np.zeros(7)
        shaped_joint_positions[:len(joint_position)] = joint_position
        self.pin_q[:7] = shaped_joint_positions
        pinocchio.framesForwardKinematics(self.pin_model, self.pin_data, self.pin_q)

        if rod:
            current_c_pos = np.array(self.pin_data.oMf[self.pin_rod_frame_id].translation)
        else:
            current_c_pos = np.array(self.pin_data.oMf[self.pin_end_effector_frame_id].translation)
        return current_c_pos
