from typing import Optional, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from Circle import Circle


class InteractivePlanarRobot:
    def __init__(self, context: np.ndarray, dynamic: str = "acceleration", delta_time: float = 0.2,
                 include_rotation: bool = True, num_links: int = 5, initial_target: Optional[List] = None,
                 current_rollout: int = 1, current_context: int = 0, common_suplot: Optional = None):
        self._num_links = num_links
        self._include_rotation = include_rotation
        self._delta_time = delta_time
        target_points = context.reshape(-1, 2)
        self._target_circles = [Circle(0.5, *target_point) for target_point in target_points]

        if self._include_rotation:  # x,y,rotation angle as state
            # current and previous end-effector (cursor) position in x,y and rotation
            if initial_target is not None:
                assert len(
                    initial_target) == 3, f"Need to provide x,y, rotation for initial target. Got {initial_target}"
                target = np.array(initial_target)
            else:
                target = np.array([self._num_links, 0, 0])

            self.rotation = 0
        else:  # x,y as state
            if initial_target is not None:
                assert len(initial_target) == 2, f"Need to provide x,y for initial target. Got {initial_target}"
                target = np.array(initial_target)
            else:
                target = np.array([self._num_links, 0])
        self.target_end_effector_state = target
        self.previous_end_effector_state = target

        # current robot angles
        self.robot_joint_angles = np.zeros(self._num_links)  # + 0.1

        self._current_step = 0
        self._current_target = 0

        # set dynamics and their derivates
        self.dynamic = dynamic
        self.velocity = np.zeros(3 if self._include_rotation else 2)  # for acceleration as dynamic
        self.acceleration = np.zeros(3 if self._include_rotation else 2)  # for jerk as dynamic

        self._previous_joint_velocity = np.zeros(num_links)

        # set colors for cursor
        self._num_colors = 100
        self._colors = [plt.cm.hsv((x + 0.5) / self._num_colors) for x in np.arange(0, self._num_colors)]

        # reset and format the plot
        self.fig, self.ax = common_suplot if common_suplot is not None else plt.subplots()
        self.ax.clear()
        # plt.clf()

        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        axes = plt.gca()
        axes.set_aspect(aspect="equal")
        axes.set_xlim([-self._num_links * 1.3, self._num_links * 1.3])
        axes.set_ylim([-self._num_links * 1.3, self._num_links * 1.3])
        plt.gca().add_patch(Rectangle(xy=(-0.1, -0.1), width=0.2, height=0.2, facecolor="grey", alpha=1, zorder=0))
        self.circle_plots = [circle.plot(position=position, fill=position == 0, fillcolor="b")
                             for position, circle in enumerate(self._target_circles)]
        border = Circle(radius=self._num_links, x_coordinate=0, y_coordinate=0)
        border.plot(fill=False)
        plt.scatter([], [], marker=f"$A$", color="g", label="Save and continue")
        plt.scatter([], [], marker=f"$B$", color="r", label="Re-do context")
        plt.scatter([], [], marker=f"$X$", color="b", label="Go back")
        plt.scatter([], [], marker=f"$Y$", color="y", label="Quit")
        plt.scatter([], [], marker=f"$L$", color="k", label="Move")
        plt.scatter([], [], marker=f"$R$", color="k", label="Rotate")
        plt.legend(loc="upper left", ncol=3, fontsize=8)
        plt.title(f"Gathering rollout #{current_rollout} for context #{current_context}")

        # keep an index of the previous robot to potentially remove it from the plot
        self.previous_robot, = self.ax.plot([], "o-", alpha=0.5)

        self.fig.canvas.draw()  # note that the first draw comes before setting data
        self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # plt.show(block=False)

        # record a history
        self.history = [np.zeros(self._num_links)]

    def forward_kinematic(self, robot_joint_angles: np.array,
                          last_step_only: bool = False,
                          include_orientation: bool = False) -> np.array:
        absolute_angles = np.cumsum(robot_joint_angles, axis=-1)

        pos = np.zeros([self._num_links + 1, 2])
        for i in range(self._num_links):
            pos[i + 1, 0] = pos[i, 0] + np.cos(absolute_angles[i])
            pos[i + 1, 1] = pos[i, 1] + np.sin(absolute_angles[i])

        if include_orientation:
            orientation = np.concatenate(([0], absolute_angles), axis=0)[:, None]
            pos = np.concatenate((pos, orientation), axis=-1)

        if last_step_only:
            return pos[-1, :]
        else:
            return pos

    def jacobian(self, robot_joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the jacobian for the given joint angles. For a total of n angles of unit length, the i-th row is
        computed as
        [-\sum_{j=i}^n sin(cumulative_angle[j]),
         \sum_{j=i}^n cos(cumulative_angle[j])]

        Args:
            robot_joint_angles: The current joint angles as an array of shape (num_links)

        Returns: The jacobian Matrix of shape (2, num_links), where the entry ij corresponds to the influence of link j
          on the planar position for dimension i.

        """

        # start by computing the cumulative/absolute angles
        cumulative_angles = np.cumsum(robot_joint_angles, axis=-1)

        jacobian = np.ones((2, self._num_links))
        for link in range(0, self._num_links):
            current_column = np.zeros(2)
            for i in range(0, self._num_links - link):
                current_column[0] -= np.sin(cumulative_angles[-(1 + i)])
                current_column[1] += np.cos(cumulative_angles[-(1 + i)])
            jacobian[:, link] = current_column
        if self._include_rotation:
            jacobian = np.concatenate((jacobian, np.ones((1, self._num_links))))
        return jacobian

    def step(self, action: np.ndarray):
        if not self._include_rotation:
            action = action[:2]
        self.get_velocity(action=action, dynamic_type=self.dynamic)

        new_target = self.target_end_effector_state + self.velocity

        self.project_target(new_target)

        learning_rate = 0.01
        updates_per_step = 100
        stability_constant = 3.0e-1
        rest_posture_scale = 1
        update_bound_type = "acceleration"  # either "acceleration, "velocity" or None
        total_update_bound = 3.0e-2

        stability_matrix = stability_constant * np.eye(3 if self._include_rotation else 2)
        current_robot_end_effector_state = self.forward_kinematic(robot_joint_angles=self.robot_joint_angles,
                                                                  last_step_only=True,
                                                                  include_orientation=self._include_rotation)

        for _ in range(updates_per_step):
            current_dimensionwise_error = new_target - current_robot_end_effector_state
            if self._include_rotation:  # account for periodicity of the end-effector angle
                current_dimensionwise_error[2] = self.angle_normalize(current_dimensionwise_error[2])
            current_total_error = np.sum(current_dimensionwise_error ** 2)

            joint_update_velocity = self.compute_update_velocity(current_dimensionwise_error,
                                                                 rest_posture_scale,
                                                                 stability_matrix)

            joint_update_velocity = self.bound_update_step(joint_update_velocity=joint_update_velocity,
                                                           learning_rate=learning_rate,
                                                           total_update_bound=total_update_bound,
                                                           update_bound_type=update_bound_type)

            joint_update_velocity = self._delta_time * joint_update_velocity
            proposed_robot_joint_angles = self.robot_joint_angles + joint_update_velocity
            proposed_robot_joint_angles = self.angle_normalize(proposed_robot_joint_angles)

            # check if step was successful and apply update if it was
            resulting_robot_end_effector_state = self.forward_kinematic(
                robot_joint_angles=proposed_robot_joint_angles,
                last_step_only=True,
                include_orientation=self._include_rotation)
            resulting_dimensionwise_error = new_target - resulting_robot_end_effector_state
            if self._include_rotation:  # account for periodicity of the end-effector angle
                resulting_dimensionwise_error[2] = self.angle_normalize(resulting_dimensionwise_error[2])
            resulting_total_error = np.sum(resulting_dimensionwise_error ** 2)
            success = resulting_total_error <= current_total_error + 1.0e-6  # check if we get no worse than before
            if success:   # successful gradient step. Update robot and increase learning rate
                learning_rate = learning_rate * 1.2
                self.robot_joint_angles = proposed_robot_joint_angles
                current_robot_end_effector_state = resulting_robot_end_effector_state

                self._previous_joint_velocity = joint_update_velocity

            else:   # failed gradient step. Do not update robot and decrease learning rate
                learning_rate = learning_rate / 2

        if not np.allclose(self.history[-1], self.robot_joint_angles):
            # append current step to history if the robot moved
            self.history.append(self.robot_joint_angles)

        self.previous_end_effector_state = self.target_end_effector_state
        self.target_end_effector_state = new_target
        self._current_step = self._current_step + 1

        self.update_target_circles(current_robot_end_effector_state)

    def get_velocity(self, action: np.ndarray, dynamic_type: str):
        """
        Compute the velocity given the action. May factor in higher-order dynamics if the dynamic type is
        "acceleration" or "jerk".
        Args:
            action: The proposed action, i.e., the controller input
            dynamic_type: Either "velocity", "acceleration" or "jerk"

        Returns:

        """
        if dynamic_type == "velocity":
            self.velocity = np.where(np.abs(action) > 0.15, action, 0) * self._delta_time
        elif dynamic_type == "acceleration":
            acceleration = np.where(np.abs(action) > 0.15, action, 0) * self._delta_time ** 2
            self.velocity = self.velocity + acceleration
        elif dynamic_type == "jerk":
            jerk = np.where(np.abs(action) > 0.15, action, 0) * self._delta_time ** 3
            self.acceleration = self.acceleration + jerk
            if np.linalg.norm(self.acceleration) > 0.2:  # clip acceleration
                self.acceleration = self.acceleration / (5 * np.linalg.norm(self.acceleration))
            self.velocity = self.velocity + self.acceleration
        else:
            raise ValueError(f"Unknown dynamic type '{dynamic_type}'")

    def project_target(self, new_target: np.ndarray):
        """
        check if target can be reached by the robot and project otherwise
        Args:
            new_target: The new target to reach

        Returns: The projected target. This target is projected onto the circle that is within the robots reach, and
         onto the first period of the end-effector rotation

        """
        distance_to_origin = np.linalg.norm(new_target[:2])
        if distance_to_origin > self._num_links:
            new_target[:2] = (new_target[:2] / distance_to_origin) * self._num_links
        if self._include_rotation:  # make sure that the rotation is normalized to the first period
            new_target[2] = self.angle_normalize(new_target[2])

    def update_target_circles(self, current_robot_end_effector_state: np.ndarray) -> None:
        """
        check if the currently active circle has been reached. Do this only if there still is a target to reach
        Args:
            current_robot_end_effector_state: The current end-effector position of the robot. This may differ
            from the position of the cursor due to imprecise updates.

        Returns: None

        """
        if len(self._target_circles) > self._current_target:
            target_circle = self._target_circles[self._current_target]
            distance_to_current_target = target_circle.closest_distance(current_robot_end_effector_state[:2])
            if distance_to_current_target < 0:
                self._target_circles[self._current_target].plot(fill=True, fillcolor="g")
                self._current_target += 1
                if len(self._target_circles) > self._current_target:
                    self._target_circles[self._current_target].plot(fill=True, fillcolor="b")

    def compute_update_velocity(self, current_dimensionwise_error: np.ndarray,
                                rest_posture_scale: float,
                                stability_matrix: np.ndarray):
        """
        Computes the proposed velocity update of each robot joint. Does this using a damped pseudoivnerse of the
        jacobian of the joints wrt. the x and y position and potentially the angle of the end-effector.
        Factors in a rest posture if the scale is > 0
        Args:
            current_dimensionwise_error:
            rest_posture_scale:
            stability_matrix:

        Returns:

        """
        assert rest_posture_scale >= 0, f"Must have a non-negative rest_posture_scale, given '{rest_posture_scale}'"
        jacobian = self.jacobian(self.robot_joint_angles)
        if rest_posture_scale > 0:
            # damped pseudoinverse with rest posture
            distance_to_rest_posture = rest_posture_scale * (np.zeros(self._num_links) - self.robot_joint_angles)
            transformed_error = np.linalg.solve(jacobian @ jacobian.T + stability_matrix,
                                                current_dimensionwise_error - np.dot(jacobian,
                                                                                     distance_to_rest_posture))
            joint_update_velocity = jacobian.T @ transformed_error + distance_to_rest_posture
        else:
            # damped pseudoinverse method
            transformed_error = np.linalg.solve((jacobian @ jacobian.T) + stability_matrix,
                                                current_dimensionwise_error)
            joint_update_velocity = jacobian.T @ transformed_error
        return joint_update_velocity

    def bound_update_step(self, joint_update_velocity: np.ndarray, learning_rate: float,
                          total_update_bound: float, update_bound_type: Optional[str]) -> np.ndarray:
        """
        Bounds the norm of the current update step if it would exceed some given upper bound, or applies the learning
        rate otherwise. The bound may depend on the velocity or the acceleration of the update.
        Args:
            joint_update_velocity: The proposed update velocity for each joint.
            learning_rate: The learning rate to apply if the bound is not violated
            total_update_bound: The upper bound of the update. The bound is only applied if the norm of the
              joint velocity/acceleration exceeds this value
            update_bound_type: What kind of bound to use. May be None, "velocity" or "acceleration"

        Returns: The bounded/scaled update step

        """
        if update_bound_type is None:
            joint_update_velocity = joint_update_velocity * learning_rate
        elif update_bound_type == "velocity":
            # bound total update velocity
            total_velocity = np.linalg.norm(joint_update_velocity)
            if total_velocity > total_update_bound:
                joint_update_velocity = (joint_update_velocity / total_velocity) * total_update_bound
            else:
                joint_update_velocity = joint_update_velocity * learning_rate
        elif update_bound_type == "acceleration":
            # bound total update acceleration
            joint_update_acceleration = self._previous_joint_velocity - joint_update_velocity
            total_acceleration = np.linalg.norm(joint_update_acceleration)
            if total_acceleration > total_update_bound:
                joint_update_acceleration = (joint_update_acceleration / total_acceleration) * total_update_bound
                joint_update_velocity = self._previous_joint_velocity - joint_update_acceleration
            else:
                joint_update_velocity = joint_update_velocity * learning_rate
        else:
            raise ValueError(f"Unknown update_bound_type '{update_bound_type}'")
        return joint_update_velocity

    def render(self):
        current_color = self._colors[self._current_step % self._num_colors]

        # draw the marker of the current target. Since the marker is never deleted, this also draws a history of
        # where the robot was supposed to go
        from matplotlib.markers import MarkerStyle
        if self._include_rotation:
            m = MarkerStyle(f"$>$")
            m._transform.rotate_deg(np.degrees(self.target_end_effector_state[2]))
        else:
            m = MarkerStyle("x")
        a = plt.scatter(self.target_end_effector_state[0],
                        self.target_end_effector_state[1],
                        marker=m,
                        color=current_color,
                        s=120)

        positions = self.forward_kinematic(robot_joint_angles=self.robot_joint_angles)

        self.previous_robot.set_data(positions[:, 0], positions[:, 1])
        self.previous_robot.set_color(current_color)

        self.fig.canvas.restore_region(self.axbackground)
        self.ax.draw_artist(self.previous_robot)
        [self.ax.draw_artist(circle) for circle in self.circle_plots]
        self.ax.draw_artist(a)

        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    @staticmethod
    def angle_normalize(angles: np.array) -> np.array:
        """
        Normalizes the angles to be in [-pi, pi]
        :param angles: Unnormalized input angles
        :return: Normalized angles
        """
        return ((angles + np.pi) % (2 * np.pi)) - np.pi
