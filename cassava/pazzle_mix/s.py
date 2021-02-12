class FeatureGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def norm(self, cx, cy):
        return torch.sqrt(torch.pow(cx, 2) + torch.pow(cy, 2))

    def get_angle(self, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
        a_x = p1_x - p2_x
        a_y = p1_y - p2_y
        b_x = p3_x - p2_x
        b_y = p3_y - p2_y

        prod = a_x * b_x + a_y * b_y
        norm = self.norm(a_x, a_y) * self.norm(b_x, b_y)
        return torch.acos(prod / norm)

    def forward(self, x):
        # head y to left wrist y
        head_to_left_wrist_y = torch.abs(x[1] - x[7 * 2 + 1])
        # head y to right wrist y
        head_to_right_wrist_y = torch.abs(x[1] - x[4 * 2 + 1])
        # head y to left shoulder y
        head_to_left_shoulder_y = torch.abs(x[1] - x[5 * 2 + 1])
        # head y to right should
        head_to_right_shoulder_y = torch.abs(x[1] - x[2 * 2 + 1])

        angle_left_shoulder_elbow_wrist = self.get_angle(p1_x=x[5 * 2], p1_y=x[5 * 2 + 1],
                                                         p2_x=x[6 * 2], p2_y=x[6 * 2 + 1],
                                                         p3_x=x[7 * 2], p3_y=x[7 * 2 + 1])

        angle_right_shoulder_elbow_wrist = self.get_angle(p1_x=x[2 * 2], p1_y=x[2 * 2 + 1],
                                                          p2_x=x[3 * 2], p2_y=x[3 * 2 + 1],
                                                          p3_x=x[4 * 2], p3_y=x[4 * 2 + 1])

        angle_left_shoulder_hip_knee = self.get_angle(p1_x=x[5 * 2], p1_y=x[5 * 2 + 1],
                                                      p2_x=x[11 * 2], p2_y=x[11 * 2 + 1],
                                                      p3_x=x[12 * 2], p3_y=x[12 * 2 + 1])

        angle_right_shoulder_hip_knee = self.get_angle(p1_x=x[2 * 2], p1_y=x[2 * 2 + 1],
                                                       p2_x=x[8 * 2], p2_y=x[8 * 2 + 1],
                                                       p3_x=x[9 * 2], p3_y=x[9 * 2 + 1])

        angle_left_hip_knee_wrist_ankle = self.get_angle(p1_x=x[8 * 2], p1_y=x[8 * 2 + 1],
                                                         p2_x=x[10 * 2], p2_y=x[10 * 2 + 1],
                                                         p3_x=x[12 * 2], p3_y=x[12 * 2 + 1])

        angle_right_hip_knee_wrist_ankle = self.get_angle(p1_x=x[9 * 2], p1_y=x[9 * 2 + 1],
                                                          p2_x=x[11 * 2], p2_y=x[11 * 2 + 1],
                                                          p3_x=x[13 * 2], p3_y=x[13 * 2 + 1])

        angle_left_elbow_shoulder_hip = self.get_angle(p1_x=x[4 * 2], p1_y=x[4 * 2 + 1],
                                                       p2_x=x[2 * 2], p2_y=x[2 * 2 + 1],
                                                       p3_x=x[8 * 2], p3_y=x[8 * 2 + 1])

        angle_right_elbow_shoulder_hip = self.get_angle(p1_x=x[5 * 2], p1_y=x[5 * 2 + 1],
                                                        p2_x=x[3 * 2], p2_y=x[3 * 2 + 1],
                                                        p3_x=x[7 * 2], p3_y=x[7 * 2 + 1])

        return head_to_left_wrist_y, head_to_right_wrist_y, \
               head_to_left_shoulder_y, head_to_right_shoulder_y, \
               angle_left_shoulder_elbow_wrist, angle_right_shoulder_elbow_wrist, \
               angle_left_shoulder_hip_knee, angle_right_shoulder_hip_knee, \
               angle_left_hip_knee_wrist_ankle, angle_right_hip_knee_wrist_ankle, \
               angle_left_elbow_shoulder_hip, angle_right_elbow_shoulder_hip
