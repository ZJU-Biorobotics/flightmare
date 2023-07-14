import torch
import numpy as np
import collections
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import random
import time

device = torch.device('cuda')

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, self.odom_cb)
        rospy.Subscriber("/depth", Image, self.depth_cb)
        rospy.Subscriber("/final_cost", Float32, self.cost_cb)
        rospy.Subscriber("/rl_goal", PoseStamped, self.goal_cb)
        self.bridge = CvBridge()
        self.cur_state = collections.namedtuple('cur_state', ['cur_pos', 'goal_pos', 'rel_pos', 'cur_vel', 'cur_q', 'cur_omg', 'depth', 'cost'], defaults=[None]*7)
        self.cur_state.cur_pos = np.array([0, 0, 0])
        self.cur_state.goal_pos = np.array([0, 0, 0])
        self.cur_state.rel_pos = np.array([0, 0, 0])
        self.cur_state.cur_q = np.array([0, 0, 0, 1])
        self.cur_state.cur_vel = np.array([0, 0, 0])
        self.cur_state.cur_omg = np.array([0, 0, 0])
        self.cur_state.depth = np.array([0])


    def add(self):
        self.buffer.append((self.cur_state.rel_pos, self.cur_state.cur_vel, self.cur_state.cur_q, self.cur_state.cur_omg, self.cur_state.depth, self.cur_state.cost))

    def odom_cb(self, odom_msg: Odometry):
        self.cur_state.cur_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        self.cur_state.cur_q = np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
        self.cur_state.cur_vel = np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z])
        self.cur_state.cur_omg = np.array([odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z])

    def depth_cb(self, depth_msg: Image):
        depth = self.bridge.imgmsg_to_cv2(depth_msg)
        depth = np.asarray(depth)
        self.cur_state.depth = depth

    def goal_cb(self, goal_msg: PoseStamped):
        self.cur_state.goal_pos = np.array([goal_msg.pose.position.x, goal_msg.pose.position.y, goal_msg.pose.position.z])
    
    def cost_cb(self, cost_msg:Float32):
        self.cur_state.cost = cost_msg.data
        self.cur_state.rel_pos = self.cur_state.goal_pos - self.cur_state.cur_pos
        self.add()
        # rospy.loginfo("fucking nice "+ str(len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        rel_pos, cur_vel, cur_q, cur_omg, depth, cost = zip(*batch)
        try:
            return np.array(rel_pos), np.array(cur_vel), np.array(cur_q), np.array(cur_omg), np.array(depth), np.array(cost)
        except ValueError as e:
            print(f'rel_pos: {rel_pos}', f'cur_vel: {cur_vel}', f'cur_q: {cur_q}', f'cur_omg: {cur_omg}', f'depth: {depth}', f'cost: {cost}')



class RewardNet(torch.nn.Module):
    def __init__(self):
        super(RewardNet, self).__init__()
        self.cnns = torch.nn.Sequential(
            torch.nn.LazyConv2d(8, 9),
            torch.nn.ReLU(),
            torch.nn.LazyConv2d(8, 5),
            torch.nn.ReLU(),
            torch.nn.LazyConv2d(4, 3),
            torch.nn.Flatten(),
        )
        self.mlp1 = torch.nn.Sequential(
            torch.nn.LazyLinear(512),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(512),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(64)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(64),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(1)
        ) 
    
    def forward(self, rel_pos, cur_vel, cur_q, cur_omg, depth):
        depth_encode = self.mlp1(self.cnns(depth))
        depth_encode_and_other = torch.concat((depth_encode, rel_pos, cur_vel, cur_q, cur_omg), dim=1)
        # print(f"depth_encode: {depth_encode.shape}")
        # print(f"depth_encode_and_other: {depth_encode_and_other.shape}")
        return self.mlp2(depth_encode_and_other)


if __name__ == '__main__':
    rospy.init_node("learning_reward")
    time.sleep(1)
    rb = ReplayBuffer(200000)
    BATCH_SZIE = 128
    START_LEARNING_SIZE = 256
    reward_net = RewardNet()
    # reward_net.double()
    reward_net.to(device)
    print("Cuda Onset")
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=2e-3)
    while not rospy.is_shutdown():
        if len(rb) > START_LEARNING_SIZE:
            for i in range(1000000):
                # print("111111")
                rel_pos, cur_vel, cur_q, cur_omg, depth, cost = rb.sample(BATCH_SZIE)
                rel_pos = torch.tensor(rel_pos, dtype=torch.float32).to(device)
                cur_vel = torch.tensor(cur_vel, dtype=torch.float32).to(device)
                cur_q = torch.tensor(cur_q, dtype=torch.float32).to(device)
                cur_omg = torch.tensor(cur_omg, dtype=torch.float32).to(device)
                depth = torch.unsqueeze(torch.tensor(depth, dtype=torch.float32), 1).to(device)
                cost = torch.tensor(cost, dtype=torch.float32).to(device)
                # print(f'rel_pos.shape: {rel_pos.shape}')
                # print(f'cur_vel.shape: {cur_vel.shape}')
                # print(f'cur_q.shape: {cur_q.shape}')
                # print(f'cur_omg.shape: {cur_omg.shape}')
                # print(f'depth.shape: {depth.shape}')
                # print(f'cost.shape: {cost.shape}')

                out = reward_net(rel_pos, cur_vel, cur_q, cur_omg, depth)
                reward_loss = torch.mean(torch.nn.functional.mse_loss(out, cost))
                optimizer.zero_grad()
                reward_loss.backward()
                optimizer.step()
                # print(f'out.shape: {out.shape}')
                if i % 10 == 0:
                    print(f'episode {i}: loss {float(reward_loss)}')
    rospy.spin()
    