import gym
import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal


class ReplayBeffer():
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return torch.FloatTensor(state_list).to(device), \
               torch.FloatTensor(action_list).to(device), \
               torch.FloatTensor(reward_list).unsqueeze(-1).to(device), \
               torch.FloatTensor(next_state_list).to(device), \
               torch.FloatTensor(done_list).unsqueeze(-1).to(device)

    def buffer_len(self):
        return len(self.buffer)

# Critic网络
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SoftQNet, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# Actor网络
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

        self.log_std_linear = nn.Linear(256, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action

    # 运行返回行动和log
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = (normal.log_prob(mean + std * z.to(device)) 
                    - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob


class SAC:
    def __init__(self, env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, N):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # 初始化超参量
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -self.action_dim
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=q_lr) 
        self.alphas = []
        
        # 初始化网络
        self.q_net = [SoftQNet(self.state_dim, self.action_dim).to(device) for i in range(N)]
        self.t_q_net = [SoftQNet(self.state_dim, self.action_dim).to(device) for i in range(N)]
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)

        self.q_optimizer = [optim.Adam(self.q_net[i].parameters(), lr=q_lr) for i in range(N)]
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # 初始化经验池
        self.buffer = ReplayBeffer(buffer_maxlen)

    def get_action(self, state):
        action = self.policy_net.action(state)
        return action

    def update(self, batch_size, G, M, N):
        q_bias_list = []
        for k in range(G):
            state, action, reward, next_state, done = self.buffer.sample(batch_size)
            new_action, log_prob = self.policy_net.evaluate(next_state)
            
            #随机从N中选取 m个序号
            m = np.random.choice(N, M, replace=False)
            
            #从target critic网络得到最小的Q值
            with torch.no_grad():
                target_value = self.t_q_net[m[0]](next_state, new_action)
                for i in range(1,M):
                    target_value = torch.min(self.t_q_net[m[i]](next_state, new_action), target_value)
            
            # 计算得到 y值来更新critic网络
            target_value = target_value - log_prob * self.alpha.to(device)
            target_q_value = reward + done * self.gamma * target_value
            
            # 对于每个网络分别进行更新
            for i in range(N):
                #计算 loss值
                q_value = self.q_net[i](state, action)
                q_value_loss = F.mse_loss(q_value, target_q_value.detach())
                
                #参数记录，与更新无关
                q = q_value.cpu().detach().numpy()
                t_q = target_q_value.cpu().detach().numpy()
                t_q[t_q < 10] = 10
                q_bias_list.append((q-t_q)/abs(t_q))
                
                #梯度更新
                self.q_optimizer[i].zero_grad()
                q_value_loss.backward()
                self.q_optimizer[i].step()
                
                #软更新target网络
                for target_param, param in zip(self.t_q_net[i].parameters(), self.q_net[i].parameters()):
                    target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        
        #重新从经验值中选取样本
        state, action, reward, next_state, done = self.buffer.sample(batch_size)  
        action_now , log_porb = self.policy_net.evaluate(state)
        
        #计算平均 q值
        avg_q_value = 0
        for i in range(N):
            avg_q_value += self.q_net[i](state, action_now)
        avg_q_value /= N
        
        #得到loss
        policy_loss = (log_prob * self.alpha.to(device) - avg_q_value).mean()
        
        #更新actor网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        alpha_loss = - (self.log_alpha.exp() * (log_prob.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()
        self.alphas.append(self.alpha.detach())
        
        return np.mean(q_bias_list)

def main(env, agent, Episode, batch_size, N, M, G):
    Return = []
    action_range = [env.action_space.low, env.action_space.high]
    
    epi = 0
    score = 0
    bias = 0
    state = env.reset()
    for episode in range(Episode):
        epi += 1
        #获取行动
        action = agent.get_action(state)
        action_in =  action * (action_range[1] - action_range[0]) / 2.0 +  (action_range[1] + action_range[0]) / 2.0
        #获取结果并加入经验池
        next_state, reward, done, _ = env.step(action_in)
        done_mask = 0.0 if done else 1.0
        agent.buffer.push((state, action, reward, next_state, done_mask))
        state = next_state
        score += reward
        
        #更新
        if agent.buffer.buffer_len() > 5000:
            bias += agent.update(batch_size, G, M, N)
        
        #游戏结束时输出并重置
        if done:
            print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent.buffer.buffer_len()))
            with open('result.txt','a') as f:
                print(episode," ",score," ",bias/epi, file=f)  #文件的写操作
            Return.append(score)
            epi = 0
            score = 0
            bias = 0
            state = env.reset()

    env.close()
    plt.plot(Return)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    env = gym.make("Walker2d")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #按照 MBPO的设置定义超参量
    tau = 0.005
    gamma = 0.99
    q_lr = 3e-4
    value_lr = 3e-4
    policy_lr = 3e-4
    buffer_maxlen = 1000000

    Episode = 300000
    batch_size = 256

    agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, 5)
    main(env, agent, Episode, batch_size, 5, 2, 10)