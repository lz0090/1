# import torch
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# import gym
# import argparse
# from normalization import Normalization, RewardScaling
# from replaybuffer import ReplayBuffer
# from RLbrain import PPO_continuous
# import gym.envs.classic_control.myenv.myenv
#
#
# def evaluate_policy_dynamic_all(args, env, agent, state_norm):
#     times = 1
#     evaluate_reward_dynamic_all = 0
#     for _ in range(times):
#         s = env.reset()
#         s_tmp_now = s  # 没有正则化的当前状态
#         if args.use_state_norm:
#             s = state_norm(s, update=False)  # During the evaluating,update=False
#         done = False
#         episode_reward = 0
#         loc_T = dict()  # 创建一个状态时延字典
#         while not done:
#             a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
#             print(a)
#
#             if args.policy_dist == "Beta":
#                 action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
#             else:
#                 action = a
#             s_, r, done,  informa = env.step(action)
#             s_tmp = s_  # 没有正则化的下一状态
#             if args.use_state_norm:
#                 s_ = state_norm(s_, update=False)
#             if s_ != s:
#                 loc_T.update({float(s_tmp_now): informa['action is OK']})
#             episode_reward += r
#             s = s_
#             s_tmp_now = s_tmp
#
#         evaluate_reward_dynamic_all += episode_reward
#
#     return evaluate_reward_dynamic_all / times/60, informa, loc_T
#
#
# def evaluate_policy_dynamic_psi(args, env, agent, state_norm):
#     times = 1
#     evaluate_reward_dynamic_psi = 0
#     delta_psi_decision_max = 0.2  # 重分配决策变量的变化量阈值，大于这个值代表系统需要进行重新分配
#     for _ in range(times):
#         s = env.reset()
#         s_tmp_now = s  # 没有正则化的当前状态
#         if args.use_state_norm:
#             s = state_norm(s, update=False)  # During the evaluating,update=False
#         done = False
#         episode_reward = 0
#
#         loc_T = dict()  # 创建一个状态时延字典
#         num_agv_psi = 0
#         while not done:
#
#             if s == state_norm(np.array([23.0]), update=False):
#                 a = agent.evaluate(s)
#
#             s_, r, done, informa = env.step(a)
#             s_tmp1 = s_  # 没有正则化的下一状态
#             s_ = state_norm(s_, update=False)
#             if s_ != s:
#                 loc_T.update({float(s_tmp_now): informa['action is OK']})
#             episode_reward += r
#             s = s_
#             s_tmp_now = s_tmp1
#
#             if gym.envs.classic_control.myenv.myenv.flag_agv_tmp == 1 and num_agv_psi == 0:   # 正常情况下else中的程序只会在AGV需要切换节点时执行
#                 a = agent.evaluate(s)
#                 # s_, r, done, informa = env.step(a)
#                 # s_tmp2 = s_  # 没有正则化的下一状态
#                 # s_ = state_norm(s_, update=False)
#                 # if s_ != s:
#                 #     loc_T.update({float(s_tmp_now): float(informa['action is OK'])})
#                 # episode_reward += r
#                 # s = s_
#                 # s_tmp_now = s_tmp2
#                 num_agv_psi += 1
#
#             elif gym.envs.classic_control.myenv.myenv.delta_psi_decision >= delta_psi_decision_max:  #  重决策变量的变化值的绝对值大于阈值，进行重新分配
#                 a = agent.evaluate(s)
#
#
#
#         evaluate_reward_dynamic_psi += episode_reward
#
#     return evaluate_reward_dynamic_psi / times, loc_T
#
#
#
# def evaluate_policy_static(args, env, agent, state_norm):
#     times = 1
#     evaluate_reward_static = 0
#     for _ in range(times):
#         s = env.reset()
#         s_tmp_now = s  # 没有正则化的当前状态
#         if args.use_state_norm:
#             s = state_norm(s, update=False)  # During the evaluating,update=False
#         done = False
#         episode_reward = 0
#
#         loc_T = dict()  # 创建一个状态时延字典
#         num_agv_static = 0
#         while not done:
#
#             if s == state_norm(np.array([23.0]), update=False):
#                 a = agent.evaluate(s)
#
#             s_, r, done, informa = env.step(a)
#             s_tmp1 = s_  # 没有正则化的下一状态
#             s_ = state_norm(s_, update=False)
#             if s_ != s:
#                 loc_T.update({float(s_tmp_now): informa['action is OK']})
#             episode_reward += r
#             s = s_
#             s_tmp_now = s_tmp1
#
#             if gym.envs.classic_control.myenv.myenv.flag_agv_tmp == 1 and num_agv_static == 0:   # 正常情况下只会在AGV需要切换节点时执行
#                 a = agent.evaluate(s)
#                 num_agv_static += 1
#
#
#
#         evaluate_reward_static += episode_reward
#
#     return evaluate_reward_static / times, loc_T
#
# def evaluate_policy(args, env, agent, state_norm):
#     times = 1
#     evaluate_reward = 0
#     evaluate_t = 0
#     step_time = 0
#     for _ in range(times):
#         s = env.reset()
#         if args.use_state_norm:
#             s = state_norm(s, update=False)  # During the evaluating,update=False
#         done = False
#         episode_reward = 0
#         while not done:
#             a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
#             if args.policy_dist == "Beta":
#                 action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
#             else:
#                 action = a
#             print(a)
#             s_, r, done, informa = env.step(action)
#
#             # evaluate_t += informa['T']
#
#             if args.use_state_norm:
#                 s_ = state_norm(s_, update=False)
#             episode_reward += r
#             s = s_
#             step_time += 1
#         evaluate_reward += episode_reward
#         if done and step_time == 50:
#             judgement = 1
#             print(1)
#             evaluate_t = evaluate_t / 50
#         else:
#             judgement = 0
#             evaluate_t = evaluate_t / 50
#
#     return evaluate_reward / times, evaluate_t, informa
#
#
# def main(args, env_name, number, seed):
#     env = gym.make(env_name)
#     env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
#     # Set random seed
#     env.seed(seed)
#     env.action_space.seed(seed)
#     env_evaluate.seed(seed)
#     env_evaluate.action_space.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     # args.state_dim = env.observation_space.shape[0]
#     args.state_dim = 1
#     args.action_dim = env.action_space.shape[0]
#     args.max_action = float(env.action_space.high[0])
#     args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
#     print("env={}".format(env_name))
#     print("state_dim={}".format(args.state_dim))
#     print("action_dim={}".format(args.action_dim))
#     print("max_action={}".format(args.max_action))
#     print("max_episode_steps={}".format(args.max_episode_steps))
#
#     evaluate_num = 0  # Record the number of evaluations
#     evaluate_rewards = []  # Record the rewards during the evaluating
#     evaluate_average_T = []
#     total_steps = 0  # Record the total steps during the training
#
#     replay_buffer = ReplayBuffer(args)
#     agent = PPO_continuous(args)
#
#     # Build a tensorboard
#     writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))
#
#     state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
#
#     if args.use_reward_norm:  # Trick 3:reward normalization
#         reward_norm = Normalization(shape=1)
#     elif args.use_reward_scaling:  # Trick 4:reward scaling
#         reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
#
#     while total_steps < args.max_train_steps:
#         s = env.reset()
#         global flag_action
#         if args.use_state_norm:
#             s = state_norm(s)
#         if args.use_reward_scaling:
#             reward_scaling.reset()
#         episode_steps = 0
#         done = False
#         while not done:
#             episode_steps += 1
#
#
#             a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
#             if args.policy_dist == "Beta":
#                 action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
#             else:
#                 action = a
#             s_, r, done, _ = env.step(action)
#             if s != s_:
#                 flag_action = action  # 当下一状态不等于当前状态，说明此时的action满足条件，存为flag_action
#
#             if args.use_state_norm:
#                 s_ = state_norm(s_)
#             if args.use_reward_norm:
#                 r = reward_norm(r)
#             elif args.use_reward_scaling:
#                 r = reward_scaling(r)
#
#             # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
#             # dw means dead or win,there is no next state s';
#             # but when reaching the max_episode_steps,there is a next state s' actually.
#             if done and episode_steps != args.max_episode_steps:
#                 dw = True
#             else:
#                 dw = False
#
#             # Take the 'action'，but store the original 'a'（especially for Beta）
#             replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
#             s = s_
#             total_steps += 1
#
#             # When the number of transitions in buffer reaches batch_size,then update
#             if replay_buffer.count == args.batch_size:
#                 agent.update(replay_buffer, total_steps)
#                 replay_buffer.count = 0
#
#             # Evaluate the policy every 'evaluate_freq' steps
#             if total_steps % args.evaluate_freq == 0:
#                 evaluate_num += 1
#                 evaluate_reward, information, judge = evaluate_policy(args, env_evaluate, agent, state_norm)
#                 evaluate_rewards.append(evaluate_reward)
#                 evaluate_average_T.append(information)
#                 print("evaluate_num:{} \t evaluate_reward:{} \t  information :{} \t".format(evaluate_num, evaluate_reward, judge))
#
#                 # if evaluate_reward > 19 * np.power(10, 4):  # 说明模型已经收敛
#                 #     np.save('./loc_delay/loc_delay_dynamic_all.npy', loc_t_dynamic_all)  # 保存全动态的状态-时延数据
#                 #     evaluate_reward_static, loc_t_static = evaluate_policy_static(args, env_evaluate, agent, state_norm)
#                 #     np.save('./loc_delay/loc_delay_static.npy', loc_t_static)      # 保存静态的状态-时延数据
#                 #     evaluate_reward_dynamic_psi, loc_delay_dynamic_psi = evaluate_policy_dynamic_psi(args, env_evaluate, agent, state_norm)
#                 #     np.save('./loc_delay/loc_delay_dynamic_psi.npy', loc_delay_dynamic_psi)  # 保存决策变量的状态-时延数据
#
#                 writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
#                 # Save the rewards
#                 if evaluate_num % args.save_freq == 0:
#                     np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))
#
#
#
#
#
#
#
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
#     parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
#     parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
#     parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
#     parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
#     parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
#     parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
#     parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
#     parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
#     parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
#     parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
#     parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
#     parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
#     parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
#     parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
#     parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
#     parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
#     parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
#     parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
#     parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
#     parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
#     parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
#     parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
#     parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
#
#     args = parser.parse_args()
#
#     # env_name = ['BipedalWalker-v3', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
#     # env_index = 1
#
#     main(args, env_name="MyEnv-v0", number=1, seed=10)


import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from RLbrain import PPO_discrete


def evaluate_policy(args, env, agent, state_norm):
    times = 1
    evaluate_reward = 0
    evaluate_t = 0
    step_time = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            print(a)
            s_, r, done, informa = env.step(action)

            # evaluate_t += informa['T']

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
            step_time += 1
        evaluate_reward += episode_reward
        if done and step_time == 50:
            judgement = 1
            print(1)
            evaluate_t = evaluate_t / 50
        else:
            judgement = 0
            evaluate_t = evaluate_t / 50

    return evaluate_reward / times, evaluate_t, informa


def main(args, env_name, number, seed):
    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = 1
    args.action_dim = 23
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_average_T = []
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, information, judge = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                evaluate_average_T.append(information)
                print("evaluate_num:{} \t evaluate_reward:{} \t  information :{} \t".format(evaluate_num, evaluate_reward, judge))


                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    main(args, env_name="MyEnv-v0", number=1, seed=0)