import numpy as np
from scipy import io


# for i in range(50):
#     delta_psi_decision_max = 0.01 * (i + 1)
#     a = np.load('E:/project/PPO_static/alpha/delta_psi_decision_max_{:.2f}.npy'.format(delta_psi_decision_max), allow_pickle=True)
#     b = np.load('E:/project/PPO_static/alpha/num_psi_alpha_{:.2f}.npy'.format(delta_psi_decision_max), allow_pickle=True)
#     c = np.load('E:/project/PPO_static/alpha/num_1_alpha_{:.2f}.npy'.format(delta_psi_decision_max), allow_pickle=True)
#     d = np.load('E:/project/PPO_static/alpha/num_all_alpha_{:.2f}.npy'.format(delta_psi_decision_max), allow_pickle=True)
#
#     io.savemat('E:/project/PPO_static/mat/delay/delay{}.mat'.format(round(delta_psi_decision_max * 100)), {'delay{}'.format(round(delta_psi_decision_max * 100)): a})
#     io.savemat('E:/project/PPO_static/mat/num_psi/num_psi{}.mat'.format(round(delta_psi_decision_max * 100)), {'num_psi{}'.format(round(delta_psi_decision_max * 100)): b})
#     io.savemat('E:/project/PPO_static/mat/num_1/num_1{}.mat'.format(round(delta_psi_decision_max * 100)), {'num_1{}'.format(round(delta_psi_decision_max * 100)): c})
#     io.savemat('E:/project/PPO_static/mat/num_all/num_all{}.mat'.format(round(delta_psi_decision_max * 100)), {'num_all{}'.format(round(delta_psi_decision_max * 100)): d})
#     print(b)


for i in range(5):
    alpha = 0.1 + 0.2 * i
    a = np.load('E:/project/PPO_static/test2/alpha_{:.2f}.npy'.format(alpha), allow_pickle=True)
    b = np.load('E:/project/PPO_static/test2/num_psi_alpha_{:.2f}.npy'.format(alpha), allow_pickle=True)
    c = np.load('E:/project/PPO_static/test2/num_1_alpha_{:.2f}.npy'.format(alpha), allow_pickle=True)
    d = np.load('E:/project/PPO_static/test2/num_all_alpha_{:.2f}.npy'.format(alpha), allow_pickle=True)
    e = np.load('E:/project/PPO_static/test2/num_differ_{:.2f}.npy'.format(alpha), allow_pickle=True)

    io.savemat('E:/project/PPO_static/test2mat/delay/delay{}.mat'.format(round(alpha * 100)), {'delay{}'.format(round(alpha * 100)): a})
    io.savemat('E:/project/PPO_static/test2mat/num_psi/num_psi{}.mat'.format(round(alpha * 100)), {'num_psi{}'.format(round(alpha * 100)): b})
    io.savemat('E:/project/PPO_static/test2mat/num_1/num_1{}.mat'.format(round(alpha * 100)), {'num_1{}'.format(round(alpha * 100)): c})
    io.savemat('E:/project/PPO_static/test2mat/num_all/num_all{}.mat'.format(round(alpha * 100)), {'num_all{}'.format(round(alpha * 100)): d})
    io.savemat('E:/project/PPO_static/test2mat/num_differ/num_differ{}.mat'.format(round(alpha * 100)), {'num_differ{}'.format(round(alpha * 100)): e})
    print(a)

# for i in range(5):
#     alpha = 0.1 + 0.2 * i
#     a = np.load('E:/project/PPO_static/test3/evalu_compre_{:.2f}.npy'.format(alpha), allow_pickle=True)
#
#     io.savemat('E:/project/PPO_static/test3mat/evalu_compre{}.mat'.format(round(alpha * 100)), {'evalu_compre{}'.format(round(alpha * 100)): a})
