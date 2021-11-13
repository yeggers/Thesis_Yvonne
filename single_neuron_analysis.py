import numpy as np 
import matplotlib.pyplot as plt 

lambda_X_list = [5]
alpha_list = [0.1]
v_th_list = [0.3]
t_max = 7
time = np.arange(t_max)
input_slow = np.zeros_like(time)
input_slow[1:4] = np.array([2,1,0.5])
#input_slow[np.arange(5, t_max, 10)] = 1
input_fast = np.zeros_like(time)
input_fast[1:4] = np.array([1, 1.5, 1.4999])
#input_fast[np.arange(5,t_max, 5)] = 1
w = 1


lambda_v_list = [5]

colours = ['r', 'b', 'g', 'k' ,'c', 'm', 'y', 'lime', 'gold']
linestyles = ['-', '--', 'dotted']

i = 0 
for v_th_index_index, v_th in enumerate(v_th_list):
    for lambda_v_index, lambda_v in enumerate(lambda_v_list):
        for lambda_X_index, lambda_X in enumerate(lambda_X_list):
            for index_alpha, alpha in enumerate(alpha_list):
                X_slow = 0
                X_collect_slow = []

                X_fast = 0
                X_collect_fast = []

                v_slow = 0
                v_collect_slow = []

                v_fast = 0
                v_collect_fast = []

                i_collect_slow = []

                i_collect_fast = []

                output_slow = np.zeros_like(time)
                output_fast = np.zeros_like(time)

                for t in time:
                    dX_slow = 1/lambda_X *(-X_slow) + 1/lambda_X*alpha * input_slow[t] 
                    X_slow += dX_slow
                    i_slow = w*input_slow[t] - X_slow
                    i_collect_slow.append(i_slow)
                    X_collect_slow.append(X_slow)

                    dv_slow = 1/lambda_v*(-v_slow) + 1/lambda_v*i_slow
                    v_slow += dv_slow

                
                    v_collect_slow.append(v_slow)
                    if v_slow >= v_th or v_fast>= v_th:
                            v_slow = 0
                            output_slow[t] = 1

                    

                    dX_fast = 1/lambda_X *(-X_fast) + 1/lambda_X*alpha * input_fast[t] 
                    X_fast += dX_fast
                    i_fast = w*input_fast[t] - X_fast
                    i_collect_fast.append(i_fast)
                    X_collect_fast.append(X_fast)

                    dv_fast = 1/lambda_v*(-v_fast) + 1/lambda_v*i_fast
                    v_fast += dv_fast

                    v_collect_fast.append(v_fast)
                    if v_fast >= v_th or v_slow >= v_th:
                        v_fast = 0
                        output_fast[t] = 1



                plt.plot(X_collect_slow, label = 'slow, alpha = {alpha}, lambda_X = {lambda_X}, lambda_v = {lambda_v}, v_th = {v_th}'.format(alpha = alpha, lambda_X = lambda_X, lambda_v = lambda_v,  v_th = v_th), color = colours[i])
                plt.plot(X_collect_fast, label = 'fast, alpha = {alpha}, lambda_X = {lambda_X}, lambda_v = {lambda_v},  v_th = {v_th}'.format(alpha = alpha, lambda_X = lambda_X, lambda_v = lambda_v,  v_th = v_th), linestyle = '--', color = colours[i])
                # plt.plot(output_slow, color = colours[i], label = 'slow, alpha = {alpha}, lambda_X = {lambda_X}, lambda_v = {lambda_v},  v_th = {v_th}'.format(alpha = alpha, lambda_X = lambda_X, lambda_v = lambda_v,  v_th = v_th), linestyle = linestyles[i])
                #plt.plot(output_fast, color = colours[i], label = 'fast, alpha = {alpha}, lambda_X = {lambda_X}, lambda_v = {lambda_v},  v_th = {v_th}'.format(alpha = alpha, lambda_X = lambda_X, lambda_v = lambda_v,  v_th = v_th), linestyle = linestyles[i])
                # plt.plot(abs(np.array(v_collect_slow) - np.array(v_collect_fast)), label = 'alpha = {alpha}, lambda_X = {lambda_X}, lambda_v = {lambda_v}'.format(alpha = alpha, lambda_X = lambda_X, lambda_v = lambda_v), color = colours[i])
                i += 1

plt.legend()
plt.xlabel('Time [ms]')
plt.ylabel('Voltage [-]')
plt.grid()
plt.show()
