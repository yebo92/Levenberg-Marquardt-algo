#if !defined(_LS_MINIMIZER_H_)
#define _LS_MINIMIZER_H_

#ifdef max
#undef max
#endif

#include <limits>
#include <vector>
#include <cxcore.h>
//#include "mini_solver.h"
#include "ls_observation.h"
#include "growmat.h"
/*
术语：
状态向量(state 或 measurement vector)：也就是最小二乘法估计的参数。维数记为n(state_size)
观测向量(observation 或 parameter vector 或 b)：一次观测得到的m个测量(measurement)值。维数为
误差公式(error 或 epsilon)：e = f_i(state) - b ， 其中f_i()是第i次观测的向量函数:R(N)->R(M)
代价函数(cost 或 总残差residual)：F(x)=|e|=0.5*e'*e 为标量函数，也是优化的目标函数.
雅克比矩阵(J)：大小为m*n
海赛均值(H)：大小为n*n  H=J'*J
梯度向量(G)：G(x)=J'*f(x)

算法：

设计思路：
由于所有的观测都是用同一个方程计算，且大部分参数设置相同，故设计observation_manager来管理全部对象和全部参数
*/


class ls_minimizer
{
public:
	enum {MAX_B_SIZE = 4};

public:
	void set_debug_msg_level(int l);
	ls_minimizer();

	virtual ~ls_minimizer();

	// 访问状态
	double * get_state(){return _state;}
	// 初始化
	int init(int state_dims, int obs_dims, int data_dims, observation_manager * obs_mgr);
	// 设置用户参数
	void set_user_data(int slot_index, void * ptr);
	// 用LM算法最小化
	int minimize_using_levenberg_marquardt_from(double * initial_state);
	// 计算残差
	double compute_residual(double * state);

	// LM算法参数设置
	void lm_set_initial_lambda(double l){lm_initial_lambda = l;}
	void lm_set_max_iterations(int it){	lm_max_iterations = it;}
	void lm_set_max_failures_in_a_row(int f){lm_max_failures_in_a_row = f;}
	void lm_set_tol(double t){lm_tol = t;}
protected:
	// 初始化标志
	int _init_flag;
	// 用户自定义数据，用于eval_func
	void * _user_data[10];
	// 观测对象管理器
	observation_manager * _obs_mgr_ptr;
	// 参数向量维数
	int _state_dims;
	// 观测向量的维数
	int _obs_dims; 
	// 数据向量的维数
	int _data_dims;

	// 状态
	double * _state, * _lm_state;
	grow_mat _mStates;

	// 雅克比矩阵(M*N)
	grow_mat _mJ;
	// epsilon误差矩阵(M*1) (eps_previous)
	grow_mat _mEps;
	// 最小二乘问题的Hessian矩阵用JtJ矩阵来近似 (N*N)
	grow_mat _mHessian;
	// 求步长的线性方程的右端(F'(x)=J'f(x)) (N*1)
	grow_mat _mGradient;
	// 求迭代步长(N*1)
	grow_mat _mDs;

	// LM算法
	// 混合海赛矩阵，大小为N*N
	grow_mat _mAug_H;
	// 初始lambda值
	double lm_initial_lambda;
	// 混合方法 0:默认
	int lm_augment_method;
	// 最大迭代次数
	int lm_max_iterations;
	// 修改lambda最大失败次数
	int lm_max_failures_in_a_row;
	// 退出条件，最小余弦值
	double lm_tol;

protected:
	// 全部清空为缺省值
	void set_default_values(void);
	// 申请矩阵内存
	int alloc_matrices();
	// 释放所有内存
	void free_matrices(void);
	// 设置新状态
	void set_state(double * state, const double *new_state);
	// 计算雅克比矩阵_mJ，梯度_mGradient，海塞矩阵_mHessian和残差residual
	double build_J_G_H_r(double * state);
	// 计算LM混合矩阵
	int compute_augment_H(double lambda);

	// 显示状态值
	void show_state(int vl, double * state, double r);
};


#endif
