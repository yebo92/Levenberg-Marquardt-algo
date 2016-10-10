// ls_minimizer.cpp: implementation of the ls_minimizer class.
//////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "ls_minimizer.h"


ls_minimizer::ls_minimizer()
{
	set_default_values();
}

 ls_minimizer::~ls_minimizer()
{

}

void ls_minimizer::set_debug_msg_level(int l)
{

}

void ls_minimizer::set_default_values(void)
{
	int i;
	for (i=0;i<10;i++)
		_user_data[i]=0;
	lm_initial_lambda = 1e-3;
	lm_max_iterations = 50;
	lm_augment_method = 1;
	lm_max_failures_in_a_row = 10;
	lm_tol = 1e-9;
	_state = 0;
}

int ls_minimizer::alloc_matrices()
{
	const int obs_count = _obs_mgr_ptr->get_obs_count();
	if (_init_flag==0 || obs_count<=0 || _state_dims<=0)
		return 0;
	const int N = _state_dims;
	// 假设每个观测对象得到的观测值数量固定，故M＝观测个数×观测维数
	const int M = _obs_dims*obs_count;
	const int D = _data_dims;

	// 检查是否第一次初始化？
	if (_mJ.get_ptr()==0)
	{
		if(!_mStates.init(2, N))
			goto error_out;
		_state = _mStates.data.db;
		_lm_state = _mStates.data.db + N;
		if(!_mJ.init(M, N))
			goto error_out;
		if(!_mEps.init(M, 1))
			goto error_out;
		if(!_mHessian.init(N, N))
			goto error_out;
		if(!_mGradient.init(N, 1))
			goto error_out;
		if(!_mDs.init(N,1))
			goto error_out;
		if(!_mAug_H.init(N,N))
			goto error_out;
		return 1;
	}
 
	// 若矩阵都存在，则只需要改变大小即可！
	if(!_mStates.resize(2, N))
		goto error_out;
	_state = _mStates.data.db;
	_lm_state = _mStates.data.db + N;
	if(!_mJ.resize(M, N))
		goto error_out;
	if(!_mEps.resize(M, 1))
		goto error_out;
	if(!_mHessian.resize(N, N))
		goto error_out;
	if(!_mGradient.resize(N, 1))
		goto error_out;
	if(!_mDs.resize(N,1))
		goto error_out;
	if(!_mAug_H.resize(N,N))
		goto error_out;
	return 1;
error_out:
	free_matrices();
	return 0;
}

void ls_minimizer::free_matrices(void)
{
	_mStates.free();
	_mJ.free();
	_mEps.free();
	_mHessian.free();
	_mGradient.free();
	_mDs.free();
	_mAug_H.free();
}

int ls_minimizer::init(int state_dims, int obs_dims, int data_dims, observation_manager * obs_mgr)
{
	if (state_dims<=0 || obs_dims<=0 || obs_mgr==0)
		return 0;
	_obs_mgr_ptr = obs_mgr;
	if (state_dims!=_state_dims || obs_dims!=_obs_dims)
	{
		_state_dims = state_dims;
		_obs_dims = obs_dims;
		alloc_matrices();
	}

	_init_flag = 1;
	return 1;
}

void ls_minimizer::set_user_data(int slot_index, void * ptr)
{
	assert(slot_index<10);
	_user_data[slot_index] = ptr;
}

// 设置新的状态，
void ls_minimizer::set_state(double * state, const double *new_state)
{
	// 比较新状态和原来的状态是否相等？
	if (memcmp(new_state, state, sizeof(double)*_state_dims)==0) return;
	// 若不相等，则将新状态复制到旧状态
	for(int i = 0; i < _state_dims; i++)
		state[i] = new_state[i];
}


double ls_minimizer::compute_residual(double * state)
{
	double residual = 0.;
	double b[MAX_B_SIZE];
	observation_vector * observations =	_obs_mgr_ptr->get_obs_vector();

	for (observation_vector::iterator it = observations->begin();
		it != observations->end(); ++it)
	{
		observation *obs = *it;
		double res=0.f;
		double * ob = obs->get_obsv();
		// 调用评价函数求本状态对应的理论值fx
		obs->eval_func(state, b, 0, 0, _user_data);
		// 通过权重等计算残差，并累加起来
		res = obs->residual(b);
		assert(res >= 0);
		residual += res;
	}
	return residual;
}

double ls_minimizer::build_J_G_H_r(double * state)
{
	double residual = 0.;
	double b[MAX_B_SIZE];
	observation_vector * observations =	_obs_mgr_ptr->get_obs_vector();
	const int state_size = this->_state_dims;
	int n;

	n = 0;
	for (observation_vector::iterator it = observations->begin();
		it != observations->end(); ++it)
	{
		observation *obs = *it;
		double res=0.f;
		double * eps = (double *)(_mEps.data.ptr + n*_mEps.step);
		double * ob = obs->get_obsv();
		// 调用评价函数求本状态对应的理论值fx
		obs->eval_func(state, b, (double *)(_mJ.data.ptr + n*_mJ.step), _mJ.step, _user_data);
		// 通过权重等计算残差，并累加起来
		for (int i=0; i<obs->get_obs_dims(); ++i)
		{
			double d = ob[i] - b[i];
			res += d*d;
			eps[i] = d ;
		}
		assert(res >= 0);
		residual += res;
		n += _obs_mgr_ptr->get_obs_dims();
	}

	// 计算梯度 g = J' * eps
	cvGEMM(&_mJ, &_mEps, 1, 0, 1, &_mGradient, CV_GEMM_A_T);

	// 计算最小二乘问题的海赛矩阵
	cvMulTransposed(&_mJ, &_mHessian, 1);

	return residual;
}

int ls_minimizer::compute_augment_H(double lambda)
{
	int i;
	const int state_size = this->_state_dims;
	// 最简单的混合！
	switch(lm_augment_method)
	{
		case 0:
		{
			cvCopy(&_mHessian,&_mAug_H);
			for(i = 0; i < state_size; i++)
			{
				double * augH = (double *)(_mAug_H.data.ptr + _mAug_H.step*i);
				augH[i] = augH[i] + lambda;
			}break;
		}
		case 1:
		{
			cvCopy(&_mHessian,&_mAug_H);
			for(i = 0; i < state_size; i++)
			{
				double * augH = (double *)(_mAug_H.data.ptr + _mAug_H.step*i);
				augH[i] = augH[i] + lambda*(1+augH[i]*augH[i]);
			}break;
		}
	}
	return 1;
}

void ls_minimizer::show_state(int vl, double * state, double r)
{
	const int state_size = this->_state_dims;
	for(int i = 0; i < state_size; i++) 
	{
		printf("%g", state[i]);
		if (i < state_size - 1)
			printf(", ");
	}
	printf(" --> %g\n", r);
}

int ls_minimizer::minimize_using_levenberg_marquardt_from(double * initial_state)
{
	if (_init_flag==0)
		return 0;

	// 根据需要申请内存
	if(alloc_matrices()==0)
		return 0;
	
	double lambda = lm_initial_lambda;
	int iter = 0,comput_j=0;
	int failures_in_a_row = 0;
	int terminate_reason = 0;
	int updateJ;
	double r=0.f,lm_r=0.f;
	double * state = _state;
	double * lm_state = _lm_state;
	double tol;

	set_state(state,initial_state);
	updateJ=1;

	while(iter < lm_max_iterations)
	{
		if(updateJ==1)
		{
			// 计算雅克比矩阵，海塞矩阵和残差
			double temp_r = build_J_G_H_r(state);comput_j++;
			if(iter==0) r = temp_r;
			iter ++;
			printf("LM: ---------------------------- %2d ----------------------------\n",iter);
		}

		// 根据阻尼系数lamda混合得到H矩阵
		compute_augment_H(lambda);


		// 计算步长ds
		if (!cvSolve(&_mAug_H,&_mGradient,&_mDs))
		{
			terminate_reason = -1;
			goto lm_end;
		}
		// 计算新的可能估计值lm_state
		for(int i = 0; i < _state_dims; i++)
		{
			double * ds = (double *)(_mDs.data.ptr + _mDs.step*i);
			lm_state[i] = state[i] +ds[0];
		}
		// 根据lm_state求残差e
		lm_r = compute_residual(lm_state);

		if (lm_r<r)
		{
			// 若残差减小
			r=lm_r;
			tol = cvDotProduct(&_mDs, &_mDs);
			if (tol<lm_tol)
			{
				terminate_reason = 3;
				goto lm_end;
			}
			set_state(state,lm_state);
			failures_in_a_row = 0;
			lambda=lambda/10;
			updateJ=1;
			printf("LM: iteration succeeded: new lambda = %g\n",lambda);
			show_state(1, state, r);
		}
		else
		{
			// 若残差变大
			failures_in_a_row++;
			if (failures_in_a_row > lm_max_failures_in_a_row) 
			{
				terminate_reason = 2;
				goto lm_end;
			}
			updateJ=0;
			if (lambda == 0) lambda = 1e-3; else lambda *= 10;
			printf("LM: iteration failed --> %g new lambda = %g\n",lm_r,lambda);
		}
	}
lm_end:
	if (iter >= lm_max_iterations)
		terminate_reason = 4;

	switch(terminate_reason) 
	{
    case 2: printf("LM: %d failures in a row.\n",failures_in_a_row); break;
    case 3: printf("LM: Termination criterion satisfied: tol = %g < %g\n",tol,lm_tol);
		break;
    case 4: printf("LM: Too many iterations. max_iter = %d\n",lm_max_iterations); break;
    case -1: printf("LM: step_solver failed.\n"); break;
    default:
		printf("LM: unkown termination reason\n");
	}
	printf("LM: "); show_state(1, state, r);
	return 1;
}

