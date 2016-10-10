
#include "stdafx.h"
#include "cv.h"

#include <iostream>
using namespace std;


// 最小二乘法最优化Levenberg-Marquardt
#include "ls_observation.h"
#include "ls_minimizer.h"


class my_obs : public observation
{
public:
	int k;
	virtual int eval_func(const double *state, double *fx, double *J, int Jstep, void **user_data) 
	{
		double t = exp(-1*state[1]*_data[0]);
		*fx = state[0]*t;
		if (J != 0)
		{
			J[0]=t;
			J[1]=-state[0]*_data[0]*exp(-1*state[1]*_data[0]);
		}
		return 1;
	}
};

int main( int argc, char** argv )
{
	class observation_manager obs_mgr;
	ls_minimizer lsm;
	double t[]={0.25,0.5,1,1.5,2,3,4,6,8};
	double c[]={19.21,18.15,15.36,14.10,12.89,9.32,7.45,5.24,3.01};
	double initial_state[]={10.0,0.5},* state,re;
	double residual;
	my_obs * o;
	const int n=9;
	int i;

	lsm.set_debug_msg_level(2);
	obs_mgr.init(2,1,sizeof(my_obs),1,10);
	lsm.init(20,10,10,&obs_mgr);
	lsm.init(2,1,1,&obs_mgr);

	// 加入观测值
	obs_mgr.begin_add_obs();
	for (i=0;i<n;i++)
	{
		double *v = &c[i];
		double *d = &t[i];
		o = new my_obs;
		o->set_values(v,d);
		obs_mgr.add_next_obs(o);
	}
	obs_mgr.end_add_obs();

	// 优化
	lsm.minimize_using_levenberg_marquardt_from(initial_state);
	state = lsm.get_state();
	re = lsm.compute_residual(state);

	cout <<endl<< "计算结果:" <<state[0]<<","<<state[1]*-1<<" 残差="<<re<<endl;
	cout <<endl;


}

