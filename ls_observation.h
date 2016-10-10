

#ifndef LS_OBSERVATION_TYPES_H
#define LS_OBSERVATION_TYPES_H

#include <vector>
#include "cv.h"

class observation;
using namespace std ;
typedef std::vector<observation *> observation_vector;

//! 观测对象管理类
/*! 
内存池功能，管理observaton对象的分配，保留，扩展，释放；
目前只处理所有observation大小固定的情况，同时保存在一块连续区域中。

buffer的分配：( 观测对象+内存池是否使用标记+观测向量+数据向量) + 指针数组
_observation_count*(_one_observation_size+sizeof(bool)+(_obs_dims*sizeof(double)+_data_dims*sizeof(double))+sizeof(void *) )

*/

class observation_manager
{
public:
	observation_manager();
	~observation_manager();

	enum OBS_MGR_METHOD {FixedSize/*,FixedMaxSize,ChangeableSize*/};
	void set_debug_msg_level(int l);

	// (1)初始化
	int init(int state_dims, int obs_dims, size_t obs_size, int data_dims, int reserve_n, OBS_MGR_METHOD method = FixedSize);
	// 观测向量的维数
	inline int get_obs_dims(){return _obs_dims;}
	// 数据向量的维数
	inline int get_data_dims(){return _data_dims;}
	// 返回指向vector的指针
	observation_vector * get_obs_vector();
	// 返回对象个数
	inline int get_obs_count(){return _observations.size();}

	// (2)迭代添加观测对象
	// 开始
	int begin_add_obs();
	// 添加1个对象
	int add_next_obs(observation * o);
	// 结束
	int end_add_obs();
	// 清空
	int clear_all_obs();
	// 显示调试信息
	void disp_all_obs();

	// (2). 内存池函数
	// 预留n个观测对象的内存空间
	int pool_reserve(int n, bool copy = false);
	// 分配内存（仅由new调用）
	void * pool_alloc(size_t size);
	// 释放内存（仅由delete调用）
	void pool_free(void * p, size_t size);

	// 添加多个对象
	int add_n_obs_values(double * v, double * d, int size){return 0;}
	// 返回指向第i个对象的指针
	int get_obs_ptr(int i){return 0;}
	// 排序
	void sort(void){;}
protected:
	// 设置内存池中分配C++对象的大小（不含数据）
	void pool_set_element_size(size_t s);
protected:
	// 初始化标志
	int _init_flag;
	// (1) 以下是所有观测对象共同的属性
	// 观测向量的维数
	int _obs_dims; 
	// 数据向量的维数
	int _data_dims;
	// 参数向量维数
	int _state_dims;


	// (3) 内存管理的内部参数
	// 当前观测对象的指针向量
	observation_vector _observations;

	// (4). 内存池
	// 指向C++观测对象起始位置的指针
	char * _pool_ptr;
	// 指向是否使用标记起始位置的指针
	char * _pool_used_ptr;
	// 指向观测值起始位置的指针
	char * _pool_obsv_ptr;
	// 指向数据值起始位置的指针
	char * _pool_data_ptr;

	// 观测对象大小（C++观测对象大小）（字节）
	size_t _element_size;
	// 每元素占用内存大小（字节）
	size_t _element_total_size;
	// 内存池大小（字节）
	size_t _pool_total_size;
	// 内存池可以容纳的最大对象数
	int _pool_max_count;
	// 下一个空白位置
	int _pool_next_index;
	// 分配内存的模式（mode=0无 mode=1迭代添加模式）
	int _pool_alloc_mode;

	// 缓冲区大小（字节）
	size_t _current_buffer_size;
	// 缓冲区，容纳所有的(观测对象+内存池元素是否使用+测量值+数据值+指针数组)
	char * _buffer_ptr;
	// 每次重新分配缓冲区的增量(4K)
	int _buffer_increment;
};

//! 观测对象类
/*! 
一次观测得到一个有关状态参数的观测向量. 
注意：观测对象的所有成员变量都没有初始化，需要用户负责
注意：观测对象是轻量级的，即只包含指向数据的指针，不负责内存的分配和释放。
*/
class observation
{
public:
	// 设置观测值和数据值
	inline void set_values(double * v, double * d)
	{
		if(v) memcpy(_obsv, v, _obj_mgr->get_obs_dims()*sizeof(double));
		if(d) memcpy(_data, d, _obj_mgr->get_data_dims()*sizeof(double));
	}
	double * get_obsv(){return _obsv;}
	double * get_data(){return _data;}
	int get_obs_dims(){return _obj_mgr->get_obs_dims();}
	void * operator new(size_t size) throw(bad_alloc);
	void operator delete(void * p, size_t size);// throw();
protected:
	// 指向观测向量值的指针
	double * _obsv;
	// 指向数据向量值的指针
	double * _data;
	// 指向观测对象管理器指针
	static observation_manager * _obj_mgr;

	friend class observation_manager;
	friend class ls_minimizer;
protected:
	// 设置缺省参数
	void set_default_values(void) { _obsv=0;_data=0;}
	// 计算本次观测的残差
	double residual(const double *computed_b)
	{
		int n;
		if (_obj_mgr==0) return 0.f;
		n = get_obs_dims();
		double r=0;
		for (int i=0; i<n; i++) {
			//double d = computed_b[i] - _obsv[i];
			double d = _obsv[i] - computed_b[i];
			r += d*d;
		}
		return r;
	}
	// 输入参数向量state计算观测
	// state:输入本次迭代估计的状态向量；fx:输出函数值；J:输出雅克比矩阵；user_data:输入指向指针的指针（指针数组）
	virtual int eval_func(const double *state, double *fx, double *J, int J_step, void **user_data) {return 0;}

	// 备注：以上3个指针，1个virtual指针，共占用16字节。
private:
	//inline void set_obj_mgr_values(class observation_manager * obj_mgr){if(!_obj_mgr) _obj_mgr=obj_mgr;}
	// virtual int get_nb_measures() {return _obj_mgr->get_obs_dims();}//（用于变长对象）
	// 当前观测对象的维数
	// int _obs_dims;//（用于变长对象）
	// observation_manager * _obj_mgr;
	// 获得目前观测对象的测量值个数，目前是固定的大小。
	//inline int get_nb_measures() {return _obj_mgr->get_obs_dims();}
};

// 模板继承
//template <int def_obs_dims, int def_data_dims> class observation2 : public observation<def_obs_dims>


#endif // LS_OBSERVATION_TYPES_H