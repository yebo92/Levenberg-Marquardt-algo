// 自动增长的矩阵

#ifndef GROW_MAT_H
#define GROW_MAT_H

#include "cv.h"

class grow_mat : public CvMat 
{
public:
	grow_mat();
	// 特殊构造函数，复制data数据到当前矩阵，用法类似CvMAT，不同点是本函数复制数据data
	grow_mat(int r, int c, int type=CV_64FC1, void* data = 0, int step = CV_AUTOSTEP);
	~grow_mat();

	// 初始化
	int init(int r, int c, int type=CV_64FC1, int maxrows=0, int maxcols=0);
	// 改变大小
	int resize(int r, int c);
	// 保留存储空间
	int reserve(int maxrows, int maxcols);
	// 返回真实的矩阵指针
	inline CvMat * get_ptr(){return _mat;}
	// 释放内存
	void free();
	// 设置参数
	inline void set_expand_factor(double f){if(f>1.f) _expand_factor = f;}
	inline void set_copy_flag(int c){_copy_flag=c;}
	inline void set_zero_flag(int z){_zero_flag=z;}
private:
	// 真实的矩阵，保存实际内存
	CvMat * _mat;
	// 增长系数
	double _expand_factor;
	// 增长时是否复制数据
	int _copy_flag;
	// 创建时候是否清空
	int _zero_flag;

	void clear_all(void);
};

#endif
