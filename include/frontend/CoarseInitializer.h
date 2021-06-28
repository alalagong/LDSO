#pragma once
#ifndef LDSO_COARSE_INITIALIZER_H_
#define LDSO_COARSE_INITIALIZER_H_

#include "NumTypes.h"
#include "Settings.h"
#include "AffLight.h"
#include "internal/OptimizationBackend/MatrixAccumulators.h"

#include "Camera.h"
#include "Frame.h"
#include "Point.h"

using namespace ldso;
using namespace ldso::internal;

namespace ldso {

    /**
     * point structure used in coarse initializer
     */
    struct Pnt {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // index in jacobian. never changes (actually, there is no reason why).
        float u, v;

        // idepth / isgood / energy during optimization.
        float idepth;				//!< 该点对应参考帧的逆深度
        bool isGood;				//!< 点在新图像内, 相机前, 像素值有穷则好
        Vec2f energy;				//!< [0]残差的平方, [1]正则化项(逆深度减一的平方) // (UenergyPhotometric, energyRegularizer)	
        bool isGood_new;
        float idepth_new;			//!< 该点在新的一帧(当前帧)上的逆深度
        Vec2f energy_new;			//!< 迭代计算的新的能量

        float iR;					//!< 逆深度的期望值
        float iRSumNum;				//!< 子点逆深度信息矩阵之和

        float lastHessian;			//!< 逆深度的Hessian, 即协方差, dd*dd
        float lastHessian_new;		//!< 新一次迭代的协方差

        // max stepsize for idepth (corresponding to max. movement in pixel-space).
        float maxstep;				//!< 逆深度增加的最大步长

        // idx (x+y*w) of closest point one pyramid level above.
        int parent;		  			//!< 上一层中该点的父节点 (距离最近的)的id
        float parentDist;			//!< 上一层中与父节点的距离

        // idx (x+y*w) of up to 10 nearest points in pixel space.
        int neighbours[10];			//!< 图像中离该点最近的10个点
        float neighboursDist[10];   //!< 最近10个点的距离

        float my_type; 				//!< 第0层提取是1, 2, 4, 对应d, 2d, 4d, 其它层是1
        float outlierTH; 			//!< 外点阈值
    };

    /**
     * initializer for monocular slam
     */
    class CoarseInitializer {
    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        CoarseInitializer(int w, int h);

        ~CoarseInitializer();


        void setFirst(shared_ptr<CalibHessian> HCalib, shared_ptr<FrameHessian> newFrameHessian);

        bool trackFrame(shared_ptr<FrameHessian> newFrameHessian);

        void calcTGrads(shared_ptr<FrameHessian> newFrameHessian);

        int frameID = -1;
        bool fixAffine = true;
        bool printDebug = false;

        Pnt *points[PYR_LEVELS];
        int numPoints[PYR_LEVELS];
        AffLight thisToNext_aff;
        SE3 thisToNext;


        shared_ptr<FrameHessian> firstFrame;
        shared_ptr<FrameHessian> newFrame;
    private:
        Mat33 K[PYR_LEVELS];
        Mat33 Ki[PYR_LEVELS];
        double fx[PYR_LEVELS];
        double fy[PYR_LEVELS];
        double fxi[PYR_LEVELS];
        double fyi[PYR_LEVELS];
        double cx[PYR_LEVELS];
        double cy[PYR_LEVELS];
        double cxi[PYR_LEVELS];
        double cyi[PYR_LEVELS];
        int w[PYR_LEVELS];
        int h[PYR_LEVELS];

        void makeK(shared_ptr<CalibHessian> HCalib);

        bool snapped;
        int snappedAt;

        // pyramid images & levels on all levels
        Eigen::Vector3f *dINew[PYR_LEVELS];
        Eigen::Vector3f *dIFist[PYR_LEVELS];

        Eigen::DiagonalMatrix<float, 8> wM;

        // temporary buffers for H and b.
        Vec10f *JbBuffer;            // 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
        Vec10f *JbBuffer_new;

        Accumulator9 acc9;
        Accumulator9 acc9SC;

        Vec3f dGrads[PYR_LEVELS];

        float alphaK;
        float alphaW;
        float regWeight;
        float couplingWeight;

        Vec3f calcResAndGS(
                int lvl,
                Mat88f &H_out, Vec8f &b_out,
                Mat88f &H_out_sc, Vec8f &b_out_sc,
                const SE3 &refToNew, AffLight refToNew_aff,
                bool plot);

        Vec3f calcEC(int lvl); // returns OLD NERGY, NEW ENERGY, NUM TERMS.
        void optReg(int lvl);

        void propagateUp(int srcLvl);

        void propagateDown(int srcLvl);

        float rescale();

        void resetPoints(int lvl);

        void doStep(int lvl, float lambda, Vec8f inc);

        void applyStep(int lvl);

        void makeGradients(Eigen::Vector3f **data);

        void makeNN();
    };

    /**
     * minimal flann point cloud
     */
    struct FLANNPointcloud {
        inline FLANNPointcloud() {
            num = 0;
            points = 0;
        }

        inline FLANNPointcloud(int n, Pnt *p) : num(n), points(p) {}

        int num;
        Pnt *points;

        inline size_t kdtree_get_point_count() const { return num; }

        inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t /*size*/) const {
            const float d0 = p1[0] - points[idx_p2].u;
            const float d1 = p1[1] - points[idx_p2].v;
            return d0 * d0 + d1 * d1;
        }

        inline float kdtree_get_pt(const size_t idx, int dim) const {
            if (dim == 0) return points[idx].u;
            else return points[idx].v;
        }

        template<class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
    };
}

#endif // LDSO_COARSE_INITIALIZER_H_
