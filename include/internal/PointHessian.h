#pragma once
#ifndef LDSO_POINT_HESSIAN_H_
#define LDSO_POINT_HESSIAN_H_

#include "Point.h"
#include "Settings.h"
#include "internal/Residuals.h"
#include "Feature.h"

namespace ldso {
    namespace internal {
        class PointFrameResidual;

        class ImmaturePoint;

        /**
         * Point hessian is the internal structure of a map point
         */
        class PointHessian {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            // create the point hessian from immature point
            PointHessian(shared_ptr<ImmaturePoint> rawPoint);

            PointHessian() {}
            /**
             * @brief 设置逆深度的值
             * @param idepth 逆深度
             ***/            
            inline void setIdepth(float idepth) {
                this->idepth = idepth;
                this->idepth_scaled = SCALE_IDEPTH * idepth;
                if (point->mHostFeature.expired()) {
                    LOG(FATAL) << "host feature expired!" << endl;
                }
                point->mHostFeature.lock()->invD = idepth;
            }
            /**
             * @brief 设置scaled的逆深度的值
             * @param idepth 乘了倍数的逆深度
             ***/      
            inline void setIdepthScaled(float idepth_scaled) {
                this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
                this->idepth_scaled = idepth_scaled;
                if (point->mHostFeature.expired()) {
                    LOG(FATAL) << "host feature expired!" << endl;
                }
                point->mHostFeature.lock()->invD = idepth;
            }
            /**
             * @brief 设置逆深度的线性化点值
             * @param idepth 逆深度
             ***/      
            inline void setIdepthZero(float idepth) {
                idepth_zero = idepth;
                idepth_zero_scaled = SCALE_IDEPTH * idepth;
                // 零空间的基(似乎没有使用)
                nullspaces_scale = -(idepth * 1.001 - idepth / 1.001) * 500;
            }
            /**
             * @brief 判断点是否超出边界, 满足要被边缘化or删除的条件
             * @param toMarg  要被边缘化的帧的向量
             ***/ 
            // judge if this point is out of boundary
            inline bool isOOB(std::vector<shared_ptr<FrameHessian>> &toMarg) {

                int visInToMarg = 0;
                for (shared_ptr<PointFrameResidual> &r : residuals) {
                    if (r->state_state != ResState::IN) continue;
                    for (shared_ptr<FrameHessian> k : toMarg)
                        if (r->target.lock() == k) visInToMarg++;
                }

                if ((int) residuals.size() >= setting_minGoodActiveResForMarg &&
                    numGoodResiduals > setting_minGoodResForMarg + 10 &&
                    (int) residuals.size() - visInToMarg < setting_minGoodActiveResForMarg)
                    return true;

                if (lastResiduals[0].second == ResState::OOB)
                    return true;
                if (residuals.size() < 2) return false;
                if (lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER)
                    return true;
                return false;
            }
            /**
             * @brief 该点是不是内点
             ***/ 
            inline bool isInlierNew() {
                return (int) residuals.size() >= setting_minGoodActiveResForMarg
                       && numGoodResiduals >= setting_minGoodResForMarg;
            }


            shared_ptr<Point> point = nullptr;

            float u = 0, v = 0;                 // pixel position
            float energyTH = 0;                 // 能量阈值
            bool hasDepthPrior = false;         // 有没有逆深度先验(初始化的点有)
            float my_type = 0;                  // 点的类型(来自ImmaturePoint)
            float idepth_scaled = 0;            // scaled逆深度
            float idepth_zero_scaled = 0;       // scaled的线性化点逆深度
            float idepth_zero = 0;              // 线性化点逆深度
            float idepth = 0;                   // 逆深度
            float step = 0;                     // 优化的increment
            float step_backup;                  // 优化中备份优化增量
            float idepth_backup;                // 优化中逆深度优化
            float nullspaces_scale;             // 尺度零空间基
            float idepth_hessian = 0;           // 逆深度的Hessian(协方差逆)
            float maxRelBaseline = 0;           // 最大的(虚拟)基线长度
            int numGoodResiduals = 0;           // 投影小于阈值的residual数目

            // residuals in many keyframes
            std::vector<shared_ptr<PointFrameResidual>> residuals;   // only contains good residuals (not OOB and not OUTLIER). Arbitrary order.

            // the last two residuals
            std::pair<shared_ptr<PointFrameResidual>, ResState> lastResiduals[2];  // contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).

            // static values
            float color[MAX_RES_PER_POINT];         // colors in host frame
            float weights[MAX_RES_PER_POINT];       // host-weights for respective residuals.
            unsigned char label[MAX_RES_PER_POINT] = {19};

            // ======================================================================== 、、
            // optimization data
            /**
            * @brief 逆深度的先验H, delta值
            ***/
            void takeData() {
                priorF = hasDepthPrior ? setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH : 0;
                if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
                    priorF = 0;
                deltaF = idepth - idepth_zero;
            }

            float priorF = 0;           // 先验的Hessian
            float deltaF = 0;           // 和先验的delta

            // H and b blocks
            float bdSumF = 0;                   // 点的 b=J*res 部分(先验 + 当前)
            float HdiF = 0;                     // 优化后逆深度的Hessian逆, 即协方差, 表示点的逆深度的不确定
            float Hdd_accLF = 0;                // 边缘化点的逆深度hessian
            VecCf Hcd_accLF = VecCf::Zero();    // 边缘化逆深度和相机参数部分hessian
            float bd_accLF = 0;                 // 边缘化逆深度的b
            float Hdd_accAF = 0;                // 当前点的逆深度Hessian
            VecCf Hcd_accAF = VecCf::Zero();    // 当前点逆深度和相机参数部分hessian
            float bd_accAF = 0;                 // 当前点的逆深度b
            bool alreadyRemoved = false;        // ???
        };

    }

}

#endif // LDSO_POINT_HESSIAN_H_
