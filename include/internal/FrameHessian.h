#pragma once
#ifndef LDSO_FRAME_HESSIAN_H_
#define LDSO_FRAME_HESSIAN_H_

#include "Frame.h"
#include "NumTypes.h"
#include "Settings.h"
#include "AffLight.h"

#include "internal/FrameFramePrecalc.h"

using namespace std;

namespace ldso {

    namespace internal {

        class PointHessian;

        class CalibHessian;

        //! 帧间相对状态
        struct FrameFramePrecalc;   

        /**
         * Frame hessian is the internal structure used in dso
         */
        class FrameHessian {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            FrameHessian(shared_ptr<Frame> frame) {
                this->frame = frame;
            }

            ~FrameHessian() {
                for (int i = 0; i < pyrLevelsUsed; i++) {
                    delete[] dIp[i];
                    delete[]  absSquaredGrad[i];
                }
            }

            //***************** accessors *****************
            /**
             * @brief 得到相机外参待估计值(也是线性化点)
             **/ 
            EIGEN_STRONG_INLINE const SE3 &get_worldToCam_evalPT() const {
                return worldToCam_evalPT;
            }
            /**
             * @brief 线性化点的位姿增量值(0), 线性化点光度参数值
             **/ 
            EIGEN_STRONG_INLINE const Vec10 &get_state_zero() const {
                return state_zero;
            }
            /**
             * @brief (优化后)当前状态的位姿增量值, 线性化点光度参数值
             **/ 
            EIGEN_STRONG_INLINE const Vec10 &get_state() const {
                return state;
            }
            /**
             * @brief (优化后)当前状态的scaled位姿增量值, 线性化点scaled光度参数值
             **/             
            EIGEN_STRONG_INLINE const Vec10 &get_state_scaled() const {
                return state_scaled;
            }

            // state - state0
            /**
             * @brief (优化后)当前状态(位姿+光度)与线性化点状态增量
             **/ 
            EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const {
                return get_state() - get_state_zero();
            }
            /**
             * @brief 位姿左扰动增量, 
             **/ 
            inline Vec6 w2c_leftEps() const {
                return get_state_scaled().head<6>();
            }
            /**
             * @brief 当前光度参数值
             **/ 
            inline AffLight aff_g2l() {
                return AffLight(get_state_scaled()[6], get_state_scaled()[7]);
            }
            /**
             * @brief 线性化点处的光度参数值
             **/ 
            inline AffLight aff_g2l_0() const {
                return AffLight(get_state_zero()[6] * SCALE_A, get_state_zero()[7] * SCALE_B);
            }
            // 设置线性化点增量
            void setStateZero(const Vec10 &state_zero);
            /**
             * @brief 设置当前位姿增量, 更新相机外参&位姿, 设置当前光度参数值
             * 
             * @param state  优化后的状态
             **/ 
            inline void setState(const Vec10 &state) {

                this->state = state;
                state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
                state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
                state_scaled[6] = SCALE_A * state[6];
                state_scaled[7] = SCALE_B * state[7];
                state_scaled[8] = SCALE_A * state[8];
                state_scaled[9] = SCALE_B * state[9];

                PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
                PRE_camToWorld = PRE_worldToCam.inverse();
            };
            /**
             * @brief 设置当前scaled位姿增量, 更新相机外参&位姿, 设置当前scaled光度参数
             * 
             * @param state_scaled  优化后的scaled状态
             **/ 
            inline void setStateScaled(const Vec10 &state_scaled) {

                this->state_scaled = state_scaled;
                state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
                state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
                state[6] = SCALE_A_INVERSE * state_scaled[6];
                state[7] = SCALE_B_INVERSE * state_scaled[7];
                state[8] = SCALE_A_INVERSE * state_scaled[8];
                state[9] = SCALE_B_INVERSE * state_scaled[9];

                PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
                PRE_camToWorld = PRE_worldToCam.inverse();
            };
            /**
             * @brief 设置位姿增量状态, 设置当前光度参数, 并且更新线性化点
             * 
             * @param worldToCam_evalPT 当前估计的位姿状态(线性化点)
             * @param state   当前各状态的增量值
             **/             
            inline void setEvalPT(const SE3 &worldToCam_evalPT, const Vec10 &state) {

                this->worldToCam_evalPT = worldToCam_evalPT;
                setState(state);
                setStateZero(state);
            };
            /**
             * @brief 设置scaled位姿增量状态, 设置scaled当前光度参数, 并且更新线性化点
             * 
             * @param worldToCam_evalPT  当前估计的位姿状态(线性化点)
             * @param aff_g2l  当前光度参数值
             **/   
            // set the pose Tcw
            inline void setEvalPT_scaled(const SE3 &worldToCam_evalPT, const AffLight &aff_g2l) {
                Vec10 initial_state = Vec10::Zero();
                initial_state[6] = aff_g2l.a;
                initial_state[7] = aff_g2l.b;
                this->worldToCam_evalPT = worldToCam_evalPT;
                setStateScaled(initial_state);
                setStateZero(this->get_state());
            };

            /**
             * @brief create the images and gradient from original image
             * @param [in] HCalib camera intrinsics with hessian
             */
            void makeImages(float *image, const shared_ptr<CalibHessian> &HCalib);

            /**
             * @brief 各状态的先验Hessian, 第一帧的位姿有, 其他帧没有(0), 光度是一直都有
             **/   
            inline Vec10 getPrior() {
                Vec10 p = Vec10::Zero();
                if (frame->id == 0) {
                    p.head<3>() = Vec3::Constant(setting_initialTransPrior);
                    p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
                    if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR) {
                        p.head<6>().setZero();
                    }
                    p[6] = setting_initialAffAPrior;
                    p[7] = setting_initialAffBPrior;
                } else {
                    if (setting_affineOptModeA < 0) {
                        p[6] = setting_initialAffAPrior;
                    } else {
                        p[6] = setting_affineOptModeA;
                    }
                    if (setting_affineOptModeB < 0) {
                        p[7] = setting_initialAffBPrior;
                    } else {
                        p[7] = setting_affineOptModeB;
                    }
                }
                p[8] = setting_initialAffAPrior;
                p[9] = setting_initialAffBPrior;
                return p;
            }

            inline Vec10 getPriorZero() {
                return Vec10::Zero();
            }

            // Data
            int frameID = 0;              // key-frame ID, will be set when adding new keyframes
            shared_ptr<Frame> frame = nullptr;    // link to original frame

            // internal structures used in DSO
            // image pyramid and gradient image
            // dIp[i] is the i-th pyramid with dIp[i][0] is the original image，[1] is dx and [2] is dy
            // by default, we have 6 pyramids, so we have dIp[0...5]
            // created in makeImages()
            Vec3f *dIp[PYR_LEVELS];

            // absolute squared gradient of each pyramid
            float *absSquaredGrad[PYR_LEVELS];  // only used for pixel select (histograms etc.). no NAN.

            // dI = dIp[0], the first pyramid
            Vec3f *dI = nullptr;     // trace, fine tracking. Used for direction select (not for gradient histograms etc.)
            unsigned char *label = nullptr;
            unsigned char *bel = nullptr;

            // Photometric Calibration Stuff
            float frameEnergyTH = 8 * 8 * patternNum;    // set dynamically depending on tracking residual
            float ab_exposure = 0;  // the exposure time // 曝光时间

            bool flaggedForMarginalization = false; // flag for margin
            Mat66 nullspaces_pose = Mat66::Zero();
            Mat42 nullspaces_affine = Mat42::Zero();
            Vec6 nullspaces_scale = Vec6::Zero();

            // variable info.
            SE3 worldToCam_evalPT;  // Tcw (in ORB-SLAM's framework)

            // state variable，[0-5] is se3, 6-7 is light param a,b
            Vec10 state;        // [0-5: worldToCam-leftEps. 6-7: a,b]

            // variables used in optimization
            Vec10 step = Vec10::Zero();
            Vec10 step_backup = Vec10::Zero();
            Vec10 state_backup = Vec10::Zero();
            Vec10 state_zero = Vec10::Zero();
            Vec10 state_scaled = Vec10::Zero();

            // precalculated values, will be send to frame when optimization is done.
            SE3 PRE_worldToCam; // TCW
            SE3 PRE_camToWorld; // TWC

            std::vector<FrameFramePrecalc, Eigen::aligned_allocator<FrameFramePrecalc>> targetPrecalc;

            // ======================================================================================== //
            // Energy stuffs
            // Frame status: 6 dof pose + 2 dof light param
            void takeData();                // take data from frame hessian
            Vec8 prior = Vec8::Zero();             // prior hessian (diagonal)
            Vec8 delta_prior = Vec8::Zero();       // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
            Vec8 delta = Vec8::Zero();             // state - state_zero.
            int idx = 0;                         // the id in the sliding window, used for constructing matricies

        };
    }
}

#endif // LDSO_FRAME_HESSIAN_H_
