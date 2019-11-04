#include "Map.h"
#include "Feature.h"

#include "frontend/FullSystem.h"
#include "internal/GlobalCalib.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/PR.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;
using namespace ldso::internal;

namespace ldso {

    /**
     * @brief 添加关键帧
     ***/
    void Map::AddKeyFrame(shared_ptr<Frame> kf) {
        unique_lock<mutex> mapLock(mapMutex);
        if (frames.find(kf) == frames.end()) {
            frames.insert(kf);
        }
    }
    /**
     * @brief 多线程优化所有关键帧位姿图
     ***/
    bool Map::OptimizeALLKFs() {
        {
            unique_lock<mutex> lock(mutexPoseGraph);
            if (poseGraphRunning)
                return false; // is already running ...
            // if not, starts it
            poseGraphRunning = true;
            // lock frames to prevent adding new kfs
            unique_lock<mutex> mapLock(mapMutex);
            framesOpti = frames;
            currentKF = *frames.rbegin();
        }

        //  start the pose graph thread
        thread th = thread(&Map::runPoseGraphOptimization, this);
        th.detach();    // it will set posegraphrunning to false when returns
        return true;
    }
    /**
     * @brief 更新位姿后, 更新地图点坐标位置
     ***/
    void Map::UpdateAllWorldPoints() {
        unique_lock<mutex> lock(mutexPoseGraph);
        for (shared_ptr<Frame> frame: frames) {
            for (auto &feat: frame->features) {
                if (feat->point) {
                    feat->point->ComputeWorldPos();
                }
            }
        }
    }

    /**
     * @brief 优化整个地图，更新新的位姿下的地图点
     ***/
    void Map::runPoseGraphOptimization() {

        LOG(INFO) << "start pose graph thread!" << endl;
        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        typedef BlockSolver<BlockSolverTraits<7, 3> > BlockSolverType;
        BlockSolverType::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>();
        BlockSolverType *solver_ptr = new BlockSolverType(linearSolver);
        // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        // g2o::OptimizationAlgorithmDogleg *solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        // keyframes
        int maxKFid = 0;
        int cntEdgePR = 0;

        for (const shared_ptr<Frame> &fr: framesOpti) {

            // each kf has Sim3 pose
            int idKF = fr->kfId;
            if (idKF > maxKFid) {
                maxKFid = idKF;     // 更新最大的关键帧ID
            }

            // P+R
            VertexSim3 *vSim3 = new VertexSim3();
            Sim3 Scw = fr->getPoseOpti();
            CHECK(Scw.scale() > 0);
            vSim3->setEstimate(Scw);  // 估计值
            vSim3->setId(idKF);
            optimizer.addVertex(vSim3);

            // 固定当前帧
            //TODO 只需要固定当前帧么
            // fix the last one since we don't want to affect the frames in window
            if (fr == currentKF) {
                vSim3->setFixed(true);
            }

        }

        // edges
        for (const shared_ptr<Frame> &fr: framesOpti) {
            unique_lock<mutex> lock(fr->mutexPoseRel);
            for (auto &rel: fr->poseRel) {
                // 边的节点，当前vPR1 相连vPR2
                VertexSim3 *vPR1 = (VertexSim3 *) optimizer.vertex(fr->kfId);
                VertexSim3 *vPR2 = (VertexSim3 *) optimizer.vertex(rel.first->kfId);
                EdgeSim3 *edgePR = new EdgeSim3();
                if (vPR1 == nullptr || vPR2 == nullptr)
                    continue;
                edgePR->setVertex(0, vPR1);
                edgePR->setVertex(1, vPR2);
                edgePR->setMeasurement(rel.second.Tcr);
                // 信息矩阵
                if (rel.second.isLoop)
                    edgePR->setInformation(rel.second.info /* *10 */);
                else
                    edgePR->setInformation(rel.second.info);

                optimizer.addEdge(edgePR);
                cntEdgePR++;
            }
        }

        optimizer.initializeOptimization();
        optimizer.optimize(25);

        // 读出优化后的位姿，更新点的坐标
        // recover the pose and points estimation
        for (shared_ptr<Frame> frame: framesOpti) {
            VertexSim3 *vSim3 = (VertexSim3 *) optimizer.vertex(frame->kfId);
            Sim3 Scw = vSim3->estimate();
            CHECK(Scw.scale() > 0);

            frame->setPoseOpti(Scw);
            // reset the map point world position because we've changed the keyframe pose
            for (auto &feat: frame->features) {
                if (feat->point) {
                    feat->point->ComputeWorldPos();
                }
            }
        }

        poseGraphRunning = false;

        if (fullsystem) fullsystem->RefreshGUI();
    }

}
