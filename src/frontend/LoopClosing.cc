#include "Feature.h"
#include "internal/PR.h"
#include "internal/GlobalCalib.h"

#include "frontend/LoopClosing.h"
#include "frontend/FeatureMatcher.h"
#include "frontend/FullSystem.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>


namespace ldso {

    // -----------------------------------------------------------
    LoopClosing::LoopClosing(FullSystem *fullsystem) :
            kfDB(new DBoW3::Database(*fullsystem->vocab)), voc(fullsystem->vocab),
            globalMap(fullsystem->globalMap), Hcalib(fullsystem->Hcalib->mpCH),
            coarseDistanceMap(fullsystem->GetDistanceMap()),
            fullSystem(fullsystem) {

        mainLoop = thread(&LoopClosing::Run, this);
        idepthMap = new float[wG[0] * hG[0]];
    }
    /**
     * @brief：向闭环线程插入关键帧
     ***/
    void LoopClosing::InsertKeyFrame(shared_ptr<Frame> &frame) {
        unique_lock<mutex> lock(mutexKFQueue);
        KFqueue.push_back(frame);
    }
    /**
     * @brief：闭环检测的主循环, 
     ***/
    void LoopClosing::Run() {
        finished = false;

        while (1) {

            if (needFinish) {
                LOG(INFO) << "find loop closing thread need finish flag!" << endl;
                break;
            }
            // 从关键帧队列最前面取出一帧
            {
                // get the oldest one
                unique_lock<mutex> lock(mutexKFQueue);
                if (KFqueue.empty()) {
                    lock.unlock();
                    usleep(5000);
                    continue;
                }
                currentKF = KFqueue.front();
                KFqueue.pop_front();

                if (KFqueue.size() > 20)
                    KFqueue.clear();
                // 加入allKF(好像没啥用)
                allKF.push_back(currentKF);
            }
            // 计算特征向量
            currentKF->ComputeBoW(voc);
            // 检测是否有闭环
            if (DetectLoop(currentKF)) {
                // 全局优化是否空闲, 正在运行则false
                bool mapIdle = globalMap->Idle();  
                // 计算闭环帧和当前帧之间位姿
                if (CorrectLoop(Hcalib)) {
                    // start a pose graph optimization
                    if (mapIdle) {
                        // 进行全局优化
                        LOG(INFO) << "call global pose graph!" << endl;
                        bool ret = globalMap->OptimizeALLKFs();
                        if (ret)
                            needPoseGraph = false;
                    } else {
                        LOG(INFO) << "still need pose graph optimization!" << endl;
                        needPoseGraph = true;
                    }
                }
            }

            // 有未完成的全局优化, 则执行
            if (needPoseGraph && globalMap->Idle()) {
                LOG(INFO) << "run another pose graph!" << endl;
                if (globalMap->OptimizeALLKFs())
                    needPoseGraph = false;
            }

            usleep(5000);
        }

        finished = true;
    }
    /**
     * @brief 检测闭环, 得到闭环候选帧, 更新数据库
     * @param frame 当前帧
     ***/
    bool LoopClosing::DetectLoop(shared_ptr<Frame> &frame) {

        DBoW3::QueryResults results;
        // 最多返回一个, 临近的kfGap(10)个帧是不算在内的
        kfDB->query(frame->bowVec, results, 1, maxKFId - kfGap);
        
        // 如果没有, 则把当前帧加入数据库中
        if (results.empty()) {
            DBoW3::EntryId id = kfDB->add(frame->bowVec, frame->featVec);
            maxKFId = id;
            checkedKFs[id] = frame;  // 记录关键帧
            return false;
        }

        // 得到候选闭环帧
        DBoW3::Result r = results[0];
        candidateKF = checkedKFs[r.Id];

        // 找到当前帧相连的帧, 判断候选帧是不是在内
        auto connected = frame->GetConnectedKeyFrames();
        unsigned long minKFId = 9999999, maxKFId = 0;

        for (auto &kf: connected) {
            if (kf->kfId < minKFId)
                minKFId = kf->kfId;
            if (kf->kfId > maxKFId)
                maxKFId = kf->kfId;
        }

        if (candidateKF->kfId <= maxKFId && candidateKF->kfId >= minKFId) {
            // candidate is in active window
            return false;
        }

        LOG(INFO) << "candidate kf id: " << candidateKF->kfId << ", max id: " << maxKFId << ", min id: " << minKFId
                  << endl;

        // 如果得分小, 区别度大, 则也加入数据库
        if (r.Score < minScoreAccept) {
            DBoW3::EntryId id = kfDB->add(frame->bowVec, frame->featVec);
            maxKFId = id;
            checkedKFs[id] = frame;
            // 也当做闭环
            candidateKF = checkedKFs[r.Id];
            LOG(INFO) << "add loop candidate from " << candidateKF->kfId << ", current: " << frame->kfId << ", score: "
                      << r.Score << endl;
            return true;
        }

        // detected a possible loop
        candidateKF = checkedKFs[r.Id];
        LOG(INFO) << "add loop candidate from " << candidateKF->kfId << ", current: " << frame->kfId << ", score: "
                  << r.Score << endl;
        return true;   // don't add into database
    }
    /**
     * @brief 计算闭环帧和当前帧之间的相对位姿
     ***/
    bool LoopClosing::CorrectLoop(shared_ptr<CalibHessian> Hcalib) {

        // We compute first ORB matches for each candidate
        FeatureMatcher matcher(0.75, true);
        bool success = false;
        int nCandidates = 0; //candidates with enough matches

        // intrinsics
        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = Hcalib->fxl();
        K.at<float>(1, 1) = Hcalib->fyl();
        K.at<float>(0, 2) = Hcalib->cxl();
        K.at<float>(1, 2) = Hcalib->cyl();

        // [步骤1]: 通过词袋向量搜索匹配点, 构造3d点和2d点对应
        shared_ptr<Frame> pKF = candidateKF;
        vector<Match> matches;
        int nmatches = matcher.SearchByBoW(currentKF, pKF, matches);

        if (nmatches < 10) {
            LOG(INFO) << "no enough matches: " << nmatches << endl;
        } else {
            LOG(INFO) << "matches: " << nmatches << endl;

            // 得到闭环关键帧的3D点, 和当前帧的2D点
            // now we have a candidate proposed by dbow, let's try opencv's solve pnp ransac to see if there are enough inliers
            vector<cv::Point3f> p3d;
            vector<cv::Point2f> p2d;
            cv::Mat inliers;
            vector<int> matchIdx;

            for (size_t k = 0; k < matches.size(); k++) {
                auto &m = matches[k];
                shared_ptr<Feature> &featKF = pKF->features[m.index2];
                shared_ptr<Feature> &featCurrent = currentKF->features[m.index1];

                if (featKF->status == Feature::FeatureStatus::VALID &&
                    featKF->point->status != Point::PointStatus::OUTLIER) {
                    // there should be a 3d point
                    // pt unused?
                    //shared_ptr<Point> &pt = featKF->point;
                    // compute 3d pos in ref
                    Vec3f pt3 = (1.0 / featKF->invD) * Vec3f(
                            Hcalib->fxli() * (featKF->uv[0] - Hcalib->cxl()),
                            Hcalib->fyli() * (featKF->uv[1] - Hcalib->cyl()),
                            1
                    );
                    cv::Point3f pt3d(pt3[0], pt3[1], pt3[2]);
                    p3d.push_back(pt3d);
                    p2d.push_back(cv::Point2f(featCurrent->uv[0], featCurrent->uv[1]));
                    matchIdx.push_back(k);
                }
            }

            if (p3d.size() < 10) {
                LOG(INFO) << "3d points not enough: " << p3d.size() << endl;
                return false;
            }

            // [步骤2]: PnP求解得到当前帧的估计Sim3, 得到内点匹配对
            cv::Mat R, t;
            cv::solvePnPRansac(p3d, p2d, K, cv::Mat(), R, t, false, 100, 8.0, 0.99, inliers);
            int cntInliers = 0;

            vector<Match> inlierMatches;
            for (int k = 0; k < inliers.rows; k++) {
                inlierMatches.push_back(matches[matchIdx[inliers.at<int>(k, 0)]]);
                cntInliers++;
            }

            if (cntInliers < 10) {
                LOG(INFO) << "Ransac inlier not enough: " << cntInliers << endl;
                return false;
            }

            LOG(INFO) << "Loop detected from kf " << currentKF->kfId << " to " << pKF->kfId
                      << ", inlier matches: " << cntInliers << endl;

            // and then test with the estimated Tcw
            SE3 TcrEsti(
            SO3::exp(Vec3(R.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(2, 0))),
                    Vec3(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)));

            Sim3 ScrEsti(TcrEsti.matrix());
            ScrEsti.setScale(1.0);

            Mat77 Hessian;
            
            // [步骤3]: 优化估计的Tcr位姿
            if (ComputeOptimizedPose(pKF, ScrEsti, Hcalib, Hessian) == false) {
                return false;
            }

            // setup pose graph
            // [步骤4]: 把计算的结果加入相对位姿构成约束
            // 相当于DSO的优化是提供了一个相对位姿的约束
            // TODO 优化的位姿初始值不是DSO的优化结果, 两个绝对位姿实际上是分开了, 可改进
            {
                Sim3 SCurRef = ScrEsti;
                unique_lock<mutex> lock(currentKF->mutexPoseRel);
                currentKF->poseRel[pKF] = Frame::RELPOSE(SCurRef, Hessian, true);   // and an pose graph edge
                pKF->poseRel[currentKF] = Frame::RELPOSE(SCurRef.inverse(), Hessian, true);
            }

            success = true;

            if (setting_showLoopClosing && success) {
                LOG(INFO) << "please see loop closing between " << currentKF->kfId << " and " << pKF->kfId << endl;
                setting_pause = true;
                matcher.DrawMatches(currentKF, pKF, inlierMatches);
                setting_pause = false;
            }

            setting_pause = false;
        }
        nCandidates++;
        return success;
    }
    /**
     * @brief 对当前帧和闭环候选帧之间位姿进行优化
     * 
     * @param pKF           候选闭环帧
     * @param Scr           当前帧和候选帧间的Sim3估计值
     * @param Hcalib        相机参数
     * @param H             返回闭环帧和当前帧的Hessian矩阵
     * @param windowSize    网格大小
     * 
     ***/
    bool LoopClosing::ComputeOptimizedPose(shared_ptr<Frame> pKF, Sim3 &Scr, shared_ptr<CalibHessian> Hcalib,
                                           Mat77 &H, float windowSize) {

        LOG(INFO) << "computing optimized pose" << endl;
        int TH_HIGH = 50;
        vector<shared_ptr<Frame>> activeFrames = fullSystem->GetActiveFrames();
        // make the idepth map
        memset(idepthMap, 0, sizeof(float) * wG[0] * hG[0]);

        VecVec2 activePixels;
        // NOTE these residuals are not locked!
        // tcw unused?
        //SE3 Tcw = currentKF->getPose();
        // [步骤1]: 将滑窗内关键帧上的点投影到当前帧, 得到逆深度图
        for (shared_ptr<Frame> fh: activeFrames) {
            if (fh == currentKF) continue;
            for (shared_ptr<Feature> feat: fh->features) {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;
                    if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) {
                        shared_ptr<PointFrameResidual> r = ph->lastResiduals[0].first;
                        if (r->target.lock() != currentKF->frameHessian) continue;

                        int u = r->centerProjectedTo[0] + 0.5f;
                        int v = r->centerProjectedTo[1] + 0.5f;

                        float new_idepth = r->centerProjectedTo[2];
                        idepthMap[u + wG[0] * v] = new_idepth;
                    }
                }
            }
        }

        // dilate idepth by 1.
        //bug 这个activePixels是空的???
        for (auto &px: activePixels) {
            int idx = int(px[1] * wG[0] + px[0]);
            float idep = idepthMap[idx];

            idepthMap[idx - 1] = idep;
            idepthMap[idx + 1] = idep;
            idepthMap[idx + wG[0]] = idep;
            idepthMap[idx - wG[0]] = idep;
            idepthMap[idx - 1 + wG[0]] = idep;
            idepthMap[idx - 1 - wG[0]] = idep;
            idepthMap[idx + 1 - wG[0]] = idep;
            idepthMap[idx + 1 + wG[0]] = idep;
        }

        // [步骤2]: 使用网格进行Bow匹配, 得到匹配点对
        // optimize the current Tcw
        currentKF->SetFeatureGrid();

        // vector<shared_ptr<Feature>> matchedFeatures;
        VecVec3 matchedPoints;
        VecVec3 matchedFeatures;
        VecVec2 matchedPixels;

        // find more matches in the local map of pKF
        // 设置闭环候选帧上的点为候选点(candidateFeatures)
        vector<shared_ptr<Feature>> candidateFeatures;

        for (auto &feat: pKF->features) {
            if (feat->status == Feature::FeatureStatus::VALID &&
                feat->point->status != Point::PointStatus::OUTLIER) {
                candidateFeatures.push_back(feat);
            }
        }

        int nmatches = 0;
        Mat33 Ki;
        Ki << Hcalib->fxli(), 0, Hcalib->cxli(), 0, Hcalib->fyli(), Hcalib->cyli(), 0, 0, 1;

        // search by projection
        for (auto &p: candidateFeatures) {
            
            // 候选特征点投影到相机坐标
            Vec3 pRef = (1.0 / p->invD) * Vec3(
                    Hcalib->fxli() * (p->uv[0] - Hcalib->cxl()),
                    Hcalib->fyli() * (p->uv[1] - Hcalib->cyl()),
                    1
            );
            // 变换到当前帧坐标系, 并投影
            Vec3 pc = Scr * pRef;

            float x = pc[0] / pc[2];
            float y = pc[1] / pc[2];
            float u = Hcalib->fxl() * x + Hcalib->cxl();
            float v = Hcalib->fyl() * y + Hcalib->cyl();

            int bestDist = 256;
            int bestDist2 = 256;
            int bestIdx = -1;

            // look for points nearby
            auto indices = currentKF->GetFeatureInGrid(u, v, windowSize);
            float idepth = 0;

            // 在投影点附近找匹配的点
            for (size_t &k: indices) {
                shared_ptr<Feature> &feat = currentKF->features[k];
                if (fabsf(feat->angle - p->angle) < 0.2) {
                    // check rotation first
                    int dist = FeatureMatcher::DescriptorDistance(feat->descriptor,
                                                                  p->descriptor);

                    int ui = int(feat->uv[0] + 0.5f), vi = int(feat->uv[1] + 0.5f);
                    idepth = idepthMap[vi * wG[0] + ui];

                    if (idepth == 0) {
                        // NOTE don't need this idepth =0 because we need to estimate the scale
                        // well in stereo case you can still do this
                        continue;
                    }

                    if (dist < bestDist) {
                        bestDist2 = bestDist; // 次优值
                        bestDist = dist;
                        bestIdx = k;
                    } else if (dist < bestDist2) {
                        bestDist2 = dist;
                    }
                }
            }

            if (bestDist <= TH_HIGH) {
                auto bestFeat = currentKF->features[bestIdx];

                int ui = int(bestFeat->uv[0] + 0.5f), vi = int(bestFeat->uv[1] + 0.5f);
                idepth = idepthMap[vi * wG[0] + ui];

                Vec3 pcurr = (1.0f / idepth) * (Ki * Vec3(bestFeat->uv[0], bestFeat->uv[1], 1));
                matchedPoints.push_back(pRef);
                matchedFeatures.push_back(pcurr);

                matchedPixels.push_back(Vec2(bestFeat->uv[0], bestFeat->uv[1]));

                nmatches++; // 计数君
            }
        }

        if (nmatches < 10) {
            LOG(INFO) << "local map matches not enough: " << nmatches << endl;
            return false;
        }

        // [步骤3]: 利用匹配点3D误差\重投影误差进行优化, 得到位姿和Hessian
        // pose optimization, note there maybe some mismatches
        // NOTE seems like there are multiple solutions if just use 3d-3d point pairs
        // setup g2o and solve the problem
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        // current KF's pose
        VertexSim3 *vSim3 = new VertexSim3();
        vSim3->setId(0);
        vSim3->setEstimate(Scr);
        optimizer.addVertex(vSim3);

        float sigma = 1.0;
        float sigma2 = sigma * sigma;
        float infor = 1.0 / sigma2;
        float th = 5.991 * infor; // 自由度2

        vector<EdgePointSim3 *> edgesSim3;
        vector<EdgeProjectPoseOnlySim3 *> edgesProjection;
        for (size_t i = 0; i < matchedFeatures.size(); i++) {

            // 当前系下, 3D点之间的残差
            // EdgeProjectPoseOnlySim3 *eProj = new EdgeProjectPoseOnlySim3(Hcalib->mpCam, matchedPoints[i]);
            EdgePointSim3 *e3d = new EdgePointSim3(matchedPoints[i]);
            e3d->setId(i);
            e3d->setVertex(0, vSim3);

            Mat33 inforMat = infor * Matrix3d::Identity();
            e3d->setInformation(inforMat);    // TODO should not be identity.
            e3d->setMeasurement(matchedFeatures[i]);
            edgesSim3.push_back(e3d);

            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            rk->setDelta(th);
            e3d->setRobustKernel(rk);
            optimizer.addEdge(e3d);

            // 当前图像上的重投影误差
            EdgeProjectPoseOnlySim3 *eProj = new EdgeProjectPoseOnlySim3(Hcalib->camera, matchedPoints[i]);
            eProj->setVertex(0, vSim3);
            eProj->setInformation(Mat22::Identity());
            eProj->setMeasurement(matchedPixels[i]);
            optimizer.addEdge(eProj);
            edgesProjection.push_back(eProj);
        }

        LOG(INFO) << "Start optimization";
        optimizer.initializeOptimization(0); // 初始化为0level
        optimizer.optimize(10);

        // 去除外点
        int inliers = 0, outliers = 0;
        for (auto &e : edgesSim3) {
            if (e->chi2() > th || e->chi2() < 1e-9 /* maybe some bug in g2o */ ) {
                e->setLevel(1);  // 误差大的不包括在内
                outliers++;
            } else {
                e->setRobustKernel(nullptr);
                inliers++;
            }
        }

        LOG(INFO) << "inliers: " << inliers << ", outliers: " << outliers << endl;

        if (inliers < 15) // reject
            return false;

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        // decide the inliers
        Sim3 ScrOpti = vSim3->estimate();

        if (ScrOpti.scale() == Scr.scale() || std::isnan(ScrOpti.scale()) || ScrOpti.scale() < 0)  // optimization failed
            return false;

        // 返回优化后的位姿, 和优化后的Hessian矩阵
        Scr = ScrOpti;
        Eigen::Map<Mat77> hessianData(vSim3->hessianData());
        H = hessianData;

        return true;
    }
}
