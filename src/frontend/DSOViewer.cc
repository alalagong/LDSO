#include <thread>
#include <pangolin/pangolin.h>
#include <sys/time.h>

#include "Feature.h"
#include "frontend/DSOViewer.h"
#include "internal/GlobalCalib.h"
#include "internal/ImmaturePoint.h"

namespace ldso {

    /**
     * @brief 从关键帧中读取位姿和地图点信息
     ***/
    void KeyFrameDisplay::setFromKF(shared_ptr<FrameHessian> fh, shared_ptr<CalibHessian> HCalib) {

        if (fh->frame) {
            setFromF(fh->frame, HCalib);
        } else
            camToWorld = Sim3(fh->PRE_camToWorld.matrix());

        auto fr = fh->frame;
        int npoints = 0;
        for (auto feat: fr->features) {
            if (feat->point && feat->point->mpPH) {
                npoints++;
            } else if (feat->status == Feature::FeatureStatus::IMMATURE && feat->ip) {
                npoints++;
            }
        }
        // 扩充点的数量
        if (numSparseBufferSize < npoints) {
            if (originalInputSparse != 0) delete originalInputSparse;
            numSparseBufferSize = npoints + 100;
            originalInputSparse = new InputPointSparse<MAX_RES_PER_POINT>[numSparseBufferSize];
        }
        // 点的信息传给originalInputSparse
        InputPointSparse<MAX_RES_PER_POINT> *pc = originalInputSparse;
        numSparsePoints = 0;

        for (auto feat: fr->features) {
            if (feat->point && feat->point->mpPH) {
                auto p = feat->point->mpPH;

                for (int i = 0; i < patternNum; i++)
                {
                    pc[numSparsePoints].color[i] = p->color[i];
                    pc[numSparsePoints].label[i] = p->label[i];
                }
                pc[numSparsePoints].u = p->u;
                pc[numSparsePoints].v = p->v;
                pc[numSparsePoints].idpeth = p->idepth_scaled;
                pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
                pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
                pc[numSparsePoints].numGoodRes = 0;
                pc[numSparsePoints].status = 1;
                numSparsePoints++;
            }
        }

        assert(numSparsePoints <= npoints);

        needRefresh = true;
    }
    /**
     * @brief 从帧中得到状态信息
     ***/
    void KeyFrameDisplay::setFromF(shared_ptr<Frame> fs, shared_ptr<CalibHessian> HCalib) {
        
        // 因为内参在更新
        id = fs->id;
        fx = HCalib->fxl();
        fy = HCalib->fyl();
        cx = HCalib->cxl();
        cy = HCalib->cyl();
        width = wG[0];
        height = hG[0];
        fxi = 1 / fx;
        fyi = 1 / fy;
        cxi = -cx / fx;
        cyi = -cy / fy;
        // 位姿
        camToWorld = fs->getPoseOpti().inverse();
        scale = camToWorld.scale();

        needRefresh = true;
        originFrame = fs;
    }
    /**
     * @brief 画出相机的模型
     * 
     * @param lineWidth 线宽, color 线色 
     * @param sizeFactor 尺度大小, drawOrig 画在原轨迹上
     * 
     ***/
    void KeyFrameDisplay::drawCam(float lineWidth, float *color, float sizeFactor, bool drawOrig) {

        if (width == 0)
            return;
        // 画在原始轨迹上
        if (drawOrig && originFrame)
            camToWorld = Sim3(originFrame->getPose().inverse().matrix());
        // 不画在原始轨迹上, 是全局优化后的
        else if (drawOrig == false)
            camToWorld = originFrame->getPoseOpti().inverse();

        float sz = sizeFactor;

        glPushMatrix();

        Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
        glMultMatrixf((GLfloat *) m.data());

        if (color == 0) {
            glColor3f(1, 0, 0);
        } else
            glColor3f(color[0], color[1], color[2]);

        glLineWidth(lineWidth);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glEnd();
        glPopMatrix();
    }
    /**
     * @brief 当有参数改变时, 更新显示的点云
     * 
     * @param canRefresh  是否满足刷新条件
     * @param scaledTH / absTH / mode / minBS / sparsity  viewer显示设置的参数
     * @param forceRefresh  强制刷新
     *
     ***/
    bool KeyFrameDisplay::refreshPC(
        bool canRefresh, float scaledTH, float absTH, int mode, float minBS,
        int sparsity, bool forceRefresh) {
        // 强制更新
        if (forceRefresh) {
            needRefresh = true;
        } else if (canRefresh) {
            // 有设置的量改变, 则更新
            needRefresh = needRefresh ||
                          my_scaledTH != scaledTH ||
                          my_absTH != absTH ||
                          my_displayMode != mode ||
                          my_minRelBS != minBS ||
                          my_sparsifyFactor != sparsity;
        }

        if (!needRefresh) return false;
        needRefresh = false;

        my_scaledTH = scaledTH;
        my_absTH = absTH;
        my_displayMode = mode;
        my_minRelBS = minBS;
        my_sparsifyFactor = sparsity;

        // if there are no vertices, done!
        if (numSparsePoints == 0)
            return false;

        // make data
        // 申请地图点的坐标和颜色存储
        Vec3f *tmpVertexBuffer = new Vec3f[numSparsePoints * patternNum];
        Vec3b *tmpColorBuffer = new Vec3b[numSparsePoints * patternNum];
        int vertexBufferNumPoints = 0;

        for (int i = 0; i < numSparsePoints; i++) {

            /* display modes (PC_mode):
             * my_displayMode==0 - all pts, color-coded
             * my_displayMode==1 - normal points
             * my_displayMode==2 - active only
             * my_displayMode==3 - nothing
             */
            // LDSO这里点状态都是设置1, 所以这里用处不大
            if (my_displayMode == 1 && originalInputSparse[i].status != 1 &&
                originalInputSparse[i].status != 2)
                continue;
            if (my_displayMode == 2 && originalInputSparse[i].status != 1) continue;
            if (my_displayMode > 2) continue;

            // 处理深度值及其协方差
            if (originalInputSparse[i].idpeth < 0) continue;

            float depth = 1.0f / (originalInputSparse[i].idpeth);
            float depth4 = depth * depth;
            depth4 *= depth4;
            float var = (1.0f / (originalInputSparse[i].idepth_hessian + 0.01));

            if (var * depth4 > my_scaledTH)
                continue;

            if (var > my_absTH)
                continue;
            
            // 点的基线距离
            if (originalInputSparse[i].relObsBaseline < my_minRelBS)
                continue;

            // 点pattern显示的位置和颜色
            for (int pnt = 0; pnt < patternNum; pnt++) {

                if (my_sparsifyFactor > 1 && rand() % my_sparsifyFactor != 0) continue;
                int dx = patternP[pnt][0];
                int dy = patternP[pnt][1];
                int label = originalInputSparse[i].label[pnt];
                
                tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u + dx) * fxi + cxi) * depth;
                tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v + dy) * fyi + cyi) * depth;
                tmpVertexBuffer[vertexBufferNumPoints][2] = depth * (1 + 2 * fxi * (rand() / (float) RAND_MAX - 0.5f));

                // 0时是彩色的? 好像没法设置(只能1-4之间)
                if (my_displayMode == 0) {
                    if (originalInputSparse[i].status == 0) {
                        tmpColorBuffer[vertexBufferNumPoints][0] = 0;
                        tmpColorBuffer[vertexBufferNumPoints][1] = 255;
                        tmpColorBuffer[vertexBufferNumPoints][2] = 255;
                    } else if (originalInputSparse[i].status == 1) {
                        tmpColorBuffer[vertexBufferNumPoints][0] = color_map.getColorByLabel(label).x();
                        tmpColorBuffer[vertexBufferNumPoints][1] = color_map.getColorByLabel(label).y();
                        tmpColorBuffer[vertexBufferNumPoints][2] = color_map.getColorByLabel(label).z();
                    } else if (originalInputSparse[i].status == 2) {
                        tmpColorBuffer[vertexBufferNumPoints][0] = 0;
                        tmpColorBuffer[vertexBufferNumPoints][1] = 0;
                        tmpColorBuffer[vertexBufferNumPoints][2] = 255;
                    } else if (originalInputSparse[i].status == 3) {
                        tmpColorBuffer[vertexBufferNumPoints][0] = 255;
                        tmpColorBuffer[vertexBufferNumPoints][1] = 0;
                        tmpColorBuffer[vertexBufferNumPoints][2] = 0;
                    } else {
                        tmpColorBuffer[vertexBufferNumPoints][0] = 255;
                        tmpColorBuffer[vertexBufferNumPoints][1] = 255;
                        tmpColorBuffer[vertexBufferNumPoints][2] = 255;
                    }

                } else {
                    tmpColorBuffer[vertexBufferNumPoints][0] = originalInputSparse[i].color[pnt];
                    tmpColorBuffer[vertexBufferNumPoints][1] = originalInputSparse[i].color[pnt];
                    tmpColorBuffer[vertexBufferNumPoints][2] = originalInputSparse[i].color[pnt];
                }
                vertexBufferNumPoints++;
                assert(vertexBufferNumPoints <= numSparsePoints * patternNum);
            }
        }

        if (vertexBufferNumPoints == 0) {
            delete[] tmpColorBuffer;
            delete[] tmpVertexBuffer;
            return true;
        }

        // 点的数量扩充到原来的1.3倍就重新初始化
        numGLBufferGoodPoints = vertexBufferNumPoints;
        if (numGLBufferGoodPoints > numGLBufferPoints) {
            numGLBufferPoints = vertexBufferNumPoints * 1.3;
            vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
            colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
        }
        vertexBuffer.Upload(tmpVertexBuffer, sizeof(float) * 3 * numGLBufferGoodPoints, 0);
        colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char) * 3 * numGLBufferGoodPoints, 0);
        bufferValid = true;
        delete[] tmpColorBuffer;
        delete[] tmpVertexBuffer;
        return true;

    }
    /**
     * @brief 画出地图点
     * 
     * @param pointSize 点的大小
     * 
     ***/
    void KeyFrameDisplay::drawPC(float pointSize) {

        if (!bufferValid || numGLBufferGoodPoints == 0)
            return;

        glDisable(GL_LIGHTING);
        glPushMatrix();

        Mat44f m;
        if (originFrame) {
            Sim3 Swc = originFrame->getPoseOpti().inverse();
            m = Swc.matrix().cast<float>();
            scale = Swc.scale();
        } else {
            m = camToWorld.matrix().cast<float>();
        }

        glMultMatrixf((GLfloat *) m.data());
        glPointSize(pointSize);
        colorBuffer.Bind();
        glColorPointer(colorBuffer.count_per_element, colorBuffer.datatype, 0, 0);
        glEnableClientState(GL_COLOR_ARRAY);

        vertexBuffer.Bind();
        glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);
        glDisableClientState(GL_VERTEX_ARRAY);
        vertexBuffer.Unbind();

        glDisableClientState(GL_COLOR_ARRAY);
        colorBuffer.Unbind();

        glPopMatrix();
    }
    /**
     * @brief 保存地图点
     ***/
    void KeyFrameDisplay::save(ofstream &of) {
        Sophus::Sim3f Swc;
        if (originFrame) {
            Swc = originFrame->getPoseOpti().inverse().cast<float>();
        } else {
            Swc = camToWorld.cast<float>();
        }

        for (int i = 0; i < numSparsePoints; ++i) {
            if (originalInputSparse[i].idpeth <= 0) continue;
            float depth = 1.0f / (originalInputSparse[i].idpeth);

            float x = (originalInputSparse[i].u * fxi + cxi) * depth;
            float y = (originalInputSparse[i].v * fyi + cyi) * depth;
            float z = depth;
            Vec3f pw = Swc * Vec3f(x, y, z);
            of << pw[0] << " " << pw[1] << " " << pw[2] << endl;
        }
    }

    // =================================================================================
    /**
     * @brief viewer构造函数，并启动线程
     ***/    
    PangolinDSOViewer::PangolinDSOViewer(int w, int h, bool startRunThread) {

        this->w = w;
        this->h = h;
        running = true;

        unique_lock<mutex> lk(openImagesMutex);
        internalVideoImg = new MinimalImageB3(w, h);
        videoImgChanged = true;

        internalVideoImg->setBlack();
        
        // 运行显示线程
        if (startRunThread)
            runThread = thread(&PangolinDSOViewer::run, this);

        // new一个当前帧显示的类 KeyFrameDisplay
        currentCam = shared_ptr<KeyFrameDisplay>(new KeyFrameDisplay());
    }
    /**
     * @brief 析构函数
     ***/
    PangolinDSOViewer::~PangolinDSOViewer() {
        close();
        if (runThread.joinable()) {
            runThread.join();
        }
    }
    /**
     * @brief DSO的pangolin显示的主循环
     ***/
    void PangolinDSOViewer::run() {

        pangolin::CreateWindowAndBind("Main", 2 * w, 2 * h);
        LOG(INFO) << "Create Pangolin DSO viewer" << endl;
        const int UI_WIDTH = 180;

        glEnable(GL_DEPTH_TEST);

        // 3D visualization
        // 设置pangolin中相机的参数
        pangolin::OpenGlRenderState Visualization3D_camera(
            pangolin::ProjectionMatrix(w, h, 400, 400, w / 2, h / 2, 0.1, 1000),
            // 3个相机位置, 3个观测点, 一个向上的方向
            pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY)
        );

        pangolin::View &Visualization3D_display = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w / (float) h)
            .SetHandler(new pangolin::Handler3D(Visualization3D_camera));

        pangolin::View &d_video = pangolin::Display("imgVideo")
            .SetAspect(w / (float) h);

        pangolin::GlTexture texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

        pangolin::CreateDisplay()
            .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(d_video);

        // parameter reconfigure gui
        // 左侧的交互窗口
        pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
        //int: 默认值，最小值，最大值，false是否以对数
        pangolin::Var<int> settings_pointCloudMode("ui.PC_mode", 0, 0, 4, false);
        //bool： 1.默认  2.true选择框
        pangolin::Var<bool> settings_showKFCameras("ui.KFCam", true, true);
        pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam", true, true);
        pangolin::Var<bool> settings_showTrajectory("ui.Trajectory", true, true);
        pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory", false, true);
        pangolin::Var<bool> settings_showActiveConstraints("ui.ActiveConst", true, true);
        pangolin::Var<bool> settings_showAllConstraints("ui.AllConst", false, true);

        pangolin::Var<bool> settings_show3D("ui.show3D", true, true);
        pangolin::Var<bool> settings_showLiveDepth("ui.showDepth", true, true);
        pangolin::Var<bool> settings_showLiveVideo("ui.showVideo", true, true);
        pangolin::Var<bool> settings_showLiveResidual("ui.showResidual", false, true);

        pangolin::Var<bool> settings_showFramesWindow("ui.showFramesWindow", false, true);
        pangolin::Var<bool> settings_showFullTracking("ui.showFullTracking", false, true);
        pangolin::Var<bool> settings_showCoarseTracking("ui.showCoarseTracking", false, true);

        pangolin::Var<int> settings_sparsity("ui.sparsity", 1, 1, 20, false);
        //double: 默认, 最小, 最大, 是否log显示
        pangolin::Var<double> settings_scaledVarTH("ui.relVarTH", 0.001, 1e-10, 1e10, true);
        pangolin::Var<double> settings_absVarTH("ui.absVarTH", 0.001, 1e-10, 1e10, true);
        pangolin::Var<double> settings_minRelBS("ui.minRelativeBS", 0.1, 0, 1, false);

        pangolin::Var<bool> settings_resetButton("ui.Reset", false, false);

        pangolin::Var<int> settings_nPts("ui.activePoints", setting_desiredPointDensity, 50, 5000, false);
        pangolin::Var<int> settings_nCandidates("ui.pointCandidates", setting_desiredImmatureDensity, 50, 5000, false);
        pangolin::Var<int> settings_nMaxFrames("ui.maxFrames", setting_maxFrames, 4, 10, false);
        pangolin::Var<double> settings_kfFrequency("ui.kfFrequency", setting_kfGlobalWeight, 0.1, 3, false);
        pangolin::Var<double> settings_gradHistAdd("ui.minGradAdd", setting_minGradHistAdd, 0, 15, false);

        pangolin::Var<double> settings_trackFps("ui.Track fps", 0, 0, 0, false);
        pangolin::Var<double> settings_mapFps("ui.KF fps", 0, 0, 0, false);


        // Default hooks for exiting (Esc) and fullscreen (tab).
        LOG(INFO) << "Looping viewer thread" << endl;
        while (!pangolin::ShouldQuit() && running) {
            // Clear entire screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            if (setting_render_display3D) {

                // Activate efficiently by object
                // render the point cloud
                Visualization3D_display.Activate(Visualization3D_camera);
                unique_lock<mutex> lk3d(model3DMutex);
                int refreshed = 0;

                float yellow[3] = {1, 1, 0};

                bool needRefreshAll = false;
                // 读取是否需要刷新
                {
                    unique_lock<mutex> lck(freshMutex);
                    needRefreshAll = freshAll;
                }
                // 如果设置了freshAll, 则刷新地图点
                if (needRefreshAll) {
                    for (auto &fh: keyframes) {
                        fh->refreshPC(true, this->settings_scaledVarTH,
                                      this->settings_absVarTH,
                                      this->settings_pointCloudMode, this->settings_minRelBS,
                                      this->settings_sparsity);
                        fh->drawPC(1);
                    }
                    unique_lock<mutex> lck(freshMutex);
                    freshAll = false;
                } else {
                // 否则前10个关键帧刷新, 这里可能写错了, =+ ??
                    for (auto fh : keyframes) {
                        refreshed = +(int) (fh->refreshPC(refreshed < 10, this->settings_scaledVarTH,
                                                          this->settings_absVarTH,
                                                          this->settings_pointCloudMode, this->settings_minRelBS,
                                                          this->settings_sparsity));
                        fh->drawPC(1);
                    }
                }

                // active key frames
                // 显示滑窗内的关键帧
                if (this->settings_showKFCameras) {
                    float blue[3] = {0, 0, 1};
                    for (auto &id: activeKFIDs) {
                        auto kfd = keyframesByKFID[id];
                        kfd->drawCam(1.5, blue, 0.1);
                    }
                }

                // current cam
                // 画当前关键帧
                if (this->settings_showCurrentCamera && currentCam)
                    currentCam->drawCam(2, 0, 0.2);

                // trajectory
                if (settings_showTrajectory) {

                    // red is before optimization
                    float colorRed[3] = {1, 0, 0};
                    glColor3f(colorRed[0], colorRed[1], colorRed[2]);
                    glLineWidth(3);
                    glBegin(GL_LINE_STRIP);
                    for (unsigned int i = 0; i < allFramePoses.size(); i++) {
                        shared_ptr<Frame> fr = allFramePoses[i];
                        Vec3 t = fr->getPose().inverse().translation();
                        glVertex3d(t[0], t[1], t[2]);
                    }
                    glEnd();

                    // yellow is after optimization
                    glBegin(GL_LINE_STRIP);
                    glColor3f(yellow[0], yellow[1], yellow[2]);
                    for (size_t i = 0; i < keyframes.size(); i++) {
                        shared_ptr<Frame> frame = keyframes[i]->originFrame;
                        if (frame) {
                            Vec3 t = frame->getPoseOpti().inverse().translation();
                            glVertex3d(t[0], t[1], t[2]);
                        }
                    }
                    glEnd();

                    // draw pose graph
                    glLineWidth(3.0);
                    glColor3f(yellow[0], yellow[1], yellow[2]);
                    glBegin(GL_LINES);
                    for (size_t i = 0; i < keyframes.size(); i++) {
                        shared_ptr<Frame> frame = keyframes[i]->originFrame;
                        if (frame) {
                            Vec3 t = frame->getPoseOpti().inverse().translation();
                            unique_lock<mutex> lck(frame->mutexPoseRel);
                            for (auto rel: frame->poseRel) {
                                auto t2 = rel.first->getPoseOpti().inverse().translation();
                                glVertex3d(t[0], t[1], t[2]);
                                glVertex3d(t2[0], t2[1], t2[2]);
                            }
                        }
                    }
                    glEnd();
                }

                lk3d.unlock();
            }
            // 计算帧率(10帧的平均值)
            {
                model3DMutex.lock();
                float sd = 0;
                for (float d : lastNTrackingMs) sd += d;
                settings_trackFps = lastNTrackingMs.size() * 1000.0f / sd;
                model3DMutex.unlock();
            }

            // video image
            // 来新帧, 则显示图像, 除此之外DSO还有显示残差, 逆深度的图像可以设置
            {
                unique_lock<mutex> lck(openImagesMutex);
                if (videoImgChanged) texVideo.Upload(internalVideoImg->data, GL_BGR, GL_UNSIGNED_BYTE);
            }

            if (setting_render_displayVideo) {
                d_video.Activate();
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                texVideo.RenderToViewportFlipY();
            }

            // update parameters
            this->settings_pointCloudMode = settings_pointCloudMode.Get();

            this->settings_showActiveConstraints = settings_showActiveConstraints.Get();
            this->settings_showAllConstraints = settings_showAllConstraints.Get();
            this->settings_showCurrentCamera = settings_showCurrentCamera.Get();
            this->settings_showKFCameras = settings_showKFCameras.Get();
            this->settings_showTrajectory = settings_showTrajectory.Get();
            this->settings_showFullTrajectory = settings_showFullTrajectory.Get();

            setting_render_display3D = settings_show3D.Get();
            setting_render_displayDepth = settings_showLiveDepth.Get();
            setting_render_displayVideo = settings_showLiveVideo.Get();
            setting_render_displayResidual = settings_showLiveResidual.Get();

            setting_render_renderWindowFrames = settings_showFramesWindow.Get();
            setting_render_plotTrackingFull = settings_showFullTracking.Get();
            setting_render_displayCoarseTrackingFull = settings_showCoarseTracking.Get();

            this->settings_absVarTH = settings_absVarTH.Get();
            this->settings_scaledVarTH = settings_scaledVarTH.Get();
            this->settings_minRelBS = settings_minRelBS.Get();
            this->settings_sparsity = settings_sparsity.Get();

            setting_desiredPointDensity = settings_nPts.Get();
            setting_desiredImmatureDensity = settings_nCandidates.Get();
            setting_maxFrames = settings_nMaxFrames.Get();
            setting_kfGlobalWeight = settings_kfFrequency.Get();
            setting_minGradHistAdd = settings_gradHistAdd.Get();

            if (settings_resetButton.Get()) {
                printf("RESET!\n");
                settings_resetButton.Reset();
                setting_fullResetRequested = true;
            }

            // Swap frames and Process Events
            pangolin::FinishFrame();

            if (needReset)
                reset_internal();

            usleep(5000);
        }

        printf("QUIT Pangolin thread!\n");
        printf("So Long, and Thanks for All the Fish!\n");
    }

    void PangolinDSOViewer::close() {
        running = false;
    }

    void PangolinDSOViewer::join() {
        runThread.join();
        printf("JOINED Pangolin thread!\n");
    }

    void PangolinDSOViewer::reset() {
        needReset = true;
    }
    /**
     * @brief 向viewer中添加关键帧信息(滑窗内优化会改变)
     ***/
    void PangolinDSOViewer::publishKeyframes(
        std::vector<shared_ptr<Frame>> &frames, bool final,
        shared_ptr<CalibHessian> HCalib) {

        if (!setting_render_display3D) return;
        if (disableAllDisplay) return;

        unique_lock<mutex> lk(model3DMutex);
        activeKFIDs.clear();
        for (auto fr :frames) {
            // 添加新的
            if (keyframesByKFID.find(fr->kfId) == keyframesByKFID.end()) {
                shared_ptr<KeyFrameDisplay> kfd = shared_ptr<KeyFrameDisplay>(new KeyFrameDisplay());
                keyframesByKFID[fr->kfId] = kfd;
                keyframes.push_back(kfd);
            } else {
            }
            // 有hessian则从framehessian得到信息
            if (fr->frameHessian)
                keyframesByKFID[fr->kfId]->setFromKF(fr->frameHessian, HCalib);
            // 没有(marg掉了), 因为闭环优化所以之前的位姿会改变, 所以比DSO多这个
            else
                keyframesByKFID[fr->kfId]->setFromF(fr, HCalib);
            activeKFIDs.push_back(fr->kfId);
        }

    }
    /**
     * @brief 重置
     ***/
    void PangolinDSOViewer::reset_internal() {

        LOG(INFO) << "resetting viewer" << endl;
        unique_lock<mutex> lock(model3DMutex);
        internalVideoImg->setBlack();
        keyframes.clear();
        keyframesByKFID.clear();
        allFramePoses.clear();
        activeKFIDs.clear();
        needReset = false;
    }
    /**
     * @brief 设置当前帧(有可能是kf, 或普通frame), 位姿和图像
     ***/
    void PangolinDSOViewer::publishCamPose(shared_ptr<Frame> frame, shared_ptr<CalibHessian> HCalib) {

        if (!setting_render_display3D)
            return;
        if (disableAllDisplay)
            return;

        // 计算普通帧帧率
        unique_lock<mutex> lk(model3DMutex);
        struct timeval time_now;
        gettimeofday(&time_now, NULL);
        lastNTrackingMs.push_back(
            ((time_now.tv_sec - last_track.tv_sec) * 1000.0f + (time_now.tv_usec - last_track.tv_usec) / 1000.0f));
        if (lastNTrackingMs.size() > 10)
            lastNTrackingMs.pop_front();

        last_track = time_now;
        // 当前帧位姿
        if (currentCam)
            currentCam->setFromF(frame, HCalib);
        allFramePoses.push_back(frame);

        // 显示图像
        if (frame->frameHessian) {
            unique_lock<mutex> lk(openImagesMutex);
            for (int i = 0; i < w * h; i++)
                internalVideoImg->data[i][0] =
                internalVideoImg->data[i][1] =
                internalVideoImg->data[i][2] =
                    frame->frameHessian->dI[i][0] * 0.8 > 255.0f ? 255.0 : frame->frameHessian->dI[i][0] * 0.8;
            videoImgChanged = true;
        }
    }
    /**
     * @brief 把点云保存成ply文件
     ***/
    void PangolinDSOViewer::saveAsPLYFile(const string &file_name) {
        LOG(INFO) << "save to " << file_name;
        ofstream fout(file_name);
        if (!fout) return;
        unique_lock<mutex> lk3d(model3DMutex);

        // count number of landmarks
        int cnt_points = 0;

        for (auto kf: keyframes) {
            cnt_points += kf->numPoints();
        }
        // header
        fout << "ply" << endl << "format ascii 1.0" << endl
             << "element vertex " << cnt_points << endl
             << "property float x" << endl
             << "property float y" << endl
             << "property float z" << endl
             << "end_header" << endl;

        for (auto kf: keyframes) {
            kf->save(fout);
        }
        fout.close();
        cout << "ply file is save to " << file_name << endl;
    }

}
