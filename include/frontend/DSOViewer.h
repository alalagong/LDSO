#pragma once
#ifndef LDSO_VIEWER_H_
#define LDSO_VIEWER_H_

#include "NumTypes.h"
#include "Frame.h"
#include "Map.h"
#include "frontend/MinimalImage.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/CalibHessian.h"

#include <thread>
#include <mutex>
#include <pangolin/pangolin.h>
#include <Eigen/Core>


using namespace std;

using namespace ldso::internal;

namespace ldso {

struct LabelColorMap
{
    LabelColorMap()
    {
        label_color_map << 128, 64,128,   // wall
                           244, 35,232,   // floor 
                            70, 70, 70,   // cabinet
                            102,102,156,   // bed
                            190,153,153,   // chair
                            153,153,153,   // sofa
                            
                            250,170, 30,  // table
                            220,220,  0,  // door
                            107,142, 35,  // window
                            152,251,152,  // bookshelf
                             70,130,180,  // picture
                            
                            220, 20, 60,  // counter
                            255,  0,  0,  // blinds
                            0,  0, 142,  // desk
                             0,  0, 70,  // shelves
                             0, 60,100,  // curtain

                             0, 80,100,   // dresser
                             0,  0,230,   // pillow
                            119, 11, 32,   // mirror
                            0,  0,  0;  // floor mat
    }

    Eigen::Vector3i getColorByLabel(int label)
    {
        if(label < 0 || label > 19)
        {
            std::cout << "[ERROR]: out of label range is " << label << endl;;
            return Eigen::Vector3i();
        }
        return label_color_map.row(label);
    }

    Eigen::Matrix<int, 20, 3> label_color_map;
};

    //  Visualization for DSO

    /**
     * Point cloud struct
     */
    template<int ppp>
    struct InputPointSparse {
        float u = 0;
        float v = 0;
        float idpeth = 0;
        float idepth_hessian = 0;
        float relObsBaseline = 0;
        int numGoodRes = 0;
        unsigned char color[ppp];
        unsigned char label[ppp];
        unsigned char status = 0;
    };

    // stores a point cloud associated to a Keyframe.
    class KeyFrameDisplay {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        ~KeyFrameDisplay() {
            if (originalInputSparse)
                delete[] originalInputSparse;
        }

        // copies points from KF over to internal buffer,
        // keeping some additional information so we can render it differently.
        void setFromKF(shared_ptr<FrameHessian> fh, shared_ptr<CalibHessian> HCalib);

        // copies points from KF over to internal buffer,
        // keeping some additional information so we can render it differently.
        void setFromF(shared_ptr<Frame> fs, shared_ptr<CalibHessian> HCalib);

        // copies & filters internal data to GL buffer for rendering. if nothing to do: does nothing.
        bool refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity,
                       bool forceRefresh = false);

        // renders cam & pointcloud.
        void drawCam(float lineWidth = 1, float *color = 0, float sizeFactor = 1, bool drawOrig = true);

        void drawPC(float pointSize);

        int id = 0;
        bool active = true;
        Sim3 camToWorld = Sim3();

        inline bool operator<(const KeyFrameDisplay &other) const {
            return (id < other.id);
        }

        shared_ptr<Frame> originFrame = nullptr;

        int numPoints() const {
            int cnt = 0;
            for (int i = 0; i < numSparsePoints; ++i) {
                if (originalInputSparse[i].idpeth > 0) cnt++;
            }
            return cnt;
        }

        void save(ofstream &of);

    private:
        float fx, fy, cx, cy;
        float fxi, fyi, cxi, cyi;
        double scale = 1.0;
        int width, height;

        float my_scaledTH = 1e10, my_absTH = 1e10, my_scale = 0;
        int my_sparsifyFactor = 1;
        int my_displayMode = 0;
        float my_minRelBS = 0;
        bool needRefresh = true;

        // 地图点个数
        int numSparsePoints = 0;
        int numSparseBufferSize = 0;
        // DSO生成的地图点相关数据
        InputPointSparse<MAX_RES_PER_POINT> *originalInputSparse;

        bool bufferValid = 0;
        int numGLBufferPoints = 0;
        int numGLBufferGoodPoints = 0;
        pangolin::GlBuffer vertexBuffer;
        pangolin::GlBuffer colorBuffer;

        LabelColorMap color_map;

    };

    /**
     * viewer implemented by pangolin
     */
    class PangolinDSOViewer {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        PangolinDSOViewer(int w, int h, bool startRunThread = true);

        ~PangolinDSOViewer();

        void run();

        void close();

        void publishKeyframes(std::vector<shared_ptr<Frame>> &frames, bool final, shared_ptr<CalibHessian> HCalib);

        void publishCamPose(shared_ptr<Frame> frame, shared_ptr<CalibHessian> HCalib);

        void setMap(shared_ptr<Map> m) {
            globalMap = m;
        }

        // void pushLiveFrame( shared_ptr<Frame> image);

        /* call on finish */
        void join();

        /* call on reset */
        void reset();

        void refreshAll() {
            unique_lock<mutex> lck(freshMutex);
            LOG(INFO) << "set gui refresh!" << endl;
            freshAll = true;
        }

        void saveAsPLYFile(const string &file_name);

    private:

        bool needReset = false;

        void reset_internal();

        // void drawConstraints();

        thread runThread;
        bool running = true;
        int w, h;

        // images rendering
        mutex openImagesMutex;
        MinimalImageB3 *internalVideoImg = nullptr;
        bool videoImgChanged = true;

        // 3D model rendering
        mutex model3DMutex;
        std::vector<shared_ptr<Frame>> allFramePoses;  // trajectory

        shared_ptr<KeyFrameDisplay> currentCam = nullptr;
        std::vector<shared_ptr<KeyFrameDisplay>> keyframes; // all keyframes
        std::map<int, shared_ptr<KeyFrameDisplay>> keyframesByKFID;
        std::vector<size_t> activeKFIDs;    // active keyframes's IDs

        // render settings
        bool settings_showKFCameras = true;
        bool settings_showCurrentCamera = true;
        bool settings_showTrajectory = true;
        bool settings_showFullTrajectory;
        bool settings_showActiveConstraints;
        bool settings_showAllConstraints;

        float settings_scaledVarTH;
        float settings_absVarTH;
        int settings_pointCloudMode;
        float settings_minRelBS;
        int settings_sparsity;

        // timings
        struct timeval last_track;
        struct timeval last_map;

        bool freshAll = false;
        mutex freshMutex;

        std::deque<float> lastNTrackingMs;
        std::deque<float> lastNMappingMs;

        shared_ptr<Map> globalMap = nullptr;
    };

}

#endif // LDSO_VIEWER_H_
