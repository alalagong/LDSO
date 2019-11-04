#include "Feature.h"
#include "Point.h"
#include "internal/ImmaturePoint.h"
#include "internal/PointHessian.h"

#include <memory>

using namespace std;

using namespace ldso::internal;

namespace ldso {

    /**
     * @brief  从未成熟点创建一个特征点、地图点 
     ***/
    void Feature::CreateFromImmature() {
        if (point) {
            LOG(WARNING) << "Map point already created! You cannot create twice! " << endl;
            return;
        }
        assert(ip != nullptr);

        point = shared_ptr<Point>(new Point(ip->feature));
        point->mpPH->point = point;   // set the point hessians backward pointer
        status = Feature::FeatureStatus::VALID;
    }
    /**
     * @brief  删除未成熟点
     ***/
    void Feature::ReleaseImmature() {
        if (ip) {
            ip->feature = nullptr;
            ip = nullptr;
        }
    }
    /**
     * @brief  删除地图点
     ***/
    void Feature::ReleaseMapPoint() {
        if (point) {
            point->ReleasePH();
        }
    }
    /**
     * @brief  存储地图需要的信息
     ***/
    void Feature::save(ofstream &fout) {
        fout.write((char *) &status, sizeof(status));
        fout.write((char *) &uv[0], sizeof(float));
        fout.write((char *) &uv[1], sizeof(float));
        fout.write((char *) &invD, sizeof(float));
        fout.write((char *) &isCorner, sizeof(bool));
        fout.write((char *) &angle, sizeof(float));
        fout.write((char *) &score, sizeof(float));
        fout.write((char *) descriptor, sizeof(uchar) * 32);
        if (point && status == Feature::FeatureStatus::VALID)
            point->save(fout);
    }
    /**
     * @brief  加载地图信息
     ***/
    void Feature::load(ifstream &fin, vector<shared_ptr<Frame>> &allKFs) {

        fin.read((char *) &status, sizeof(status));
        fin.read((char *) &uv[0], sizeof(float));
        fin.read((char *) &uv[1], sizeof(float));
        fin.read((char *) &invD, sizeof(float));
        fin.read((char *) &isCorner, sizeof(bool));
        fin.read((char *) &angle, sizeof(float));
        fin.read((char *) &score, sizeof(float));
        fin.read((char *) descriptor, sizeof(uchar) * 32);

        if (status == Feature::FeatureStatus::VALID) {
            point = shared_ptr<Point>(new Point);
            point->load(fin, allKFs);
        }
    }

}