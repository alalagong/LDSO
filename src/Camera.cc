#include "Camera.h"
#include "internal/CalibHessian.h"

using namespace ldso::internal;

namespace ldso {
    Camera::Camera( double fx, double fy, double cx, double cy) {
        this->fx = fx;
        this->fy = fy;
        this->cx = cx;
        this->cy = cy;
    }
    /**
     * @brief 创建CalibHessian
     ***/
    void Camera::CreateCH(shared_ptr<Camera> cam) {
        this->mpCH = shared_ptr<CalibHessian>( new CalibHessian(cam) );
    }
    /**
     * @brief 释放这个函数的CalibHessian, 互相指针置null
     ***/
    void Camera::ReleaseCH() {
        if ( mpCH ) {
            mpCH->camera = nullptr;
            mpCH = nullptr;
        }
    }

}