/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/
/*--------------------------------------------------------------------------------------------
 * 系统以初始化成功的第一帧为世界坐标原点，且是以相机为主导，非imu
 * 计算时都是通过相机的位姿，涉及到imu位姿时利用当前相机与imu标定关系计算
 * 
 */

#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Initializer.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "PnPsolver.h"

#include <iostream>

#include <mutex>
#include <chrono>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <include/MLPnPsolver.h>

using namespace std;

namespace ORB_SLAM3
{
/* mState变化历程：最开始Tracking刚开启，或者CreateMapInAtlas，或者Reset，或者ResetActiveMap后变成NO_IMAGES_YET
*               在上面状态时（可以理解为第一帧或重置后第一帧）创建了地图后变成NOT_INITIALIZED，后面的变化在下面代码中有详细介绍
*               其中OK_KLT与SYSTEM_NOT_READY没出现过，应该是调试用的量
*/
Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase *pKFDB,
                    const string &strSettingPath, const int sensor, const string &_nameSeq) : 
                    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
                    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
                    mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys), mpViewer(NULL),
                    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
                    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr)
{
    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    bool b_parse_cam = ParseCamParamFile(fSettings);
    if (!b_parse_cam)
    {
        std::cout << "*Error with the camera parameters in the config file*" << std::endl;
    }

    // Load ORB parameters
    bool b_parse_orb = ParseORBParamFile(fSettings);
    if (!b_parse_orb)
    {
        std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
    }

    initID = 0;
    lastID = 0;

    // Load IMU parameters
    bool b_parse_imu = true;
    if (sensor == System::IMU_MONOCULAR || sensor == System::IMU_STEREO)
    {
        b_parse_imu = ParseIMUParamFile(fSettings);
        if (!b_parse_imu)
        {
            std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
        }

        mnFramesToResetIMU = mMaxFrames;
    }

    mbInitWith3KFs = false;

    //Rectification parameters
    /*mbNeedRectify = false;
if((mSensor == System::STEREO || mSensor == System::IMU_STEREO) && sCameraName == "PinHole")
{
    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fSettings["LEFT.K"] >> K_l;
    fSettings["RIGHT.K"] >> K_r;

    fSettings["LEFT.P"] >> P_l;
    fSettings["RIGHT.P"] >> P_r;

    fSettings["LEFT.R"] >> R_l;
    fSettings["RIGHT.R"] >> R_r;

    fSettings["LEFT.D"] >> D_l;
    fSettings["RIGHT.D"] >> D_r;

    int rows_l = fSettings["LEFT.height"];
    int cols_l = fSettings["LEFT.width"];
    int rows_r = fSettings["RIGHT.height"];
    int cols_r = fSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty()
            || rows_l==0 || cols_l==0 || rows_r==0 || cols_r==0)
    {
        mbNeedRectify = false;
    }
    else
    {
        mbNeedRectify = true;
        // M1r y M2r son los outputs (igual para l)
        // M1r y M2r son las matrices relativas al mapeo correspondiente a la rectificación de mapa en el eje X e Y respectivamente
        //cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
        //cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
    }


}
else
{
    int cols = 752;
    int rows = 480;
    cv::Mat R_l = cv::Mat::eye(3, 3, CV_32F);
}*/

    mnNumDataset = 0;

    if (!b_parse_cam || !b_parse_orb || !b_parse_imu)
    {
        std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
        try
        {
            throw -1;
        }
        catch (exception &e)
        {
        }
    }

    //f_track_stats.open("tracking_stats"+ _nameSeq + ".txt");
    /*f_track_stats.open("tracking_stats.txt");
f_track_stats << "# timestamp, Num KF local, Num MP local, time" << endl;
f_track_stats << fixed ;*/

#ifdef SAVE_TIMES
    f_track_times.open("tracking_times.txt");
    f_track_times << "# ORB_Ext(ms), Stereo matching(ms), Preintegrate_IMU(ms), Pose pred(ms), LocalMap_track(ms), NewKF_dec(ms)" << endl;
    f_track_times << fixed;
#endif
}

Tracking::~Tracking()
{
    //f_track_stats.close();
#ifdef SAVE_TIMES
    f_track_times.close();
#endif
}

/** 
 * @brief 读取相机参数
 * @param fSettings 参数文件
 */
bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if (sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if (!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if (!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if (!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if (!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if (!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if (!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if (!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if (!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if (!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        if (b_miss_params)
        {
            return false;
        }

        vector<float> vCamCalib{fx, fy, cx, cy};

        mpCamera = new Pinhole(vCamCalib);

        mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;

        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if (mDistCoef.rows == 5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3, 3, CV_32F);
        mK.at<float>(0, 0) = fx;
        mK.at<float>(1, 1) = fy;
        mK.at<float>(0, 2) = cx;
        mK.at<float>(1, 2) = cy;
    }
    else if (sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if (!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if (!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if (!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if (!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if (!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if (!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if (!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if (!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        if (!b_miss_params)
        {
            vector<float> vCamCalib{fx, fy, cx, cy, k1, k2, k3, k4};
            mpCamera = new KannalaBrandt8(vCamCalib);

            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3, 3, CV_32F);
            mK.at<float>(0, 0) = fx;
            mK.at<float>(1, 1) = fy;
            mK.at<float>(0, 2) = cx;
            mK.at<float>(1, 2) = cy;
        }

        if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
        {
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if (!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if (!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if (!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if (!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if (!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if (!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if (!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if (!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if (!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if (!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if (!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if (!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            if (!node.empty())
            {
                mTlr = node.mat();
                if (mTlr.rows != 3 || mTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if (!b_miss_params)
            {
                static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx, fy, cx, cy, k1, k2, k3, k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);

                static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl
                            << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n"
                            << mTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if (b_miss_params)
        {
            return false;
        }

        mpAtlas->AddCamera(mpCamera);
        mpAtlas->AddCamera(mpCamera2);
    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if (!node.empty() && node.isReal())
        {
            mbf = node.real();
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
    }

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if (!node.empty() && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf * mThDepth / fx;
            cout << endl
                    << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
    }

    if (mSensor == System::RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if (!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
    }

    if (b_miss_params)
    {
        return false;
    }

    return true;
}

/** 
 * @brief 读取ORB特征点参数
 * @param fSettings 参数文件
 */
bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if (!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if (!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if (!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if (!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if (!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if (b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl
            << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

/** 
 * @brief 读取IMU参数
 * @param fSettings 参数文件
 */
bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat Tbc;
    cv::FileNode node = fSettings["Tbc"];
    if (!node.empty())
    {
        Tbc = node.mat();
        if (Tbc.rows != 4 || Tbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }

    cout << endl;

    cout << "Left camera to Imu Transform (Tbc): " << endl
            << Tbc << endl;

    float freq, Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if (!node.empty() && node.isInt())
    {
        freq = node.operator int();
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if (!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if (!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if (!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if (!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    if (b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(freq);
    cout << endl;
    cout << "IMU frequency: " << freq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;
    // 设置imu标定参数
    mpImuCalib = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);

    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

/** 
 * @brief 处理双目数据，创建frame与track
 * @param imRectLeft 左相机
 * @param imRectRight 右相机
 * @param timestamp 时间
 * @param filename 文件名字
 * @return 当前帧位姿
 */
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
        }
    }

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
    else if (mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr);
    else if (mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);
    else if (mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr, &mLastFrame, *mpImuCalib);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    Track();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    double t_track = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    /*cout << "trracking time: " << t_track << endl;
f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ", ";
f_track_stats << mvpLocalKeyFrames.size() << ", ";
f_track_stats << mvpLocalMapPoints.size() << ", ";
f_track_stats << setprecision(6) << t_track << endl;*/

#ifdef SAVE_TIMES
    f_track_times << mCurrentFrame.mTimeORB_Ext << ", ";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ", ";
    f_track_times << mTime_PreIntIMU << ", ";
    f_track_times << mTime_PosePred << ", ";
    f_track_times << mTime_LocalMapTrack << ", ";
    f_track_times << mTime_NewKF_Dec << ", ";
    f_track_times << t_track << endl;
#endif

    return mCurrentFrame.mTcw.clone();
}

/** 
 * @brief 处理RGBD数据，创建frame与track
 * @param imRGB 彩色图
 * @param imD 深度图
 * @param timestamp 时间
 * @param filename 文件名字
 * @return 当前帧位姿
 */
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    Track();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    double t_track = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    /*f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ", ";
f_track_stats << mvpLocalKeyFrames.size() << ", ";
f_track_stats << mvpLocalMapPoints.size() << ", ";
f_track_stats << setprecision(6) << t_track << endl;*/

#ifdef SAVE_TIMES
    f_track_times << mCurrentFrame.mTimeORB_Ext << ", ";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ", ";
    f_track_times << mTime_PreIntIMU << ", ";
    f_track_times << mTime_PosePred << ", ";
    f_track_times << mTime_LocalMapTrack << ", ";
    f_track_times << mTime_NewKF_Dec << ", ";
    f_track_times << t_track << endl;
#endif

    return mCurrentFrame.mTcw.clone();
}

/** 
 * @brief 处理单目数据，创建frame与track
 * @param im 彩色图
 * @param timestamp 时间
 * @param filename 文件名字
 * @return 当前帧位姿
 */
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGray = im;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }
    // 1.根据当前的图片构建一个当前帧
    if (mSensor == System::MONOCULAR)
    {
        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET || (lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
        else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
    }
    else if (mSensor == System::IMU_MONOCULAR)
    {
        // 唯一区别就是初始化时使用的是5倍的特征点数
        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
    }

    if (mState == NO_IMAGES_YET)
        t0 = timestamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    lastID = mCurrentFrame.mnId;
    // 2.跟踪
    Track();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    double t_track = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    /*f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ", ";
f_track_stats << mvpLocalKeyFrames.size() << ", ";
f_track_stats << mvpLocalMapPoints.size() << ", ";
f_track_stats << setprecision(6) << t_track << endl;*/

#ifdef SAVE_TIMES
    f_track_times << mCurrentFrame.mTimeORB_Ext << ", ";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ", ";
    f_track_times << mTime_PreIntIMU << ", ";
    f_track_times << mTime_PosePred << ", ";
    f_track_times << mTime_LocalMapTrack << ", ";
    f_track_times << mTime_NewKF_Dec << ", ";
    f_track_times << t_track << endl;
#endif

    return mCurrentFrame.mTcw.clone();
}

/** 
 * @brief 添加imu数据
 * @param imuMeasurement imu数据
 */
void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

/** 
 * @brief IMU预积分，需要两帧
 */
void Tracking::PreintegrateIMU()
{
    //cout << "start preintegration" << endl;

    // 如果没有上一帧还做个毛预积分啊
    if (!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    // cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    // 没有imu数据，返回
    if (mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }
    // 取出当前帧与上一帧之间这段时间内的imu数据
    while (true)
    {
        bool bSleep = false;
        {
            // mlQueueImuData这个变量好像就当前线程在用，需要上锁？
            unique_lock<mutex> lock(mMutexImuQueue);
            if (!mlQueueImuData.empty())
            {
                IMU::Point *m = &mlQueueImuData.front();
                cout.precision(17);
                // imu数据比上一帧时间-0.001s还少的话就不要了，离上一帧较远
                if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - 0.001l)
                {
                    mlQueueImuData.pop_front();
                }
                else if (m->t < mCurrentFrame.mTimeStamp - 0.001l) // imu数据比当前帧时间-0.001s还少的话添加到mvImuFromLastFrame，并且删除mlQueueImuData
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m); // 除了符合上述条件外的imu信息再添加一个，之后break。
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        // 这段话是不是多余，根本执行不到usleep
        if (bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size() - 1;
    // 使用上一帧的偏置跟当前帧的标定信息初始化一个预积分类
    IMU::Preintegrated *pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias, mCurrentFrame.mImuCalib);

    for (int i = 0; i < n; i++)
    {
        float tstep;
        cv::Point3f acc, angVel;
        // 第一个imu点，一般来说第一个点都差不多符合i<(n-1)，除非n=1
        if ((i == 0) && (i < (n - 1)))
        {
            // 获取相邻两段imu的时间间隔
            float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            // 获取当前imu到上一帧的时间间隔
            float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
            // 设当前时刻imu的加速度a0，下一时刻加速度a1，时间间隔tab 为t10，tini t0p
            // 正常情况下时为了求上一帧到当前时刻imu的一个平均加速度，但是imu时间不会正好落在上一帧的时刻，需要做补偿，要求得a0时刻到上一帧这段时间加速度的改变量
            // 有了这个改变量将其加到a0上之后就可以表示上一帧时的加速度了。其中a0 - (a1-a0)*(tini/tab) 为上一帧时刻的加速度再加上a1 之后除以2就为这段时间的加速度平均值
            // 其中tstep表示a1到上一帧的时间间隔，a0 - (a1-a0)*(tini/tab)这个式子中tini可以是正也可以是负表示时间上的先后，(a1-a0)也是一样，多种情况下这个式子依然成立
            acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a - (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) * 0.5f;
            // 计算过程类似加速度
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                        (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) *
                        0.5f;
            tstep = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if (i < (n - 1))
        {
            // 中间的数据不存在帧的干扰，正常计算
            acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f;
            tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
        }
        else if ((i > 0) && (i == (n - 1))) // 直到倒数第二个imu时刻时，计算过程跟第一时刻类似，都需要考虑帧与imu时刻的关系
        {
            float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
                    (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) *
                    0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                        (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) *
                        0.5f;
            tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
        }
        else if ((i == 0) && (i == (n - 1))) // 就两个数据时使用第一个时刻的，这种情况应该没有吧，，回头应该试试看
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        // 应该是必存在的吧，一个是相对上一关键帧，一个是相对上一帧
        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);  // 从上一个关键帧一直积分到当前帧
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;  // 第一次赋值
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;  // 第一次赋值
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame; // 此时mpLastKeyFrame还表示上一个关键帧

    mCurrentFrame.setIntegrated();

    Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}

/** 
 * @brief 通过IMU预测状态
 * 两个地方用到：
 * 1. 匀速模型计算速度,但并没有给当前帧位姿赋值；
 * 2. 跟踪丢失时不直接判定丢失，通过这个函数预测当前帧位姿看看能不能拽回来，代替纯视觉中的重定位
 * @return 通过imu预测当前帧的状态成功与否
 */
bool Tracking::PredictStateIMU()
{
    if (!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }
    // 总结下都在什么时候地图更新，也就是mbMapUpdated为true
    // 1. 回环或融合
    // 2. 局部地图LocalBundleAdjustment
    // 3. IMU三阶段的初始化
    // 下面的代码流程一模一样，只不过计算时相对的帧不同，地图有更新后相对于上一关键帧做的，反之相对于上一帧
    // 地图更新后会更新关键帧与MP，所以相对于关键帧更准
    // 而没更新的话，距离上一帧更近，计算起来误差更小
    if (mbMapUpdated && mpLastKeyFrame)
    {
        // 获取帧的三信息
        const cv::Mat twb1 = mpLastKeyFrame->GetImuPosition();
        const cv::Mat Rwb1 = mpLastKeyFrame->GetImuRotation();
        const cv::Mat Vwb1 = mpLastKeyFrame->GetVelocity();

        const cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;
        // 更新信息
        cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        cv::Mat twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        cv::Mat Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);
        mCurrentFrame.mPredRwb = Rwb2.clone();
        mCurrentFrame.mPredtwb = twb2.clone();
        mCurrentFrame.mPredVwb = Vwb2.clone();
        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if (!mbMapUpdated)
    {
        const cv::Mat twb1 = mLastFrame.GetImuPosition();
        const cv::Mat Rwb1 = mLastFrame.GetImuRotation();
        const cv::Mat Vwb1 = mLastFrame.mVw;

        const cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        cv::Mat twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        cv::Mat Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);
        mCurrentFrame.mPredRwb = Rwb2.clone();
        mCurrentFrame.mPredtwb = twb2.clone();
        mCurrentFrame.mPredVwb = Vwb2.clone();
        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}

/** 
 * @brief 利用高斯牛顿一次迭代
 * 
 */
void Tracking::ComputeGyroBias(const vector<Frame *> &vpFs, float &bwx, float &bwy, float &bwz)
{
    const int N = vpFs.size();
    vector<float> vbx;
    vbx.reserve(N);
    vector<float> vby;
    vby.reserve(N);
    vector<float> vbz;
    vbz.reserve(N);

    cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
    cv::Mat grad = cv::Mat::zeros(3, 1, CV_32F);

    for (int i = 1; i < N; i++)
    {
        Frame *pF2 = vpFs[i];
        Frame *pF1 = vpFs[i - 1];
        cv::Mat VisionR = pF1->GetImuRotation().t() * pF2->GetImuRotation();
        cv::Mat JRg = pF2->mpImuPreintegratedFrame->JRg;
        cv::Mat E = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaRotation().t() * VisionR;
        cv::Mat e = IMU::LogSO3(E);
        assert(fabs(pF2->mTimeStamp - pF1->mTimeStamp - pF2->mpImuPreintegratedFrame->dT) < 0.01);

        cv::Mat J = -IMU::InverseRightJacobianSO3(e) * E.t() * JRg;
        grad += J.t() * e;
        H += J.t() * J;
    }

    cv::Mat bg = -H.inv(cv::DECOMP_SVD) * grad;
    bwx = bg.at<float>(0);
    bwy = bg.at<float>(1);
    bwz = bg.at<float>(2);

    for (int i = 1; i < N; i++)
    {
        Frame *pF = vpFs[i];
        pF->mImuBias.bwx = bwx;
        pF->mImuBias.bwy = bwy;
        pF->mImuBias.bwz = bwz;
        pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
        pF->mpImuPreintegratedFrame->Reintegrate();
    }
}

/** 
 * @brief 没用到，暂时不看，跟加速度计偏置相关，不过值得一看
 */
void Tracking::ComputeVelocitiesAccBias(const vector<Frame *> &vpFs, float &bax, float &bay, float &baz)
{
    const int N = vpFs.size();
    const int nVar = 3 * N + 3; // 3 velocities/frame + acc bias
    const int nEqs = 6 * (N - 1);

    cv::Mat J(nEqs, nVar, CV_32F, cv::Scalar(0));
    cv::Mat e(nEqs, 1, CV_32F, cv::Scalar(0));
    cv::Mat g = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);

    for (int i = 0; i < N - 1; i++)
    {
        Frame *pF2 = vpFs[i + 1];
        Frame *pF1 = vpFs[i];
        cv::Mat twb1 = pF1->GetImuPosition();
        cv::Mat twb2 = pF2->GetImuPosition();
        cv::Mat Rwb1 = pF1->GetImuRotation();
        cv::Mat dP12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaPosition();
        cv::Mat dV12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaVelocity();
        cv::Mat JP12 = pF2->mpImuPreintegratedFrame->JPa;
        cv::Mat JV12 = pF2->mpImuPreintegratedFrame->JVa;
        float t12 = pF2->mpImuPreintegratedFrame->dT;
        // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
        J.rowRange(6 * i, 6 * i + 3).colRange(3 * i, 3 * i + 3) += cv::Mat::eye(3, 3, CV_32F) * t12;
        J.rowRange(6 * i, 6 * i + 3).colRange(3 * N, 3 * N + 3) += Rwb1 * JP12;
        e.rowRange(6 * i, 6 * i + 3) = twb2 - twb1 - 0.5f * g * t12 * t12 - Rwb1 * dP12;
        // Velocity v2=v1+g*t+R1*dV12
        J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * i, 3 * i + 3) += -cv::Mat::eye(3, 3, CV_32F);
        J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * (i + 1), 3 * (i + 1) + 3) += cv::Mat::eye(3, 3, CV_32F);
        J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * N, 3 * N + 3) -= Rwb1 * JV12;
        e.rowRange(6 * i + 3, 6 * i + 6) = g * t12 + Rwb1 * dV12;
    }

    cv::Mat H = J.t() * J;
    cv::Mat B = J.t() * e;
    cv::Mat x(nVar, 1, CV_32F);
    cv::solve(H, B, x);

    bax = x.at<float>(3 * N);
    bay = x.at<float>(3 * N + 1);
    baz = x.at<float>(3 * N + 2);

    for (int i = 0; i < N; i++)
    {
        Frame *pF = vpFs[i];
        x.rowRange(3 * i, 3 * i + 3).copyTo(pF->mVw);
        if (i > 0)
        {
            pF->mImuBias.bax = bax;
            pF->mImuBias.bay = bay;
            pF->mImuBias.baz = baz;
            pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
        }
    }
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}

/**
 * @brief 跟踪，整个类就靠这个活着呢
 */
void Tracking::Track()
{
#ifdef SAVE_TIMES
    mTime_PreIntIMU = 0;
    mTime_PosePred = 0;
    mTime_LocalMapTrack = 0;
    mTime_NewKF_Dec = 0;
#endif

    if (bStepByStep)
    {
        while (!mbStep)
            usleep(500);
        mbStep = false;
    }
    // imu出问题了，要重置地图。
    if (mpLocalMapper->mbBadImu)
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }
    // 提取当前地图，如果没有则创建，如果坏了，循环一直等待
    Map *pCurrentMap = mpAtlas->GetCurrentMap();

    // 可以理解为第二帧及后面的帧到来
    if (mState != NO_IMAGES_YET)
    {
        // 判断时间上的前后关系，1是时间上不能反了，2是间隔不能超过1s，有这种情况直接重置地图
        if (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp)
        {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if (mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 1.0)
        {
            cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            if (mpAtlas->isInertial())
            {
                if (mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    if (!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }
            }

            return;
        }
    }

    // 使用上个关键帧的IMU偏置作为当前帧，此时mpImuPreintegrated为NULL，新的帧的预积分在下面PreintegrateIMU()中进行赋值
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState; // 没用
    // 预积分，mbCreatedMap在CreateMapInAtlas后为true，但类的初始化为false
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && !mbCreatedMap)
    {
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU();
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        mTime_PreIntIMU = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
#endif
    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    // 跟踪时地图不能更新，设置地图更新id
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if (nCurMapChangeIndex > nMapChangeIndex)
    {
        // cout << "Map update detected" << endl;
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }

    // 下面是本函数的主要内容！！！前面都是些地图和IMU准备相关
    // 步骤1：初始化
    if (mState == NOT_INITIALIZED)
    {
        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO)
            StereoInitialization();
        else
            MonocularInitialization();
        // 显示通过类track来控制
        mpFrameDrawer->Update(this);

        if (mState != OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }
        // 初始化成功的那帧作为第一帧，且只有一个map
        if (mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else // 步骤2：跟踪，总体来看就是跟踪丢失的话有可能下一帧成为近期丢失状态，做重定位，如果失败，下下帧就新开个地图
    {
        // System is initialized. Track Frame.
        // bOK为临时变量，用于表示每个函数是否执行成功
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // 在viewer中有个开关menuLocalizationMode，有它控制是否ActivateLocalizationMode，并最终管控mbOnlyTracking
        // mbOnlyTracking等于false表示正常VO模式（有地图更新），mbOnlyTracking等于true表示用户手动选择定位模式
        if (!mbOnlyTracking)
        {
#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point timeStartPosePredict = std::chrono::steady_clock::now();
#endif

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            // 2.1：正常初始化成功，上一帧跟踪成功
            if (mState == OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                // 检查并更新上一帧被替换的MapPoints，只有关键帧才可以添加新的MP
                // 更新Fuse函数和SearchAndFuse函数替换的MapPoints，每个MP类里面都有一个mpReplaced 也是MP类用于替换当前MP
                CheckReplacedInLastFrame(); // 更新最近一帧 lastframe 路标点
                // 如果速度为空（当上上帧或者上帧的T没有得到） 且非IMU      或者      刚刚重定位过
                if ((mVelocity.empty() && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                {
                    //Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    // 跟踪参考关键帧
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    //Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    // 先跟匀速模型的，在跟参考关键帧
                    bOK = TrackWithMotionModel();
                    if (!bOK)
                        bOK = TrackReferenceKeyFrame();
                }

                // 相对于2代新添加的，随着增加了多地图系统，多了一个近期丢失，主要是结合IMU看看能不能拽回来，如果非IMU，直接重定位
                if (!bOK)
                {
                    // TODO 这里的状态只有两个，可否直接写if   else
                    // IMU状态下刚重定位过，直接丢失
                    if (mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
                        (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))
                    {
                        mState = LOST;
                    }
                    else if (pCurrentMap->KeyFramesInMap() > 10) // 如果当前地图的关键帧数大于10，状态设定为近期丢失，设置丢失时间
                    {
                        cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                        //mCurrentFrame.SetPose(mLastFrame.mTcw);
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            else // 2.2 跟踪失败
            {

                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;
                    // IMU状态时
                    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))
                    {
                        if (pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp - mTimeStampLost > time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK = false;
                        }
                    }
                    else // 非IMU时
                    {
                        // TODO fix relocalization
                        bOK = Relocalization(); // 重定位
                        if (!bOK)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK = false;
                        }
                    }
                }
                else if (mState == LOST) // 上一帧为最近丢失且重定位失败时
                {
                    // 开启一个新地图
                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap() < 10) // 如果数量很小，就不要这个地图了
                    {
                        mpSystem->ResetActiveMap();
                        cout << "Reseting current map..." << endl;
                    }
                    else
                        CreateMapInAtlas(); // 创建新地图
                    // 上一个关键帧干掉
                    if (mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }

#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point timeEndPosePredict = std::chrono::steady_clock::now();

            mTime_PosePred = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(timeEndPosePredict - timeStartPosePredict).count();
#endif
        }
        else // 纯定位模式，时间关系暂时不看 TOSEE
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if (mState == LOST)
            {
                if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            }
            else
            {
                if (!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if (!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint *> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if (!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if (bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if (mbVO)
                        {
                            for (int i = 0; i < mCurrentFrame.N; i++)
                            {
                                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if (bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        // mpReferenceKF先是上一时刻的参考关键帧，如果当前为新关键帧则变成当前关键帧，如果不是新的关键帧则先为上一帧的参考关键帧，而后经过更新局部关键帧重新确定
        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 步骤3：在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
        // local map:当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系
        // 在步骤2中主要是两两跟踪（恒速模型跟踪上一帧、跟踪参考帧），这里搜索局部关键帧后搜集所有局部MapPoints，
        // 然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if (!mbOnlyTracking)
        {
            // 只有跟踪上或者重定位上才进行这个
            if (bOK)
            {
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point time_StartTrackLocalMap = std::chrono::steady_clock::now();
#endif
                bOK = TrackLocalMap();
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point time_EndTrackLocalMap = std::chrono::steady_clock::now();

                mTime_LocalMapTrack = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndTrackLocalMap - time_StartTrackLocalMap).count();
#endif
            }
            if (!bOK)
                cout << "Fail to track local map!" << endl;
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if (bOK && !mbVO)
                bOK = TrackLocalMap();
        }
        // 到此为止跟踪确定位姿阶段结束，下面开始做收尾工作和为下一帧做准备

        // 查看到此为止时的两个状态变化
        // bOK的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---true
        //          \               \              \---局部地图跟踪失败---false
        //           \               \---当前帧跟踪失败---false
        //            \---上一帧跟踪失败---重定位成功---局部地图跟踪成功---true
        //                          \           \---局部地图跟踪失败---false
        //                           \---重定位失败---false

        //
        // mState的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---OK
        //            \               \              \---局部地图跟踪失败---OK
        //             \               \---当前帧跟踪失败---非OK
        //              \---上一帧跟踪失败---重定位成功---局部地图跟踪成功---非OK
        //                            \           \---局部地图跟踪失败---非OK
        //                             \---重定位失败---非OK（传不到这里，因为直接return了）
        // 由上图可知当前帧的状态OK的条件是跟踪局部地图成功，重定位或正常跟踪都可
        if (bOK)
            mState = OK;
        else if (mState == OK) // 由上图可知只有当正常跟踪成功，但局部地图跟踪失败时执行
        {
            // 带有IMU时状态变为最近丢失，否则直接丢失
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if (!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState = RECENTLY_LOST;
            }
            else
                mState = LOST; // visual to lost

            if (mCurrentFrame.mnId > mnLastRelocFrameId + mMaxFrames)
            {
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            }
        }

        //
        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if ((mCurrentFrame.mnId < (mnLastRelocFrameId + mnFramesToResetIMU)) &&
            (mCurrentFrame.mnId > mnFramesToResetIMU) &&
            ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) &&
            pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame *pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if (pCurrentMap->isImuInitialized())
        {
            if (bOK)
            {
                if (mCurrentFrame.mnId == (mnLastRelocFrameId + mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    // TODO 这个函数是空的，每次重定位后都要执行什么？
                    ResetFrameIMU();
                }
                else if (mCurrentFrame.mnId > (mnLastRelocFrameId + 30))
                    mLastBias = mCurrentFrame.mImuBias;  // 没啥用，后面会重新赋值后传给普通帧
            }
        }

        // Update drawer
        mpFrameDrawer->Update(this);
        if (!mCurrentFrame.mTcw.empty())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        // 查看到此为止时的两个状态变化
        // bOK的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---true
        //          \               \              \---局部地图跟踪失败---false
        //           \               \---当前帧跟踪失败---false
        //            \---上一帧跟踪失败---重定位成功---局部地图跟踪成功---true
        //                          \           \---局部地图跟踪失败---false
        //                           \---重定位失败---false

        // mState的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---OK
        //            \               \              \---局部地图跟踪失败---非OK（IMU时为RECENTLY_LOST）
        //             \               \---当前帧跟踪失败---非OK(地图超过10个关键帧时 RECENTLY_LOST)
        //              \---上一帧跟踪失败(RECENTLY_LOST)---重定位成功---局部地图跟踪成功---OK
        //               \                           \           \---局部地图跟踪失败---LOST
        //                \                           \---重定位失败---LOST（传不到这里，因为直接return了）
        //                 \--上一帧跟踪失败(LOST)--LOST（传不到这里，因为直接return了）
        if (bOK || mState == RECENTLY_LOST)
        {
            // Update motion model
            // 更新匀速模型
            if (!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else
                mVelocity = cv::Mat();
            // 使用IMU积分的位姿显示
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches 只在定位模式下有效？
            // 清除UpdateLastFrame中为当前帧临时添加的MapPoints，个人感觉这里
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
            }
            // 上面只是指向变了，原来新添加的需要delete
            // Delete temporal MapPoints
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point timeStartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame();
#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point timeEndNewKF = std::chrono::steady_clock::now();

            mTime_NewKF_Dec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(timeEndNewKF - timeStartNewKF).count();
#endif

            // Check if we need to insert a new keyframe
            if (bNeedKF && (bOK || (mState == RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))))
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST)
        {
            if (pCurrentMap->KeyFramesInMap() <= 5)
            {
                mpSystem->ResetActiveMap();
                return;
            }
            if ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO))
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }

            CreateMapInAtlas();
        }
        // 这里是不是跟上面重复了
        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // 查看到此为止
    // mState的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---OK
    //            \               \              \---局部地图跟踪失败---非OK（IMU时为RECENTLY_LOST）
    //             \               \---当前帧跟踪失败---非OK(地图超过10个关键帧时 RECENTLY_LOST)
    //              \---上一帧跟踪失败(RECENTLY_LOST)---重定位成功---局部地图跟踪成功---OK
    //               \                           \           \---局部地图跟踪失败---LOST
    //                \                           \---重定位失败---LOST（传不到这里，因为直接return了）
    //                 \--上一帧跟踪失败(LOST)--LOST（传不到这里，因为直接return了）
    // last.记录位姿信息，用于轨迹复现
    if (mState == OK || mState == RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (!mCurrentFrame.mTcw.empty())
        {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST); // TODO 这不都是false么 需要测试一下
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }
    }
}

/**
 * @brief 双目，左右目，rgbd的地图初始化
 */
void Tracking::StereoInitialization()
{
    // 特征点数量大于500
    if (mCurrentFrame.N > 500)
    {
        if (mSensor == System::IMU_STEREO)
        {
            // 涉及imu的双目初始化需要有前一帧
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }
            // 加速度改变量太小也跳过？
            if (cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA - mLastFrame.mpImuPreintegratedFrame->avgA) < 0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            if (mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            // 双目初始化与单目不同，不需要计算前后帧的位姿，所以如果成功将以当前帧为第一帧
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO)
        {
            // 以第一帧相机坐标为原点
            cv::Mat Rwb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0, 3).colRange(0, 3).clone();
            cv::Mat twb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0, 3).col(3).clone();
            // TODO 设置IMU的位姿及速度，速度只是给了个初值，后面有非线性优化
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, cv::Mat::zeros(3, 1, CV_32F));
        }
        else
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        // Create KeyFrame
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        // 双目模式创建MP
        if (!mpCamera2)
        {
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }
        }
        // 左右目模式创建MP
        else
        {
            for (int i = 0; i < mCurrentFrame.Nleft; i++)
            {
                // 在右目有匹配点才能算得三维点
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if (rightIndex != -1)
                {
                    cv::Mat x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini, i);
                    pNewMP->AddObservation(pKFini, rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP, i);
                    pKFini->AddMapPoint(pNewMP, rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft] = pNewMP;
                }
            }
        }
        // 更新一些下一帧需要的东东
        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState = OK;
    }
}

/**
 * @brief 单目的地图初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 */
void Tracking::MonocularInitialization()
{
    // 如果单目初始器还没有被创建，则创建单目初始器，说明这是第一帧
    if (!mpInitializer)
    {
        // Set Reference Frame
        // 单目初始帧的特征点数必须大于100
        if (mCurrentFrame.mvKeys.size() > 100)
        {
            // 步骤1：得到用于初始化的第一帧，初始化需要两帧
            mInitialFrame = Frame(mCurrentFrame);
            // 记录最近的一帧, 第一帧时mLastFrame还不存在，通过mpInitializer没有初始化可知
            mLastFrame = Frame(mCurrentFrame);
            // mvbPrevMatched最大的情况就是所有特征点都被跟踪上，不过基本不可能
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            // 这两句貌似多余，但显得非常严谨，手动点赞
            if (mpInitializer)
                delete mpInitializer;

            // 由当前帧构造初始器 sigma:1.0 iterations:200，没用到这个东西，只是起了个存不存在的作用
            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            // imu下类似mpInitializer的处理，第一帧可能初始化失败，如果重新开始初始化需要重新更新mpImuPreintegratedFromLastKF
            if (mSensor == System::IMU_MONOCULAR)
            {
                if (mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            }
            return;
        }
    }
    else
    {
        // 步骤2：如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
        // 如果当前帧特征点太少，重新构造初始器
        // 因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程，或者带IMU时，帧与帧时间上要小于1s
        if (((int)mCurrentFrame.mvKeys.size() <= 100) || ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp > 1.0)))
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            return;
        }

        // Find correspondences
        // 步骤3：在mInitialFrame与mCurrentFrame中找匹配的特征点对(难理解)
        // mvbPrevMatched中记录了初始帧每个点在上一帧的匹配点的位置，如果点没有匹配上，则保留初始帧点的位置，
        // 这波操作主要是如果当前帧并未初始成功的话，但也要利用匹配好的点，因为设定了范围搜索，而匹配好的点由近些的帧对应的特征点位置更准确一些，毕竟隔一帧匹配点像素坐标的改变理论上要比隔多帧近。
        // 一句话概况mvbPrevMatched作用就是增加范围设定的精准度
        // mvIniMatches存储mInitialFrame, mCurrentFrame之间匹配的特征点
        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        // 步骤4：如果初始化的两帧之间的匹配点太少，重新初始化
        if (nmatches < 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }

        cv::Mat Rcw;                 // Current Camera Rotation
        cv::Mat tcw;                 // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        // 步骤5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints（值得细看）
        if (mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn, mCurrentFrame.mvKeysUn, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            // 步骤6：删除那些无法进行三角化的匹配点
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            // 由Rcw和tcw构造Tcw, 并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            // 步骤6：将三角化得到的3D点包装成MapPoints
            // Initialize函数会得到mvIniP3D，
            // mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
            // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
            CreateInitialMapMonocular();

            // Just for video
            // bStepByStep = true;
        }
    }
}

/**
 * @brief CreateInitialMapMonocular
 *
 * 为单目摄像头三角化生成MapPoints
 */
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    // 1.创建关键帧
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

    if (mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated *)(NULL);

    // 2.将描述子转为BoW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    // 3.将关键帧插入到地图
    // 凡是关键帧，都要插入地图
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);
    // 4.将3D点包装成MapPoints
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);
        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());

        // 步骤4.1：为该MapPoint添加属性：
        // a.观测到该MapPoint的关键帧
        // b.该MapPoint的描述子
        // c.该MapPoint的平均观测方向和深度范围
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);
        // b.
        pMP->ComputeDistinctiveDescriptors();
        // c.
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }

    // Update Connections
    // 5.更新关键帧间的连接关系，这两帧的父关键帧是对方，子关键帧也是对方
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // 没有使用
    std::set<MapPoint *> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    // 6.BA优化
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(), 20);

    pKFcur->PrintPointDistribution();

    // 步骤6：!!!将MapPoints的中值深度归一化到1，并归一化两帧之间变换
    // 评估关键帧场景深度，q=2表示中值
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if (mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f / medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f / medianDepth;
    // 初始时只有两个关键帧，所以这句的意思可以理解为当前关键帧对应的MP数量要超过50
    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_NORMAL);
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    // 位姿中的xyz全部乘以逆深度
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        // 与位姿一样乘以逆深度，同时更新
        if (vpAllMapPoints[iMP])
        {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            // 更新该MapPoint的平均观测方向和深度范围
            pMP->UpdateNormalAndDepth();
        }
    }
    // 传递预积分
    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        // 经过调试这里的偏置全都是0，所以mpImuPreintegrated->b 与 mpImuPreintegrated->GetUpdatedBias() 相同
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), pKFcur->mImuCalib);
    }

    mpLocalMapper->InsertKeyFrame(pKFini);        // 给局部地图添加关键帧
    mpLocalMapper->InsertKeyFrame(pKFcur);        // 给局部地图添加关键帧
    mpLocalMapper->mFirstTs = pKFcur->mTimeStamp; // 以当前帧为第一个时间，而不是上一帧

    mCurrentFrame.SetPose(pKFcur->GetPose()); // 当前帧的位姿（非关键帧）
    mnLastKeyFrameId = mCurrentFrame.mnId;    // 上一个关键帧的id，下一帧使用
    mpLastKeyFrame = pKFcur;                  // 上一个关键帧，下一帧要用
    mnLastRelocFrameId = mInitialFrame.mnId;  // 上一次重定位帧的ID，使用初始帧id

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame *> vKFs = mpAtlas->GetAllKeyFrames(); // 获得当前地图的所有关键帧，在这里就两个

    // T2*invT1 = invT12 = T21, 理解为1到2的变化矩阵
    cv::Mat deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mVelocity = cv::Mat();
    // 获得旋转的李代数
    Eigen::Vector3d phi = LogSO3(Converter::toMatrix3d(deltaT.rowRange(0, 3).colRange(0, 3)));

    // 计算当前帧与上一帧的时间间隔占当前帧与初始帧的时间间隔的比例，用这个比例计算两帧之间的速度
    double aux = (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) / (mCurrentFrame.mTimeStamp - mInitialFrame.mTimeStamp);
    // 计算角速度，没有下文了
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini); // 将初始化的帧添加到mvpKeyFrameOrigins，mvpKeyFrameOrigins与essentialgraph有关，里面只存放初始化成功的第一帧

    mState = OK; // 初始化成功，至此，初始化过程完成

    initID = pKFcur->mnId;
}

/** 
 * @brief 所有跟状态相关的变量全部重置
 * 1. 前后两帧对应的时间戳反了
 * 2. imu模式下前后帧超过1s
 * 3. 上一帧为最近丢失且重定位失败时
 * 4. 重定位成功，局部地图跟踪失败
 */
void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
        mpAtlas->SetInertialSensor(); // mpAtlas中map的mbIsInertial=true
    mbSetInit = false;                // 好像没什么用

    mnInitialFrameId = mCurrentFrame.mnId + 1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mVelocity = cv::Mat();
    mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId + 1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        if (mpInitializer)
            delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
    }

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
    }

    if (mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

    if (mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame *>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

/**
 * @brief 检查上一帧中的MapPoints是否被替换，如果有，则更换对应的MP
 * 
 * Local Mapping线程可能会将关键帧中某些MapPoints进行替换，由于tracking中需要用到mLastFrame，这里检查并更新上一帧中被替换的MapPoints
 * @see LocalMapping::SearchInNeighbors()
 */
void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];

        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/**
 * @brief 对参考关键帧的MapPoints进行跟踪
 * 
 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 * 2. 对属于同一node的描述子进行匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7, true);
    // 大小与mCurrentFrame中特征点数量一样
    vector<MapPoint *> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些

    //mCurrentFrame.PrintPointDistribution();

    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    // 优化当前帧
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if (mCurrentFrame.mvpMapPoints[i])
        {
            // 经过BA一些匹配关系被干掉
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                if (i < mCurrentFrame.Nleft)
                {
                    pMP->mbTrackInView = false;
                }
                else
                {
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;                // tracklocalmap 里面要用
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; // tracklocalmap 里面要用
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    // TODO check these conditions
    // IMU状态下直接返回成功，作者可能认为后面还有tracklocalmap兜底吧，非IMU需要经过BA的内点数量大于10
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
        return true;
    else
        return nmatchesMap >= 10;
}

/**
 * @brief 专门给定位模式（非单目）使用的，给上一帧凭空多添加几个MP（非单目），有助于当前帧的跟踪，可以不怎么关注
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame *pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose()); // TODO 是不是多余的操作，上一帧的pose不是已经确定么？有机会打印一下看看

    // 如果上一帧为关键帧，或者单目的情况，或者非定位模式则退出
    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    // 找到上一帧所有点的深度值并从小到大排序
    for (int i = 0; i < mLastFrame.N; i++)
    {
        float z = mLastFrame.mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    if (vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // 深度值小于配置文件中的阈值点数要超过100个
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP)
            bCreateNew = true;
        else if (pMP->Observations() < 1)
        {
            bCreateNew = true;
        }

        if (bCreateNew)
        {
            // 这些生成MapPoints后并没有通过：
            // a.AddMapPoint、
            // b.AddObservation、
            // c.ComputeDistinctiveDescriptors、
            // d.UpdateNormalAndDepth添加属性，
            // 这些MapPoint仅仅为了提高双目和RGBD的跟踪成功率
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint *pNewMP = new MapPoint(x3D, mpAtlas->GetCurrentMap(), &mLastFrame, i);

            mLastFrame.mvpMapPoints[i] = pNewMP;

            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }
        // mThDepth配置文件里的阈值
        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
        {
            break;
        }
    }
}

/**
 * @brief 跟踪匀速模型
 */
bool Tracking::TrackWithMotionModel()
{
    // 要求更低些
    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // 专门给定位模式（非单目）使用的，可以不怎么关注
    UpdateLastFrame();

    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU))
    {
        // Predict ste with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
        return true;
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    }

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

    // Project points seen in previous frame
    // th表示一个搜索范围，还会乘上一个金字塔层数
    int th;

    if (mSensor == System::STEREO)
        th = 7;
    else
        th = 15;
    // 步骤2：根据匀速度模型进行对上一帧的MapPoints进行跟踪
    // 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    // 如果跟踪的点少，则清空后扩大搜索半径再来一次
    if (nmatches < 20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);
    }

    if (nmatches < 20)
    {
        // 感觉对加imu后条件放缓了许多啊
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
            return true;
        else
            return false;
    }
    // 后面基本跟TrackReferenceKeyFrame相似
    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                if (i < mCurrentFrame.Nleft)
                {
                    pMP->mbTrackInView = false;
                }
                else
                {
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    if (mbOnlyTracking)
    {
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }
    // TODO check these conditions
    // IMU状态下直接返回成功，作者可能认为后面还有tracklocalmap兜底吧，非IMU需要经过BA的内点数量大于10
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
        return true;
    else
        return nmatchesMap >= 10;
}

/**
 * @brief 对Local Map的MapPoints进行跟踪
 * 
 * 1. 更新局部地图，包括局部关键帧和关键点
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 * @see V-D track Local Map
 */
bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;

    // 步骤1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
    UpdateLocalMap();
    // 步骤2：在局部地图中查找与当前帧匹配的MapPoints
    SearchLocalPoints();

    // TOO check outliers before PO
    // 查看内外点，调试用的
    int aux1 = 0, aux2 = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
        if (mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if (mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    // 在这个函数之前，在Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel中都有位姿优化，
    // 步骤3：更新局部所有MapPoints后对位姿再次优化
    int inliers;
    if (!mpAtlas->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else
    {
        // 初始化，重定位，重新开启一个地图都会使mnLastRelocFrameId变化
        if (mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU)
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            // mbMapUpdated变化见Tracking::PredictStateIMU()
            if (!mbMapUpdated) //  && (mnMatchesInliers>30))
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    // 查看内外点，调试用的
    aux1 = 0, aux2 = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
        if (mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if (mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    // 步骤3：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            // 由于当前帧的MapPoints可以被当前帧观测到，其被观测统计量加1
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!mbOnlyTracking)
                {
                    // 该MapPoint被其它相机观测到过，
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        mnMatchesInliers++;
                }
                else // 记录当前帧跟踪到的MapPoints，用于统计跟踪效果
                    mnMatchesInliers++;
            }
            else if (mSensor == System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // 步骤4：决定是否跟踪成功
    mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
        return false;

    if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST))
        return true;

    if (mSensor == System::IMU_MONOCULAR)
    {
        if (mnMatchesInliers < 15)
        {
            return false;
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO)
    {
        if (mnMatchesInliers < 15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
        if (mnMatchesInliers < 30)
            return false;
        else
            return true;
    }
}

/**
 * @brief 断当前帧是否为关键帧
 * @return true if needed
 */
bool Tracking::NeedNewKeyFrame()
{
    // 步骤1：系统状态与地图相关的判断
    // 有imu时，且地图没有imu初始化
    if (((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        // 当前帧距上个关键帧时间上超过0.25s
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
            return true;
        else if (mSensor == System::IMU_STEREO && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
            return true;
        else
            return false;
    }

    if (mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // 如果局部地图被闭环检测使用，则不插入关键帧
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    {
        return false;
    }

    // Return false if IMU is initialazing
    // imu在初始化的时候返回false
    if (mpLocalMapper->IsInitializing())
        return false;
    // 获取当前地图的关键帧数量
    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // 步骤2：判断是否距离上一次插入关键帧的时间太短
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率
    // 这个判定可以理解为：如果关键帧比较少，则考虑插入关键帧。或距离上一次重定位超过1s，则考虑插入关键帧
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    // 步骤3：得到参考关键帧跟踪到的MapPoints数量
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // 步骤4：查询局部地图管理器是否繁忙
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose = 0;
    // 统计双目或gbd时深度在(0, mThDepth)范围内的点数
    if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft; // 针孔双目或者鱼眼双目
        for (int i = 0; i < N; i++)
        {
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
            {
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }
    // 点数在一定范围内
    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // Thresholds
    // Thresholds
    // 设定inlier阈值，和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    if (nKFs < 2)
        thRefRatio = 0.4f; // 关键帧只有一帧，那么插入关键帧的阈值设置很低

    if (mSensor == System::MONOCULAR)
        thRefRatio = 0.9f;

    if (mpCamera2)
        thRefRatio = 0.75f;

    if (mSensor == System::IMU_MONOCULAR)
    {
        if (mnMatchesInliers > 350) // 当前帧跟踪局部地图的MP点数 Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // 很长时间没有插入关键帧，超过1s
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // localMapper处于空闲状态，且当前帧在时间上要大于上一个关键帧（这个条件基本就是为true的）
    const bool c1b = ((mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames) && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    // 跟踪要跪的节奏，0.25和0.3是一个比较低的阈值，除了单目，imu单目，imu双目
    // 含义是:当前帧track的内点相比参考关键帧来说很少
    // 或者本身track的close特征点很少,但是可以被三角化的close特征点数量足够.c1c成立说明track比较弱,有lost的风险. 但通过三角化能够通过该帧生成足够的新地图点
    // 其中:mnMatchesInliers是其中track局部地图时当前帧中 内点中 nObs>0的特征点的数量
    // 其中:nRefMatches参考关键帧的mvpMapPoints中 obn大于某阈值nMinObs 的mappoint数量
    const bool c1c = mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR && mSensor != System::IMU_STEREO && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // TOSEE 阈值比c1c要高，与之前参考帧重复度不是太高（没看出来哪有考虑重复度的问题）
    // 含义是:当前帧track的内点和参考关键帧的地图点的重合度要低于一定阈值,或者本身track的close特征点不能很多,但是当前帧track的内点要多余15个以保证当前帧的位姿精度.c2是可以被判定为关键帧的所必须满足的条件
    // 其中,如果是双目系统thRefRatio = 0.75f(但如果已有关键帧数量不超过2个,则thRefRatio = 0.4f)
    // 如果是单目系统thRefRatio = 0.9f
    const bool c2 = (((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose)) && mnMatchesInliers > 15);

    // Temporal condition for Inertial cases
    // imu新加的条件，时间上距上次添加超过了0.5s
    bool c3 = false;
    if (mpLastKeyFrame)
    {
        if (mSensor == System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
                c3 = true;
        }
        else if (mSensor == System::IMU_STEREO)
        {
            if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    // 单目imu时，跟localmap匹配的点数在[15, 75]间，或者最近丢失状态
    if ((((mnMatchesInliers < 75) && (mnMatchesInliers > 15)) || mState == RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR))) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4 = true;
    else
        c4 = false;

    if (((c1a || c1b || c1c) && c2) || c3 || c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle) // mpLocalMapper->AcceptKeyFrames()
        {
            return true;
        }
        else
        {
            // 字面意思是停止BA，具体怎么停再看
            mpLocalMapper->InterruptBA();
            // 非单目时
            // 队列里不能阻塞太多关键帧
            // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
            // 然后localmapper再逐个pop出来插入到mspKeyFrames
            if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

/**
 * @brief 创建新的关键帧
 *
 * 对于非单目的情况，同时创建新的MapPoints
 */
void Tracking::CreateNewKeyFrame()
{
    if (mpLocalMapper->IsInitializing())
        return;

    if (!mpLocalMapper->SetNotStop(true))
        return;
    // 步骤1：将当前帧构造成关键帧
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

    if (mpAtlas->isImuInitialized())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    // 步骤2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if (mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    // 刷新mpImuPreintegratedFromLastKF
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
    }
    // 这段代码和UpdateLastFrame中的那一部分代码功能相同
    // 步骤3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
    if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        // 根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // 步骤3.1：得到当前帧深度小于阈值的特征点
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if (mSensor == System::IMU_STEREO)
            maxPoint = 100;

        // 创建新的MapPoint, depth < mThDepth
        vector<pair<float, int>> vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        // 查看这个点是否有深度
        for (int i = 0; i < N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty())
        {
            // 步骤3.2：按照深度从小到大排序
            sort(vDepthIdx.begin(), vDepthIdx.end());

            // 步骤3.3：将距离比较近的点包装成MapPoints
            int nPoints = 0;
            for (size_t j = 0; j < vDepthIdx.size(); j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew)
                {
                    cv::Mat x3D;

                    // 表示在帧创建的时候已经通过左右目计算得到特征点对应的三维坐标了
                    if (mCurrentFrame.Nleft == -1)
                    {
                        x3D = mCurrentFrame.UnprojectStereo(i);
                    }
                    else
                    {
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF, i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    // 对于左右目情况下还要对右目对应特征点也要添加MP
                    if (mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0)
                    {
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
                        pNewMP->AddObservation(pKF, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }
                    // 这些添加属性的操作是每次创建MapPoint后都要做的
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++; // TODO check ???
                }
                // 这里决定了双目和rgbd摄像头时地图点云的稠密程度
                // 但是仅仅为了让地图稠密直接改这些不太好，
                // 因为这些MapPoints会参与之后整个slam过程
                if (vDepthIdx[j].first > mThDepth && nPoints > maxPoint)
                {
                    break;
                }
            }

            Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
    //cout  << "end creating new KF" << endl;
}

/**
 * @brief 在局部地图中查找与当前帧匹配的MapPoints
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // 步骤1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
    // 因为当前的mvpMapPoints一定在当前帧的视野中
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = static_cast<MapPoint *>(NULL);
            }
            else
            {
                // 更新能观测到该点的帧数加1，表示在视野内，但不一定匹配上，所以这个值大于等于mobs
                pMP->IncreaseVisible();
                // 标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该点将来不被投影，因为已经匹配过
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    // 步骤2：将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        // 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        // 步骤2.1：判断LocalMapPoints中的点是否在在视野内
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            // 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
            pMP->IncreaseVisible();
            // 只有在视野范围内的MapPoints才参与之后的投影匹配
            nToMatch++;
        }
        // mbTrackInView==false的点有几种：
        // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
        // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
        // c 不在当前相机视野中的点（即未通过isInFrustum判断）
        // 目前来看mbTrackInView为true的条件就是通过isInFrustum
        if (pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);  // 显示时候用到
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if (mSensor == System::RGBD)
            th = 3;
        if (mpAtlas->isImuInitialized())
        {
            if (mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th = 2;
            else
                th = 3;
        }
        else if (!mpAtlas->isImuInitialized() && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))
        {
            th = 10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 5;

        if (mState == LOST || mState == RECENTLY_LOST) // Lost for less than 1 second
            th = 15;                                   // 15
        // 投影匹配，匹配上的直接赋值到mCurrentFrame里面
        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

/**
 * @brief 更新局部地图
 */
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    // BUG 显示用的 这行程序放在UpdateLocalPoints函数后面是不是好一些
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    // 更新局部关键帧和局部MapPoints
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/**
 * @brief 更新局部关键帧，called by UpdateLocalMap()
 *
 * 遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧取出，更新mvpLocalKeyFrames
 */
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // 步骤1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧
    // keyframeCounter里面装的是跟当前帧有共视关系的关键帧以及对应的点数
    map<KeyFrame *, int> keyframeCounter;
    // 地图没有imu初始化或者刚完成重定位使用当前帧构建局部关键帧
    if (!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2))
    {

        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP)
            {
                // 别的线程可能把这个MP作废了，需要在当前帧去掉
                if (!pMP->isBad())
                {
                    const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                    for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }
    }
    else // 使用上一帧构建局部地图
    {
        for (int i = 0; i < mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            // BUG 这里跟里面的判定是不是重复了
            if (mLastFrame.mvpMapPoints[i])
            {
                MapPoint *pMP = mLastFrame.mvpMapPoints[i];
                if (!pMP)
                    continue;
                if (!pMP->isBad())
                {
                    const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                    for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i] = NULL;
                }
            }
        }
    }

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    // 步骤2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
    // 先清空局部关键帧
    mvpLocalKeyFrames.clear();

    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // V-D K1: shares the map points with current frame
    // 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        // mnTrackReferenceForFrame防止重复添加局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // 策略2：与策略1得到的局部关键帧共视程度很高的关键帧作为局部关键帧
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80) // 80
            break;

        KeyFrame *pKF = *itKF;

        // 策略2.1:最佳共视的10帧
        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        // 添加一个就break
        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                // 防止重复添加
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }
        // 策略2.2:自己的子关键帧
        const set<KeyFrame *> spChilds = pKF->GetChilds();
        for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad())
            {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }
        // 策略2.3:自己的父关键帧
        KeyFrame *pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }
    // 策略3:找到最近20个关键帧，英文注释写了10个？？
    // Add 10 last temporal KFs (mainly for IMU)
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mvpLocalKeyFrames.size() < 80)
    {
        //cout << "CurrentKF: " << mCurrentFrame.mnId << endl;
        KeyFrame *tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for (int i = 0; i < Nd; i++)
        {
            if (!tempKeyFrame)
                break;
            //cout << "tempKF: " << tempKeyFrame << endl;
            if (tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                tempKeyFrame = tempKeyFrame->mPrevKF;
            }
        }
    }
    // 步骤3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @brief 更新局部关键点，called by UpdateLocalMap()
 * 
 * 局部关键帧mvpLocalKeyFrames的MapPoints，更新mvpLocalMapPoints
 */
void Tracking::UpdateLocalPoints()
{
    // 步骤1：清空局部MapPoints
    mvpLocalMapPoints.clear();

    // int count_pts = 0;
    // 步骤2：遍历局部关键帧mvpLocalKeyFrames
    for (vector<KeyFrame *>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), itEndKF = mvpLocalKeyFrames.rend(); itKF != itEndKF; ++itKF)
    {
        KeyFrame *pKF = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        // 步骤2：将局部关键帧的MapPoints添加到mvpLocalMapPoints
        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {

            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            // mnTrackReferenceForFrame防止重复添加局部MapPoint
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                // count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * @brief 重定位
 */
bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    // 步骤1：计算当前帧特征点的Bow映射，下面函数要用
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // // 步骤2：找到与当前帧相似的候选关键帧，找到重定位的候选关键帧，这个函数值得一读
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if (vpCandidateKFs.empty())
    {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);

    vector<MLPnPsolver *> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs); // 每个候选关键帧都会估计一个与当前帧的相对位姿

    vector<vector<MapPoint *>> vvpMapPointMatches; //关键帧（向量）的路标点（向量）
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;
    // bow对每一个候选关键帧匹配
    for (int i = 0; i < nKFs; i++)
    {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            // 步骤3：通过BoW进行匹配
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver *pSolver = new MLPnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991); //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;
            // 步骤4：通过PnP算法估计姿态
            MLPnPsolver *pSolver = vpMLPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // 迭代次数如果达到了最高的直接不要了
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }
                // 步骤5：通过PoseOptimization对姿态进行优化求解
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;
                // 优化后一些点被干掉了
                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // 步骤6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                if (nGood < 50)
                {
                    // 返回的是匹配点数，但排除了上面BOW方式得到的匹配点
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        // 如果优化后还是达不到50个，但达到了30个
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            // 这次没有干掉被优化的点
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            // 再以优化后位姿重新搜索一遍，但这次搜索条件更苛刻一些
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                // 干掉被优化的点
                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                        // 这里如果没有经过上面的if是不是就需要再干掉被优化的点？ 有可能后面还会有干掉这个动作吧
                        // 后面确实有，这个动作执行了好几次，可能有冗余，对性能上绝对有影响
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true; // 返回
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }
}

/**
 * @brief 重置
 */
void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if (mpViewer)
    {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
    }
    mbSetInit = false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame *>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
    mvIniMatches.clear();

    if (mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

/**
 * @brief 重置活动地图，
 * 1. 初始化后MP太少
 * 2. 初始化后立马丢失
 * 3. imu模式下跟踪丢失后这个地图还没有经过完整的三段imu初始化
 */
void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if (mpViewer)
    {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            usleep(3000);
    }

    Map *pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();

    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
    }

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for (Map *pMap : mpAtlas->GetAllMaps())
    {
        if (pMap->GetAllKeyFrames().size() > 0)
        {
            if (index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for (list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if (index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames had been set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame *>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
    mvIniMatches.clear();

    if (mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

/**
 * @brief 获得局部地图的MP
 */
vector<MapPoint *> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

/**
 * @brief 更改标定参数
 */
void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

/**
 * @brief 跟踪模式更改
 */
void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

/**
 * @brief 更新了关键帧的位姿，但需要修改普通帧的位姿，因为正常跟踪需要普通帧
 * localmapping中初始化imu中使用，速度的走向（仅在imu模式使用），最开始速度定义于imu初始化时，每个关键帧都根据位移除以时间得到，经过非线性优化保存于KF中.
 * 之后使用本函数，让上一帧与当前帧分别与他们对应的上一关键帧做速度叠加得到，后面新的frame速度由上一个帧速度决定，如果使用匀速模型（大多数情况下），通过imu积分更新速度。
 * 新的关键帧继承于对应帧
 * @param  s 尺度
 * @param  b 初始化后第一帧的偏置
 * @param  pCurrentKeyFrame 当前关键帧
 */
void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame)
{
    Map *pMap = pCurrentKeyFrame->GetMap();
    // 第一帧的id（第一个地图）但是后面没用到这个值
    unsigned int index = mnFirstFrameId;
    // 每一帧的参考关键帧
    list<ORB_SLAM3::KeyFrame *>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for (list<cv::Mat>::iterator lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lbL++)
    {
        // *Tcr 表示参考关键帧到每一帧的SE(3)
        if (*lbL)
            continue;

        KeyFrame *pKF = *lRit;

        while (pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if (pKF->GetMap() == pMap)
        {
            (*lit).rowRange(0, 3).col(3) = (*lit).rowRange(0, 3).col(3) * s;
        }
    }
    // 设置偏置
    mLastBias = b;
    // 设置上一关键帧，如果说mpLastKeyFrame已经是经过添加的新的kf，而pCurrentKeyFrame还是上一个kf，mpLastKeyFrame直接指向之前的kf
    mpLastKeyFrame = pCurrentKeyFrame;
    // 更新偏置
    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);

    cv::Mat twb1;
    cv::Mat Rwb1;
    cv::Mat Vwb1;
    float t12;

    while (!mCurrentFrame.imuIsPreintegrated())
    {
        // 当前帧需要预积分完毕，这段函数实在localmapping里调用的
        usleep(500);
    }

    // frame的mpLastKeyFrame只是用于预积分（imu模式）
    // TODO 如果上一帧正好是上一帧的上一关键帧（mLastFrame.mpLastKeyFrame与mLastFrame不可能是一个，可以验证一下）
    if (mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                        mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                        mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        t12 = mLastFrame.mpImuPreintegrated->dT;
        // 根据mLastFrame的上一个关键帧的信息（此时已经经过imu初始化了，所以关键帧的信息都是校正后的）以及imu的预积分重新计算上一帧的位姿
        mLastFrame.SetImuPoseVelocity(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                        twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                        Vwb1 + Gz * t12 + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }
    // 当前帧是否做了预积分
    if (mCurrentFrame.mpImuPreintegrated)
    {
        twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                            twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                            Vwb1 + Gz * t12 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }
    // 暂时没用
    mnFirstImuFrameId = mCurrentFrame.mnId;
}

/**
 * @brief 计算基础矩阵
 */
cv::Mat Tracking::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

    cv::Mat t12x = Converter::tocvSkewMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    return K1.t().inv() * t12x * R12 * K2.inv();
}

// 这个函数没看到在哪里使用过，局部地图线程里面也有个同名函数，还没看具体实现不知道功能上有什么区别，从代码上看不太一样
void Tracking::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    const vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();

    ORBmatcher matcher(0.6, false);

    cv::Mat Rcw1 = mpLastKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpLastKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpLastKeyFrame->GetCameraCenter();

    const float &fx1 = mpLastKeyFrame->fx;
    const float &fy1 = mpLastKeyFrame->fy;
    const float &cx1 = mpLastKeyFrame->cx;
    const float &cy1 = mpLastKeyFrame->cy;
    const float &invfx1 = mpLastKeyFrame->invfx;
    const float &invfy1 = mpLastKeyFrame->invfy;

    const float ratioFactor = 1.5f * mpLastKeyFrame->mfScaleFactor;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKF2 = vpKFs[i];
        if (pKF2 == mpLastKeyFrame)
            continue;

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2 - Ow1;
        const float baseline = cv::norm(vBaseline);

        if ((mSensor != System::MONOCULAR) || (mSensor != System::IMU_MONOCULAR))
        {
            if (baseline < pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline / medianDepthKF2;

            if (ratioBaselineDepth < 0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpLastKeyFrame, pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t, size_t>> vMatchedIndices;
        matcher.SearchForTriangulation(mpLastKeyFrame, pKF2, F12, vMatchedIndices, false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0, 3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for (int ikp = 0; ikp < nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpLastKeyFrame->mvKeysUn[idx1];
            const float kp1_ur = mpLastKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur >= 0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur >= 0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if (bStereo1)
                cosParallaxStereo1 = cos(2 * atan2(mpLastKeyFrame->mb / 2, mpLastKeyFrame->mvDepth[idx1]));
            else if (bStereo2)
                cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

            cv::Mat x3D;
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<float>(3) == 0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
            }
            else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
            {
                x3D = mpLastKeyFrame->UnprojectStereo(idx1);
            }
            else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
            if (z1 <= 0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
            if (z2 <= 0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpLastKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
            const float invz1 = 1.0 / z1;

            if (!bStereo1)
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - mpLastKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
            const float invz2 = 1.0 / z2;
            if (!bStereo2)
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - mpLastKeyFrame->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D - Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D - Ow2;
            float dist2 = cv::norm(normal2);

            if (dist1 == 0 || dist2 == 0)
                continue;

            const float ratioDist = dist2 / dist1;
            const float ratioOctave = mpLastKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint *pMP = new MapPoint(x3D, mpLastKeyFrame, mpAtlas->GetCurrentMap());

            pMP->AddObservation(mpLastKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            mpLastKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            nnew++;
        }
    }
    TrackReferenceKeyFrame();
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

} // namespace ORB_SLAM3
