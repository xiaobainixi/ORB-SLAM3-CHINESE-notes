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

#include "Optimizer.h"

#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "G2oTypes.h"
#include "Converter.h"

#include <mutex>

#include "OptimizableTypes.h"

namespace ORB_SLAM3
{

bool sortByVal(const pair<MapPoint *, int> &a, const pair<MapPoint *, int> &b)
{
    return (a.second < b.second);
}

/** 
 * @brief 全局BA，全局只有在Tracking::CreateInitialMapMonocular()及LoopClosing::RunGlobalBundleAdjustment()调用，
 * Optimizer::GlobalBundleAdjustemnt(pActiveMap, 10, &mbStopGBA, nLoopKF, false);    RunGlobalBundleAdjustment 纯视觉
 * Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(), 20);                  CreateInitialMapMonocular 单目初始化（包括imu）
 * @param pMap 地图
 * @param nIterations 迭代次数
 * @param pbStopFlag 是否停止的标志
 * @param nLoopKF 回环位置mpCurrentKF->mnId
 * @param bRobust 初始化时为true，回环时发现为false
 */
void Optimizer::GlobalBundleAdjustemnt(Map *pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

/** 
 * @brief GlobalBundleAdjustemnt 调用
 * @param vpKFs KF
 * @param vpMP MP
 * @param nIterations 迭代次数
 * @param pbStopFlag 是否停止的标志
 * @param nLoopKF 回环位置mpCurrentKF->mnId
 * @param bRobust 初始化时为true，回环时发现为false
 */
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                    int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    Map *pMap = vpKFs[0]->GetMap();
    // 1. 定义优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>(); // 稀疏求解

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // 创造一些向量装边
    const int nExpectedSize = (vpKFs.size()) * vpMP.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody *> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    // Set KeyFrame vertices
    // 2. 关于关键帧的节点
    long unsigned int maxKFid = 0;
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId == pMap->GetInitKFid()); // 第一帧固定
        optimizer.addVertex(vSE3);
        if (pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;
        //cout << "KF id: " << pKF->mnId << endl;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    //cout << "start inserting MPs" << endl;
    // 3. 关于MP的节点，然后直接定义边
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        MapPoint *pMP = vpMP[i];
        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true); // 边缘化
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        // 一个MP由多个边看到
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;
            if (optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
                continue;
            nEdges++;

            const int leftIndex = get<0>(mit->second);

            if (leftIndex != -1 && pKF->mvuRight[get<0>(mit->second)] < 0)
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust)
                {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMP);
            }
            else if (leftIndex != -1 && pKF->mvuRight[leftIndex] >= 0) //Stereo observation
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust)
                {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMP);
            }

            if (pKF->mpCamera2)
            {
                int rightIndex = get<1>(mit->second);

                if (rightIndex != -1 && rightIndex < pKF->mvKeysRight.size())
                {
                    rightIndex -= pKF->NLeft;

                    Eigen::Matrix<double, 2, 1> obs;
                    cv::KeyPoint kp = pKF->mvKeysRight[rightIndex];
                    obs << kp.pt.x, kp.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);

                    e->mTrl = Converter::toSE3Quat(pKF->mTrl);

                    e->pCamera = pKF->mpCamera2;

                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKF);
                    vpMapPointEdgeBody.push_back(pMP);
                }
            }
        }

        if (nEdges == 0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
        else
        {
            vbNotIncludedMP[i] = false;
        }
    }

    //cout << "end inserting MPs" << endl;
    // Optimize!
    // 4. 优化！
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

    // Recover optimized data
    // 5. 取出结果
    //Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));

        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == pMap->GetOriginKF()->mnId)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            /*if(!vSE3->fixed())
        {
            //cout << "KF " << pKF->mnId << ": " << endl;
            pKF->mHessianPose = cv::Mat(6, 6, CV_64F);
            pKF->mbHasHessian = true;
            for(int r=0; r<6; ++r)
            {
                for(int c=0; c<6; ++c)
                {
                    //cout  << vSE3->hessian(r, c) << ", ";
                    pKF->mHessianPose.at<double>(r, c) = vSE3->hessian(r, c);
                }
                //cout << endl;
            }
        }*/

            pKF->mTcwGBA.create(4, 4, CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF; // 标记这个关键帧参与了这次全局优化

            cv::Mat mTwc = pKF->GetPoseInverse();
            cv::Mat mTcGBA_c = pKF->mTcwGBA * mTwc;
            cv::Vec3d vector_dist = mTcGBA_c.rowRange(0, 3).col(3);
            double dist = cv::norm(vector_dist);
            if (dist > 1)
            {
                int numMonoBadPoints = 0, numMonoOptPoints = 0;
                int numStereoBadPoints = 0, numStereoOptPoints = 0;
                vector<MapPoint *> vpMonoMPsOpt, vpStereoMPsOpt;

                for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
                {
                    ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
                    MapPoint *pMP = vpMapPointEdgeMono[i];
                    KeyFrame *pKFedge = vpEdgeKFMono[i];

                    if (pKF != pKFedge)
                    {
                        continue;
                    }

                    if (pMP->isBad())
                        continue;

                    if (e->chi2() > 5.991 || !e->isDepthPositive())
                    {
                        numMonoBadPoints++;
                    }
                    else
                    {
                        numMonoOptPoints++;
                        vpMonoMPsOpt.push_back(pMP);
                    }
                }

                for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
                {
                    g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
                    MapPoint *pMP = vpMapPointEdgeStereo[i];
                    KeyFrame *pKFedge = vpEdgeKFMono[i];

                    if (pKF != pKFedge)
                    {
                        continue;
                    }

                    if (pMP->isBad())
                        continue;

                    if (e->chi2() > 7.815 || !e->isDepthPositive())
                    {
                        numStereoBadPoints++;
                    }
                    else
                    {
                        numStereoOptPoints++;
                        vpStereoMPsOpt.push_back(pMP);
                    }
                }
                Verbose::PrintMess("GBA: KF " + to_string(pKF->mnId) + " had been moved " + to_string(dist) + " meters", Verbose::VERBOSITY_DEBUG);
                Verbose::PrintMess("--Number of observations: " + to_string(numMonoOptPoints) + " in mono and " + to_string(numStereoOptPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
                Verbose::PrintMess("--Number of discarded observations: " + to_string(numMonoBadPoints) + " in mono and " + to_string(numStereoBadPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
            }
        }
    }
    Verbose::PrintMess("BA: KFs updated", Verbose::VERBOSITY_DEBUG);

    //Points
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint *pMP = vpMP[i];

        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

        if (nLoopKF == pMap->GetOriginKF()->mnId)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF; // 标记这个MP参与了这次全局优化
        }
    }
}

/** 
 * @brief imu初始化优化，LocalMapping::InitializeIMU中使用，地图全部做BA。也就是imu版的GlobalBundleAdjustemnt
 * 误差包含三个残差与两个偏置，优化目标：mp，位姿，偏置，速度
 * 更新： 关键帧位姿，速度，偏置（预积分里的与关键帧里的），mp
 * @param pMap 地图
 * @param its 迭代次数
 * @param bFixLocal 是否固定局部，localmapping中为false
 * @param nLoopId 回环id
 * @param pbStopFlag 是否停止的标志
 * @param bInit 提供priorG时为true，此时偏置只优化最后一帧的至0，然后所有关键帧的偏置都赋值为优化后的值
 *              若为false，则建立每两帧之间的偏置边，优化使其相差为0
 * @param priorG 陀螺仪偏置的信息矩阵系数，主动设置时一般bInit为true，也就是只优化最后一帧的偏置，这个数会作为计算信息矩阵时使用
 * @param priorA 加速度计偏置的信息矩阵系数
 * @param vSingVal 没用，估计调试用的
 * @param bHess 没用，估计调试用的
 */
void Optimizer::FullInertialBA(Map *pMap, int its, const bool bFixLocal, const long unsigned int nLoopId,
                                bool *pbStopFlag, bool bInit, float priorG, float priorA, Eigen::VectorXd *vSingVal, bool *bHess)
{
    // 获取地图里所有mp与kf，以及最大kf的id
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

    // Setup optimizer
    // 1. 很经典的一套设置优化器流程
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-5);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);
    //
    int nNonFixed = 0;

    // 2. 设置关键帧与偏置节点
    // Set KeyFrame vertices
    KeyFrame *pIncKF; // vpKFs中最后一个id符合要求的关键帧
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;
        VertexPose *VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        pIncKF = pKFi;
        bool bFixed = false;
        if (bFixLocal)
        {
            bFixed = (pKFi->mnBALocalForKF >= (maxKFid - 1)) || (pKFi->mnBAFixedForKF >= (maxKFid - 1));
            if (!bFixed)
                nNonFixed++;
            VP->setFixed(bFixed); // 固定，这里可能跟回环有关
        }
        optimizer.addVertex(VP);
        // 如果是初始化的那几个及后面的关键帧，加入速度节点
        if (pKFi->bImu)
        {
            VertexVelocity *VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(bFixed);
            optimizer.addVertex(VV);
            // priorA==0.f 时 bInit为false，也就是又加入了偏置节点
            if (!bInit)
            {
                VertexGyroBias *VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
                VG->setFixed(bFixed);
                optimizer.addVertex(VG);
                VertexAccBias *VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
                VA->setFixed(bFixed);
                optimizer.addVertex(VA);
            }
        }
    }
    // priorA!=0.f 时 bInit为true，加入了偏置节点
    if (bInit)
    {
        VertexGyroBias *VG = new VertexGyroBias(pIncKF);
        VG->setId(4 * maxKFid + 2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias *VA = new VertexAccBias(pIncKF);
        VA->setId(4 * maxKFid + 3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }
    // TODO 暂时不理解，看到回环后再回看这里
    if (bFixLocal)
    {
        if (nNonFixed < 3)
            return;
    }

    // 3. 添加关于imu的边
    // IMU links
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        // 必须有对应的上一个关键帧，感觉跟下面的判定冲突了
        if (!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;
            // 这两个都必须为初始化后的关键帧
            if (pKFi->bImu && pKFi->mPrevKF->bImu)
            {
                // 3.1 根据上一帧的偏置设定当前帧的新偏置
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                // 3.2 提取节点
                g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1);

                g2o::HyperGraph::Vertex *VG1;
                g2o::HyperGraph::Vertex *VA1;
                g2o::HyperGraph::Vertex *VG2;
                g2o::HyperGraph::Vertex *VA2;
                // 根据不同输入配置相应的偏置节点
                if (!bInit)
                {
                    VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2);
                    VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3);
                    VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2);
                    VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3);
                }
                else
                {
                    VG1 = optimizer.vertex(4 * maxKFid + 2);
                    VA1 = optimizer.vertex(4 * maxKFid + 3);
                }

                g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1);

                if (!bInit)
                {
                    if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                    {
                        cout << "Error" << VP1 << ", " << VV1 << ", " << VG1 << ", " << VA1 << ", " << VP2 << ", " << VV2 << ", " << VG2 << ", " << VA2 << endl;

                        continue;
                    }
                }
                else
                {
                    if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                    {
                        cout << "Error" << VP1 << ", " << VV1 << ", " << VG1 << ", " << VA1 << ", " << VP2 << ", " << VV2 << endl;

                        continue;
                    }
                }
                // 3.3 设置边
                EdgeInertial *ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
                ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
                ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG1));
                ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA1));
                ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
                ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));

                g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
                ei->setRobustKernel(rki);
                // 9个自由度的卡方检验（0.05）
                rki->setDelta(sqrt(16.92));

                optimizer.addEdge(ei);
                // 加了每一个关键帧的偏置时，还要优化两帧之间偏置的误差
                if (!bInit)
                {
                    EdgeGyroRW *egr = new EdgeGyroRW();
                    egr->setVertex(0, VG1);
                    egr->setVertex(1, VG2);
                    cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
                    Eigen::Matrix3d InfoG;
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            InfoG(r, c) = cvInfoG.at<float>(r, c);
                    egr->setInformation(InfoG);
                    egr->computeError();
                    optimizer.addEdge(egr);

                    EdgeAccRW *ear = new EdgeAccRW();
                    ear->setVertex(0, VA1);
                    ear->setVertex(1, VA2);
                    cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
                    Eigen::Matrix3d InfoA;
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            InfoA(r, c) = cvInfoA.at<float>(r, c);
                    ear->setInformation(InfoA);
                    ear->computeError();
                    optimizer.addEdge(ear);
                }
            }
            else
            {
                cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
            }
        }
    }
    // 只加入pIncKF帧的偏置，优化偏置到0
    if (bInit)
    {
        g2o::HyperGraph::Vertex *VG = optimizer.vertex(4 * maxKFid + 2);
        g2o::HyperGraph::Vertex *VA = optimizer.vertex(4 * maxKFid + 3);

        // Add prior to comon biases
        EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F));
        epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
        double infoPriorA = priorA; //
        epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

        EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
        epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
        double infoPriorG = priorG; //
        epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);
    }

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    const unsigned long iniMPid = maxKFid * 5;

    vector<bool> vbNotIncludedMP(vpMPs.size(), false);
    // 5. 添加关于mp的节点与边，这段比较好理解，很传统的视觉上的重投影误差
    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        MapPoint *pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        unsigned long id = pMP->mnId + iniMPid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

        bool bAllFixed = true;

        //Set edges
        // 遍历所有能观测到这个点的关键帧
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnId > maxKFid)
                continue;

            if (!pKFi->isBad())
            {
                const int leftIndex = get<0>(mit->second);
                cv::KeyPoint kpUn;
                // 添加边
                if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] < 0) // Monocular observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono *e = new EdgeMono(0);

                    g2o::OptimizableGraph::Vertex *VP = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId));
                    if (bAllFixed)
                        if (!VP->fixed())
                            bAllFixed = false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                }
                else if (leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0) // stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo *e = new EdgeStereo(0);

                    g2o::OptimizableGraph::Vertex *VP = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId));
                    if (bAllFixed)
                        if (!VP->fixed())
                            bAllFixed = false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                }

                if (pKFi->mpCamera2)
                { // Monocular right observation
                    int rightIndex = get<1>(mit->second);

                    if (rightIndex != -1 && rightIndex < pKFi->mvKeysRight.size())
                    {
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double, 2, 1> obs;
                        kpUn = pKFi->mvKeysRight[rightIndex];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMono *e = new EdgeMono(1);

                        g2o::OptimizableGraph::Vertex *VP = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId));
                        if (bAllFixed)
                            if (!VP->fixed())
                                bAllFixed = false;

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                    }
                }
            }
        }

        if (bAllFixed)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
    }

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // 5. 取出优化结果，对应的值赋值
    // Recover optimized data
    //Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;
        VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
        if (nLoopId == 0)
        {
            cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
            pKFi->SetPose(Tcw);
        }
        else
        {
            pKFi->mTcwGBA = cv::Mat::eye(4, 4, CV_32F);
            Converter::toCvMat(VP->estimate().Rcw[0]).copyTo(pKFi->mTcwGBA.rowRange(0, 3).colRange(0, 3));
            Converter::toCvMat(VP->estimate().tcw[0]).copyTo(pKFi->mTcwGBA.rowRange(0, 3).col(3));
            pKFi->mnBAGlobalForKF = nLoopId;
        }
        if (pKFi->bImu)
        {
            VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
            if (nLoopId == 0)
            {
                pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
            }
            else
            {
                pKFi->mVwbGBA = Converter::toCvMat(VV->estimate());
            }

            VertexGyroBias *VG;
            VertexAccBias *VA;
            if (!bInit)
            {
                VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
                VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
            }
            else
            {
                VG = static_cast<VertexGyroBias *>(optimizer.vertex(4 * maxKFid + 2));
                VA = static_cast<VertexAccBias *>(optimizer.vertex(4 * maxKFid + 3));
            }

            Vector6d vb;
            vb << VG->estimate(), VA->estimate();
            IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);
            if (nLoopId == 0)
            {
                pKFi->SetNewBias(b);
            }
            else
            {
                pKFi->mBiasGBA = b;
            }
        }
    }

    //Points
    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint *pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + iniMPid + 1));

        if (nLoopId == 0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopId;
        }
    }

    pMap->IncreaseChangeIndex();
}

/** 
 * @brief 位姿优化，纯视觉时使用。优化目标：单帧的位姿 
 * @param pFrame 待优化的帧
 */
int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    // Set Frame vertex
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
    vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
    vpEdgesMono.reserve(N);
    vpEdgesMono_FHR.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeRight.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);   // 2自由度卡方检验
    const float deltaStereo = sqrt(7.815); // 3自由度卡方检验

    // 向optimizer添加边
    {

        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++)
        {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (pMP)
            {
                //Conventional SLAM
                if (!pFrame->mpCamera2) // 相机2不存在，暂时猜测可能是因为鱼眼双目与针孔双目的区别
                {
                    // Monocular observation
                    // 单目时 也有可能在双目下, 当前帧的左兴趣点找不到匹配的右兴趣点
                    if (pFrame->mvuRight[i] < 0)
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0))); // optimizer.vertex(0)上面声明过，也就是待优化的位姿
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]; // 1/(1.2^kpUn.octave)  正向金字塔，层数越高图片越小
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->pCamera = pFrame->mpCamera;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    else // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }
                //SLAM with respect a rigid body
                else
                {
                    nInitialCorrespondences++;

                    cv::KeyPoint kpUn;

                    if (i < pFrame->Nleft)
                    { //Left camera observation
                        kpUn = pFrame->mvKeys[i];

                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->pCamera = pFrame->mpCamera;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    else
                    { //Right camera observation
                        //continue;
                        kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        pFrame->mvbOutlier[i] = false;

                        ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->pCamera = pFrame->mpCamera2;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        e->mTrl = Converter::toSE3Quat(pFrame->mTrl);

                        optimizer.addEdge(e);

                        vpEdgesMono_FHR.push_back(e);
                        vnIndexEdgeRight.push_back(i);
                    }
                }
            }
        }
    } // 上锁结束

    //cout << "PO: vnIndexEdgeMono.size() = " << vnIndexEdgeMono.size() << "   vnIndexEdgeRight.size() = " << vnIndexEdgeRight.size() << endl;
    if (nInitialCorrespondences < 3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
    const int its[4] = {10, 10, 10, 10};

    int nBad = 0;
    // 删除外点，每次g2o迭代10次
    // 开始优化，总共优化四次，每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）----不知道怎么算的，查表可知 下面两个分别是服从 2和3自由度的卡方分布 0.5的值
    for (size_t it = 0; it < 4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2(); // _error.dot(information()*_error)

            if (chi2 > chi2Mono[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesMono_FHR.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = vpEdgesMono_FHR[i];

            const size_t idx = vnIndexEdgeRight[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Mono[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pFrame->mvbOutlier[idx] = false;
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        if (optimizer.edges().size() < 10)
            break;
    }

    // 设置数值并返回
    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    //cout << "[PoseOptimization]: initial correspondences-> " << nInitialCorrespondences << " --- outliers-> " << nBad << endl;

    return nInitialCorrespondences - nBad;
}

/** 
 * @brief 带IMU时位姿优化，相当于纯视觉时的PoseOptimization，IMU优化时使用的是上一个关键帧，最后的操作主要是为PoseInertialOptimizationLastFrame准备。优化目标：单帧的位姿与偏置
 * 里面的边有 视觉重投影的边，与上一关键帧残差的边，与上一关键帧相比偏置不变的边
 * @param pFrame 待优化的帧
 * @param bRecInit false
 */
int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit)
{
    // 1. 构建优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    int nInitialMonoCorrespondences = 0;   // 单目时记录当前帧的对应MP数量
    int nInitialStereoCorrespondences = 0; // 双目
    int nInitialCorrespondences = 0;

    // Set Frame vertex
    // 设置帧相关的节点：位姿，速度，偏置
    VertexPose *VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity *VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias *VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias *VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft != -1); // 非两个相机时为false

    vector<EdgeMonoOnlyPose *> vpEdgesMono;
    vector<EdgeStereoOnlyPose *> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);
        // 关于MP的节点
        for (int i = 0; i < N; i++)
        {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (pMP)
            {
                cv::KeyPoint kpUn;

                // Left monocular observation
                // (!bRight && pFrame->mvuRight[i]<0) 理解为没有右目且没有双目对应的点，表示纯单目
                // i < Nleft 表示在两个相机情况下点在左目
                if ((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft)
                {
                    if (i < Nleft)                // pair left-right
                        kpUn = pFrame->mvKeys[i]; // 两个相机时没进行畸变矫正
                    else
                        kpUn = pFrame->mvKeysUn[i]; // 单目时经过矫正了

                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose *e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 0);

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                // 双目时
                else if (!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereoOnlyPose *e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                // 右目的点
                if (bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose *e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 1);

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }
    // 统计了有多少对应点
    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    // 放入上一帧节点
    KeyFrame *pKF = pFrame->mpLastKeyFrame;
    VertexPose *VPk = new VertexPose(pKF);
    VPk->setId(4);
    VPk->setFixed(true);
    optimizer.addVertex(VPk);
    VertexVelocity *VVk = new VertexVelocity(pKF);
    VVk->setId(5);
    VVk->setFixed(true);
    optimizer.addVertex(VVk);
    VertexGyroBias *VGk = new VertexGyroBias(pKF);
    VGk->setId(6);
    VGk->setFixed(true);
    optimizer.addVertex(VGk);
    VertexAccBias *VAk = new VertexAccBias(pKF);
    VAk->setId(7);
    VAk->setFixed(true);
    optimizer.addVertex(VAk);

    // 保证一定的固定，并不完全固定
    EdgeInertial *ei = new EdgeInertial(pFrame->mpImuPreintegrated);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    // 关于陀螺仪偏置的边， 误差为两个偏置的差
    EdgeGyroRW *egr = new EdgeGyroRW();
    egr->setVertex(0, VGk);
    egr->setVertex(1, VG);
    cv::Mat cvInfoG = pFrame->mpImuPreintegrated->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
    Eigen::Matrix3d InfoG;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            InfoG(r, c) = cvInfoG.at<float>(r, c);
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    // 关于加速度计偏置的边， 误差为两个偏置的差
    EdgeAccRW *ear = new EdgeAccRW();
    ear->setVertex(0, VAk);
    ear->setVertex(1, VA);
    cv::Mat cvInfoA = pFrame->mpImuPreintegrated->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
    Eigen::Matrix3d InfoA;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            InfoA(r, c) = cvInfoA.at<float>(r, c);
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    float chi2Mono[4] = {12, 7.5, 5.991, 5.991};
    float chi2Stereo[4] = {15.6, 9.8, 7.815, 7.815};

    int its[4] = {10, 10, 10, 10};

    int nBad = 0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers = 0;
    bool bOut = false;
    // 进行四轮筛选，前几轮的卡方检验阈值较大，排出的点较少
    for (size_t it = 0; it < 4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers = 0;
        nInliersMono = 0;
        nInliersStereo = 0;
        float chi2close = 1.5 * chi2Mono[it];

        // For monocular observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            EdgeMonoOnlyPose *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.f;

            if ((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close) || !e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        // For stereo observations
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            EdgeStereoOnlyPose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1); // not included in next optimization
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad = nBadMono + nBadStereo;

        if (optimizer.edges().size() < 10)
        {
            cout << "PIOLKF: NOT ENOUGH EDGES" << endl;
            break;
        }
    }

    // If not too much tracks, recover not too bad points
    // 如果到最后点过少，mvbOutlier里去掉一些
    if ((nInliers < 30) && !bRecInit)
    {
        nBad = 0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose *e1;
        EdgeStereoOnlyPose *e2;
        for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2() < chi2MonoOut)
                pFrame->mvbOutlier[idx] = false;
            else
                nBad++;
        }
        for (size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2() < chi2StereoOut)
                pFrame->mvbOutlier[idx] = false;
            else
                nBad++;
        }
    }

    // Recover optimized pose, velocity and biases
    // 取出结果
    pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb), Converter::toCvMat(VP->estimate().twb), Converter::toCvMat(VV->estimate()));
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

    // Recover Hessian, marginalize keyFframe states and generate new prior for frame
    Eigen::Matrix<double, 15, 15> H;
    H.setZero();

    H.block<9, 9>(0, 0) += ei->GetHessian2();
    H.block<3, 3>(9, 9) += egr->GetHessian2();
    H.block<3, 3>(12, 12) += ear->GetHessian2();

    int tot_in = 0, tot_out = 0;
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        EdgeMonoOnlyPose *e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if (!pFrame->mvbOutlier[idx])
        {
            H.block<6, 6>(0, 0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        EdgeStereoOnlyPose *e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if (!pFrame->mvbOutlier[idx])
        {
            H.block<6, 6>(0, 0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }
    // 供下一帧边缘化使用
    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb, VP->estimate().twb, VV->estimate(), VG->estimate(), VA->estimate(), H);

    return nInitialCorrespondences - nBad;
}

/** 
 * @brief 带IMU时位姿优化，相当于纯视觉时的PoseOptimization，IMU优化时使用的是上一帧，其他与PoseInertialOptimizationLastKeyFrame一样，优化目标：单帧的位姿与偏置
 * 里面的边有 视觉重投影的边，与上一帧残差的边，与上一帧相比偏置不变的边，上一帧位置速度偏置不变的边
 * @param pFrame 待优化的帧
 * @param bRecInit false
 */
int Optimizer::PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int nInitialMonoCorrespondences = 0;
    int nInitialStereoCorrespondences = 0;
    int nInitialCorrespondences = 0;

    // Set Current Frame vertex
    VertexPose *VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity *VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias *VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias *VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft != -1);

    vector<EdgeMonoOnlyPose *> vpEdgesMono;
    vector<EdgeStereoOnlyPose *> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++)
        {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (pMP)
            {
                cv::KeyPoint kpUn;
                // Left monocular observation
                if ((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft)
                {
                    if (i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose *e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 0);

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if (!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereoOnlyPose *e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if (bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose *e = new EdgeMonoOnlyPose(pMP->GetWorldPos(), 1);

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }

    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    // Set Previous Frame Vertex
    Frame *pFp = pFrame->mpPrevFrame;

    VertexPose *VPk = new VertexPose(pFp);
    VPk->setId(4);
    VPk->setFixed(false);
    optimizer.addVertex(VPk);
    VertexVelocity *VVk = new VertexVelocity(pFp);
    VVk->setId(5);
    VVk->setFixed(false);
    optimizer.addVertex(VVk);
    VertexGyroBias *VGk = new VertexGyroBias(pFp);
    VGk->setId(6);
    VGk->setFixed(false);
    optimizer.addVertex(VGk);
    VertexAccBias *VAk = new VertexAccBias(pFp);
    VAk->setId(7);
    VAk->setFixed(false);
    optimizer.addVertex(VAk);

    EdgeInertial *ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP); // VertexPose* VP = new VertexPose(pFrame);
    ei->setVertex(5, VV); // VertexVelocity* VV = new VertexVelocity(pFrame);
    optimizer.addEdge(ei);

    EdgeGyroRW *egr = new EdgeGyroRW();
    egr->setVertex(0, VGk);
    egr->setVertex(1, VG);
    cv::Mat cvInfoG = pFrame->mpImuPreintegratedFrame->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
    Eigen::Matrix3d InfoG;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            InfoG(r, c) = cvInfoG.at<float>(r, c);
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    EdgeAccRW *ear = new EdgeAccRW();
    ear->setVertex(0, VAk);
    ear->setVertex(1, VA);
    cv::Mat cvInfoA = pFrame->mpImuPreintegratedFrame->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
    Eigen::Matrix3d InfoA;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            InfoA(r, c) = cvInfoA.at<float>(r, c);
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    if (!pFp->mpcpi)
        Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId), Verbose::VERBOSITY_NORMAL);

    // 优化方式，pFp->mpcpi中元素与节点元素相差为0
    // 关于上一帧的节点本来可以直接fixed，但是作者可能想保留点优化空间，这个边的信息矩阵的值非常大，经过调试发现改变量很少
    // 关于信息矩阵对整体优化的影响还需要再看
    EdgePriorPoseImu *ep = new EdgePriorPoseImu(pFp->mpcpi);

    ep->setVertex(0, VPk);
    ep->setVertex(1, VVk);
    ep->setVertex(2, VGk);
    ep->setVertex(3, VAk);
    g2o::RobustKernelHuber *rkp = new g2o::RobustKernelHuber;
    ep->setRobustKernel(rkp);
    rkp->setDelta(5);
    optimizer.addEdge(ep);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.

    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
    const int its[4] = {10, 10, 10, 10};

    int nBad = 0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers = 0;
    for (size_t it = 0; it < 4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers = 0;
        nInliersMono = 0;
        nInliersStereo = 0;
        float chi2close = 1.5 * chi2Mono[it];

        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            EdgeMonoOnlyPose *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.f;

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if ((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close) || !e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            EdgeStereoOnlyPose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad = nBadMono + nBadStereo;

        if (optimizer.edges().size() < 10)
        {
            cout << "PIOLF: NOT ENOUGH EDGES" << endl;
            break;
        }
    }

    if ((nInliers < 30) && !bRecInit)
    {
        nBad = 0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose *e1;
        EdgeStereoOnlyPose *e2;
        for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2() < chi2MonoOut)
                pFrame->mvbOutlier[idx] = false;
            else
                nBad++;
        }
        for (size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2() < chi2StereoOut)
                pFrame->mvbOutlier[idx] = false;
            else
                nBad++;
        }
    }

    nInliers = nInliersMono + nInliersStereo;

    // Recover optimized pose, velocity and biases
    pFrame->SetImuPoseVelocity(Converter::toCvMat(VP->estimate().Rwb), Converter::toCvMat(VP->estimate().twb), Converter::toCvMat(VV->estimate()));
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]);

    // Recover Hessian, marginalize previous frame states and generate new prior for frame
    // 恢复海森矩阵（作为下一帧的信息矩阵），边缘化上一帧的状态且生成pFrame新的先验，这个矩阵里的数都很大。。。
    Eigen::Matrix<double, 30, 30> H;
    H.setZero();
    // ei的定义
    // EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);
    // ei->setVertex(0, VPk);
    // ei->setVertex(1, VVk);
    // ei->setVertex(2, VGk);
    // ei->setVertex(3, VAk);
    // ei->setVertex(4, VP);  // VertexPose* VP = new VertexPose(pFrame);
    // ei->setVertex(5, VV);  // VertexVelocity* VV = new VertexVelocity(pFrame);
    // ei->GetHessian()  =  J.t * J 下同，不做详细标注了
    // 角标1表示上一帧，2表示当前帧
    //      6            3             3           3            6            3
    // Jp1.t * Jp1  Jp1.t * Jv1  Jp1.t * Jg1  Jp1.t * Ja1  Jp1.t * Jp2  Jp1.t * Jv2     6
    // Jv1.t * Jp1  Jv1.t * Jv1  Jv1.t * Jg1  Jv1.t * Ja1  Jv1.t * Jp2  Jv1.t * Jv2     3
    // Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1  Jg1.t * Ja1  Jg1.t * Jp2  Jg1.t * Jv2     3
    // Ja1.t * Jp1  Ja1.t * Jv1  Ja1.t * Jg1  Ja1.t * Ja1  Ja1.t * Jp2  Ja1.t * Jv2     3
    // Jp2.t * Jp1  Jp2.t * Jv1  Jp2.t * Jg1  Jp2.t * Ja1  Jp2.t * Jp2  Jp2.t * Jv2     6
    // Jv2.t * Jp1  Jv2.t * Jv1  Jv2.t * Jg1  Jv2.t * Ja1  Jv2.t * Jp2  Jv2.t * Jv2     3
    // 所以矩阵是24*24 的
    H.block<24, 24>(0, 0) += ei->GetHessian();
    // 经过这步H变成了
    // 列数 6            3             3           3            6           3        6
    // ---------------------------------------------------------------------------------- 行数
    // Jp1.t * Jp1  Jp1.t * Jv1  Jp1.t * Jg1  Jp1.t * Ja1  Jp1.t * Jp2  Jp1.t * Jv2   0 |  6
    // Jv1.t * Jp1  Jv1.t * Jv1  Jv1.t * Jg1  Jv1.t * Ja1  Jv1.t * Jp2  Jv1.t * Jv2   0 |  3
    // Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1  Jg1.t * Ja1  Jg1.t * Jp2  Jg1.t * Jv2   0 |  3
    // Ja1.t * Jp1  Ja1.t * Jv1  Ja1.t * Jg1  Ja1.t * Ja1  Ja1.t * Jp2  Ja1.t * Jv2   0 |  3
    // Jp2.t * Jp1  Jp2.t * Jv1  Jp2.t * Jg1  Jp2.t * Ja1  Jp2.t * Jp2  Jp2.t * Jv2   0 |  6
    // Jv2.t * Jp1  Jv2.t * Jv1  Jv2.t * Jg1  Jv2.t * Ja1  Jv2.t * Jp2  Jv2.t * Jv2   0 |  3
    //     0            0            0            0            0           0          0 |  6
    // ----------------------------------------------------------------------------------

    // egr的定义
    // EdgeGyroRW* egr = new EdgeGyroRW();
    // egr->setVertex(0, VGk);
    // egr->setVertex(1, VG);
    Eigen::Matrix<double, 6, 6> Hgr = egr->GetHessian();
    H.block<3, 3>(9, 9) += Hgr.block<3, 3>(0, 0);   // Jgr1.t * Jgr1
    H.block<3, 3>(9, 24) += Hgr.block<3, 3>(0, 3);  // Jgr1.t * Jgr2
    H.block<3, 3>(24, 9) += Hgr.block<3, 3>(3, 0);  // Jgr2.t * Jgr1
    H.block<3, 3>(24, 24) += Hgr.block<3, 3>(3, 3); // Jgr2.t * Jgr2
    // 经过这步H变成了
    // 列数 6            3                    3                      3            6           3             3         3
    // ----------------------------------------------------------------------------------------------------------------- 行数
    // Jp1.t * Jp1  Jp1.t * Jv1         Jp1.t * Jg1           Jp1.t * Ja1  Jp1.t * Jp2  Jp1.t * Jv2        0         0 |  6
    // Jv1.t * Jp1  Jv1.t * Jv1         Jv1.t * Jg1           Jv1.t * Ja1  Jv1.t * Jp2  Jv1.t * Jv2        0         0 |  3
    // Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1 + Jgr1.t * Jgr1  Jg1.t * Ja1  Jg1.t * Jp2  Jg1.t * Jv2  Jgr1.t * Jgr2   0 |  3
    // Ja1.t * Jp1  Ja1.t * Jv1         Ja1.t * Jg1           Ja1.t * Ja1  Ja1.t * Jp2  Ja1.t * Jv2        0         0 |  3
    // Jp2.t * Jp1  Jp2.t * Jv1         Jp2.t * Jg1           Jp2.t * Ja1  Jp2.t * Jp2  Jp2.t * Jv2        0         0 |  6
    // Jv2.t * Jp1  Jv2.t * Jv1         Jv2.t * Jg1           Jv2.t * Ja1  Jv2.t * Jp2  Jv2.t * Jv2        0         0 |  3
    //     0            0             Jgr2.t * Jgr1                 0            0           0       Jgr2.t * Jgr2   0 |  3
    //     0            0                    0                      0            0           0             0         0 |  3
    // -----------------------------------------------------------------------------------------------------------------

    // ear的定义
    // EdgeAccRW* ear = new EdgeAccRW();
    // ear->setVertex(0, VAk);
    // ear->setVertex(1, VA);
    Eigen::Matrix<double, 6, 6> Har = ear->GetHessian();
    H.block<3, 3>(12, 12) += Har.block<3, 3>(0, 0); // Jar1.t * Jar1
    H.block<3, 3>(12, 27) += Har.block<3, 3>(0, 3); // Jar1.t * Jar2
    H.block<3, 3>(27, 12) += Har.block<3, 3>(3, 0); // Jar2.t * Jar1
    H.block<3, 3>(27, 27) += Har.block<3, 3>(3, 3); // Jar2.t * Jar2
    // 经过这步H变成了
    // 列数 6            3                    3                            3                         6           3             3              3
    // --------------------------------------------------------------------------------------------------------------------------------------------------- 行数
    // |  Jp1.t * Jp1  Jp1.t * Jv1         Jp1.t * Jg1                 Jp1.t * Ja1            |  Jp1.t * Jp2  Jp1.t * Jv2        0              0        |  6
    // |  Jv1.t * Jp1  Jv1.t * Jv1         Jv1.t * Jg1                 Jv1.t * Ja1            |  Jv1.t * Jp2  Jv1.t * Jv2        0              0        |  3
    // |  Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1 + Jgr1.t * Jgr1        Jg1.t * Ja1            |  Jg1.t * Jp2  Jg1.t * Jv2  Jgr1.t * Jgr2        0        |  3
    // |  Ja1.t * Jp1  Ja1.t * Jv1         Ja1.t * Jg1           Ja1.t * Ja1 + Jar1.t * Jar1  |  Ja1.t * Jp2  Ja1.t * Jv2  Jar1.t * Jar2        0        |  3
    // |--------------------------------------------------------------------------------------------------------------------------------------------------
    // |  Jp2.t * Jp1  Jp2.t * Jv1         Jp2.t * Jg1                 Jp2.t * Ja1            |  Jp2.t * Jp2  Jp2.t * Jv2        0              0        |  6
    // |  Jv2.t * Jp1  Jv2.t * Jv1         Jv2.t * Jg1                 Jv2.t * Ja1            |  Jv2.t * Jp2  Jv2.t * Jv2        0              0        |  3
    // |      0            0              Jgr2.t * Jgr1                      0                |        0           0       Jgr2.t * Jgr2        0        |  3
    // |      0            0                    0                     Jar2.t * Jar1           |        0           0             0        Jar2.t * Jar2  |  3
    // ---------------------------------------------------------------------------------------------------------------------------------------------------

    // ep定义
    // EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);
    // ep->setVertex(0, VPk);
    // ep->setVertex(1, VVk);
    // ep->setVertex(2, VGk);
    // ep->setVertex(3, VAk);
    //      6            3             3           3
    // Jp1.t * Jp1  Jp1.t * Jv1  Jp1.t * Jg1  Jp1.t * Ja1     6
    // Jv1.t * Jp1  Jv1.t * Jv1  Jv1.t * Jg1  Jv1.t * Ja1     3
    // Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1  Jg1.t * Ja1     3
    // Ja1.t * Jp1  Ja1.t * Jv1  Ja1.t * Jg1  Ja1.t * Ja1     3
    H.block<15, 15>(0, 0) += ep->GetHessian(); // 上一帧 的H矩阵，矩阵太大了不写了。。。总之就是加到下标为1相关的了

    int tot_in = 0, tot_out = 0;
    // 关于位姿的海森
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        EdgeMonoOnlyPose *e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if (!pFrame->mvbOutlier[idx])
        {
            H.block<6, 6>(15, 15) += e->GetHessian(); // 当前帧的H矩阵，矩阵太大了不写了。。。总之就是加到下标为2相关的了
            tot_in++;
        }
        else
            tot_out++;
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        EdgeStereoOnlyPose *e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if (!pFrame->mvbOutlier[idx])
        {
            H.block<6, 6>(15, 15) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    H = Marginalize(H, 0, 14);
    // 保存当前帧边缘化后的信息传到下一帧供下一帧使用
    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb, VP->estimate().twb, VV->estimate(), VG->estimate(), VA->estimate(), H.block<15, 15>(15, 15));
    delete pFp->mpcpi;
    pFp->mpcpi = NULL;

    return nInitialCorrespondences - nBad;
}

/**
 * @brief 没有使用，暂时不看
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, vector<KeyFrame *> &vpNonEnoughOptKFs)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame *> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map *pCurrentMap = pKF->GetMap();

    const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame *pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }
    for (KeyFrame *pKFi : vpNonEnoughOptKFs)
    {
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap && pKFi->mnBALocalForKF != pKF->mnId)
        {
            pKFi->mnBALocalForKF = pKF->mnId;
            lLocalKeyFrames.push_back(pKFi);
        }
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint *> lLocalMapPoints;
    set<MapPoint *> sNumObsMP;
    int num_fixedKF;
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        if (pKFi->mnId == pCurrentMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        vector<MapPoint *> vpMPs = pKFi->GetMapPointMatches();
        for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {

                    if (pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
                }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame *> lFixedCameras;
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map<KeyFrame *, tuple<int, int>> observations = (*lit)->GetObservations();
        for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size() + num_fixedKF;
    if (num_fixedKF < 2)
    {
        //Verbose::PrintMess("LM-LBA: New Fixed KFs had been set", Verbose::VERBOSITY_NORMAL);
        //TODO We set 2 KFs to fixed to avoid a degree of freedom in scale
        list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin();
        int lowerId = pKF->mnId;
        KeyFrame *pLowerKf;
        int secondLowerId = pKF->mnId;
        KeyFrame *pSecondLowerKF;

        for (; lit != lLocalKeyFrames.end(); lit++)
        {
            KeyFrame *pKFi = *lit;
            if (pKFi == pKF || pKFi->mnId == pCurrentMap->GetInitKFid())
            {
                continue;
            }

            if (pKFi->mnId < lowerId)
            {
                lowerId = pKFi->mnId;
                pLowerKf = pKFi;
            }
            else if (pKFi->mnId < secondLowerId)
            {
                secondLowerId = pKFi->mnId;
                pSecondLowerKF = pKFi;
            }
        }
        lFixedCameras.push_back(pLowerKf);
        lLocalKeyFrames.remove(pLowerKf);
        num_fixedKF++;
        if (num_fixedKF < 2)
        {
            lFixedCameras.push_back(pSecondLowerKF);
            lLocalKeyFrames.remove(pSecondLowerKF);
            num_fixedKF++;
        }
    }

    if (num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        //return;
    }
    //Verbose::PrintMess("LM-LBA: There are " + to_string(lLocalKeyFrames.size()) + " KFs and " + to_string(lLocalMapPoints.size()) + " MPs to optimize. " + to_string(num_fixedKF) + " KFs are fixed", Verbose::VERBOSITY_DEBUG);

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pCurrentMap->IsInertial())
        solver->setUserLambdaInit(100.0); // TODO uncomment
    //cout << "LM-LBA: lambda init: " << solver->userLambdaInit() << endl;

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == pCurrentMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    Verbose::PrintMess("LM-LBA: opt/fixed KFs: " + to_string(lLocalKeyFrames.size()) + "/" + to_string(lFixedCameras.size()), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("LM-LBA: local MPs: " + to_string(lLocalMapPoints.size()), Verbose::VERBOSITY_DEBUG);

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody *> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    int nPoints = 0;

    int nKFs = lLocalKeyFrames.size() + lFixedCameras.size(), nEdges = 0;

    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        nPoints++;

        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

        //Set edges
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int cam0Index = get<0>(mit->second);

                // Monocular observation of Camera 0
                if (cam0Index != -1 && pKFi->mvuRight[cam0Index] < 0)
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[cam0Index];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->pCamera = pKFi->mpCamera;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);

                    nEdges++;
                }
                else if (cam0Index != -1 && pKFi->mvuRight[cam0Index] >= 0) // Stereo observation (with rectified images)
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[cam0Index];
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[cam0Index];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);

                    nEdges++;
                }

                // Monocular observation of Camera 0
                if (pKFi->mpCamera2)
                {
                    int rightIndex = get<1>(mit->second);

                    if (rightIndex != -1)
                    {
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double, 2, 1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->mTrl = Converter::toSE3Quat(pKFi->mTrl);

                        e->pCamera = pKFi->mpCamera2;

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKFi);
                        vpMapPointEdgeBody.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }
    }

    if (pbStopFlag)
        if (*pbStopFlag)
        {
            return;
        }

    optimizer.initializeOptimization();

    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int numPerform_it = optimizer.optimize(5);
    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "LBA time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    //std::cout << "Keyframes: " << nKFs << " --- MapPoints: " << nPoints << " --- Edges: " << nEdges << endl;

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore)
    {

        // Check inlier observations
        int nMonoBadObs = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                //e->setLevel(1);
                nMonoBadObs++;
            }

            //e->setRobustKernel(0);
        }

        int nBodyBadObs = 0;
        for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
            MapPoint *pMP = vpMapPointEdgeBody[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                //e->setLevel(1);
                nBodyBadObs++;
            }

            //e->setRobustKernel(0);
        }

        int nStereoBadObs = 0;
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                //e->setLevel(1);
                nStereoBadObs++;
            }

            //e->setRobustKernel(0);
        }
        Verbose::PrintMess("LM-LBA: First optimization has " + to_string(nMonoBadObs) + " monocular and " + to_string(nStereoBadObs) + " stereo bad observations", Verbose::VERBOSITY_DEBUG);

        // Optimize again without the outliers
        //Verbose::PrintMess("LM-LBA: second optimization", Verbose::VERBOSITY_DEBUG);
        //optimizer.initializeOptimization(0);
        //numPerform_it = optimizer.optimize(10);
        numPerform_it += optimizer.optimize(5);
    }

    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());

    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
        MapPoint *pMP = vpMapPointEdgeBody[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    Verbose::PrintMess("LM-LBA: outlier observations: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("LM-LBA: total of observations: " + to_string(vpMapPointEdgeMono.size() + vpMapPointEdgeStereo.size()), Verbose::VERBOSITY_DEBUG);
    bool bRedrawError = false;
    bool bWriteStats = false;

    // Get Map Mutex
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    if (!vToErase.empty())
    {

        //cout << "LM-LBA: There are " << vToErase.size() << " observations whose will be deleted from the map" << endl;
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    //Keyframes
    vpNonEnoughOptKFs.clear();
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        cv::Mat Tiw = Converter::toCvMat(SE3quat);
        cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
        cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
        double dist = cv::norm(trasl);
        pKFi->SetPose(Converter::toCvMat(SE3quat));

        pKFi->mnNumberOfOpt += numPerform_it;
        //cout << "LM-LBA: KF " << pKFi->mnId << " had performed " <<  pKFi->mnNumberOfOpt << " iterations" << endl;
        if (pKFi->mnNumberOfOpt < 10)
        {
            vpNonEnoughOptKFs.push_back(pKFi);
        }
    }
    //Verbose::PrintMess("LM-LBA: Num fixed cameras " + to_string(num_fixedKF), Verbose::VERBOSITY_DEBUG);
    //Verbose::PrintMess("LM-LBA: Num Points " + to_string(lLocalMapPoints.size()), Verbose::VERBOSITY_DEBUG);
    //Verbose::PrintMess("LM-LBA: Num optimized cameras " + to_string(lLocalKeyFrames.size()), Verbose::VERBOSITY_DEBUG);
    //Verbose::PrintMess("----------", Verbose::VERBOSITY_DEBUG);

    //Points
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    pCurrentMap->IncreaseChangeIndex();
}

/**
 * @brief Local Bundle Adjustment LocalMapping::Run() 使用，纯视觉
 *
 * 1. Vertex:
 *     - g2o::VertexSE3Expmap()，LocalKeyFrames，即当前关键帧的位姿、与当前关键帧相连的关键帧的位姿
 *     - g2o::VertexSE3Expmap()，FixedCameras，即能观测到LocalMapPoints的关键帧（并且不属于LocalKeyFrames）的位姿，在优化中这些关键帧的位姿不变
 *     - g2o::VertexSBAPointXYZ()，LocalMapPoints，即LocalKeyFrames能观测到的所有MapPoints的位置
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF        KeyFrame
 * @param pbStopFlag 是否停止优化的标志
 * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 * 
 * 总结下与ORBSLAM2的不同
 * 前面操作基本一样，但优化时2代去掉了误差大的点又进行优化了，3代只是统计但没有去掉继续优化，而后都是将误差大的点干掉
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int &num_fixedKF)
{
    //cout << "LBA" << endl;
    // 该优化函数用于LocalMapping线程的局部BA优化
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame *> lLocalKeyFrames;

    // 步骤1：将当前关键帧加入lLocalKeyFrames
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map *pCurrentMap = pKF->GetMap();

    // 步骤2：找到关键帧连接的关键帧（一级相连），加入lLocalKeyFrames中
    const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame *pKFi = vNeighKFs[i];
        // 记录局部优化id，该数为不断变化，数值等于局部化的关键帧的id，该id用于防止重复添加
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    num_fixedKF = 0;
    // 步骤3：遍历lLocalKeyFrames中关键帧，将它们观测的MapPoints加入到lLocalMapPoints
    list<MapPoint *> lLocalMapPoints;
    set<MapPoint *> sNumObsMP;
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        if (pKFi->mnId == pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        vector<MapPoint *> vpMPs = pKFi->GetMapPointMatches();
        for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {
                    /*if(sNumObsMP.find(pMP) == sNumObsMP.end())
                {
                    sNumObsMP.insert(pMP);
                    int index_mp = pMP->GetIndexInKeyFrame(pKFi);
                    if(pKFi->mvuRight[index_mp]>=0)
                    {
                        // Stereo, it has at least 2 observations by pKFi
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
                    }
                }
                else
                {*/
                    if (pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
                    //}
                }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // 步骤4：得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时不优化
    list<KeyFrame *> lFixedCameras;
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map<KeyFrame *, tuple<int, int>> observations = (*lit)->GetObservations();
        for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    // 步骤4.1：相比ORBSLAM2多出了判断固定关键帧的个数，最起码要两个固定的,如果实在没有就把lLocalKeyFrames中最早的KF固定，还是不够再加上第二早的KF固定
    num_fixedKF = lFixedCameras.size() + num_fixedKF;
    if (num_fixedKF < 2)
    {
        //Verbose::PrintMess("LM-LBA: New Fixed KFs had been set", Verbose::VERBOSITY_NORMAL);
        //TODO We set 2 KFs to fixed to avoid a degree of freedom in scale
        list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin();
        int lowerId = pKF->mnId;
        KeyFrame *pLowerKf;
        int secondLowerId = pKF->mnId;
        KeyFrame *pSecondLowerKF;

        for (; lit != lLocalKeyFrames.end(); lit++)
        {
            KeyFrame *pKFi = *lit;
            if (pKFi == pKF || pKFi->mnId == pMap->GetInitKFid())
            {
                continue;
            }

            if (pKFi->mnId < lowerId)
            {
                lowerId = pKFi->mnId;
                pLowerKf = pKFi;
            }
            else if (pKFi->mnId < secondLowerId)
            {
                secondLowerId = pKFi->mnId;
                pSecondLowerKF = pKFi;
            }
        }
        lFixedCameras.push_back(pLowerKf);
        lLocalKeyFrames.remove(pLowerKf);
        num_fixedKF++;
        if (num_fixedKF < 2)
        {
            lFixedCameras.push_back(pSecondLowerKF);
            lLocalKeyFrames.remove(pSecondLowerKF);
            num_fixedKF++;
        }
    }

    if (num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        //return;
    }
    //Verbose::PrintMess("LM-LBA: There are " + to_string(lLocalKeyFrames.size()) + " KFs and " + to_string(lLocalMapPoints.size()) + " MPs to optimize. " + to_string(num_fixedKF) + " KFs are fixed", Verbose::VERBOSITY_DEBUG);

    // Setup optimizer
    // 步骤5：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
        solver->setUserLambdaInit(100.0); // TODO uncomment
    //cout << "LM-LBA: lambda init: " << solver->userLambdaInit() << endl;

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    // 步骤6：添加顶点：Pose of Local KeyFrame
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == pMap->GetInitKFid()); // //第一帧位置固定
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }
    //Verbose::PrintMess("LM-LBA: KFs to optimize added", Verbose::VERBOSITY_DEBUG);

    // Set Fixed KeyFrame vertices
    // 步骤7：添加顶点：Pose of Fixed KeyFrame，注意这里调用了vSE3->setFixed(true)。
    for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }
    //Verbose::PrintMess("LM-LBA: Fixed KFs added", Verbose::VERBOSITY_DEBUG);

    //Verbose::PrintMess("LM-LBA: opt/fixed KFs: " + to_string(lLocalKeyFrames.size()) + "/" + to_string(lFixedCameras.size()), Verbose::VERBOSITY_DEBUG);
    //Verbose::PrintMess("LM-LBA: local MPs: " + to_string(lLocalMapPoints.size()), Verbose::VERBOSITY_DEBUG);

    // Set MapPoint vertices
    // 步骤7：添加3D顶点
    // 存放的方式(举例)
    // 边id: 1 2 3 4 5 6 7 8 9
    // KFid: 1 2 3 4 1 2 3 2 3
    // MPid: 1 1 1 1 2 2 2 3 3
    // 所以这个个数大约是点数×帧数，实际肯定比这个要少
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    // 存放单目时的边
    vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    // 存放单目时的KF
    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    // 存放单目时的MP
    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // 存放双目鱼眼时另一个相机的KF
    vector<KeyFrame *> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);
    // 存放双目鱼眼时另一个相机的边
    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody *> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);
    // 存放双目鱼眼时另一个相机的MP
    vector<MapPoint *> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    // 存放双目时的边
    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);
    // 存放双目时的KF
    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    // 存放双目时的MP
    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    int nPoints = 0;

    int nKFs = lLocalKeyFrames.size() + lFixedCameras.size(), nEdges = 0;
    // 添加顶点：MapPoint
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        // 这里的边缘化与滑动窗口不同，而是为了加速稀疏矩阵的计算BlockSolver_6_3默认了6维度的不边缘化，3自由度的三维点被边缘化，所以所有三维点都设置边缘化
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        nPoints++;

        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

        //Set edges
        // 步骤8：对每一对关联的MapPoint和KeyFrame构建边
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                // Monocular observation
                // 单目
                if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] < 0)
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId))); // 之前添加过，已确定是否固定
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->pCamera = pKFi->mpCamera;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);

                    nEdges++;
                }
                else if (leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] >= 0) // Stereo observation  双目
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);

                    nEdges++;
                }

                if (pKFi->mpCamera2)
                {
                    int rightIndex = get<1>(mit->second);

                    if (rightIndex != -1)
                    {
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double, 2, 1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->mTrl = Converter::toSE3Quat(pKFi->mTrl);

                        e->pCamera = pKFi->mpCamera2;

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKFi);
                        vpMapPointEdgeBody.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }
    }

    //Verbose::PrintMess("LM-LBA: total observations: " + to_string(vpMapPointEdgeMono.size()+vpMapPointEdgeStereo.size()), Verbose::VERBOSITY_DEBUG);

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    // 步骤9：开始优化
    optimizer.initializeOptimization();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    optimizer.optimize(5);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "LBA time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    //std::cout << "Keyframes: " << nKFs << " --- MapPoints: " << nPoints << " --- Edges: " << nEdges << endl;

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore)
    {

        // Check inlier observations
        int nMonoBadObs = 0;
        // 步骤10：检测outlier，并设置下次不优化，上面展示了怎么存储的，i是共享的，第i个边是由第i个MP与第i个KF组成的
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                // e->setLevel(1); // MODIFICATION
                nMonoBadObs++;
            }

            //e->setRobustKernel(0);
        }

        int nBodyBadObs = 0;
        for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
            MapPoint *pMP = vpMapPointEdgeBody[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                //e->setLevel(1);
                nBodyBadObs++;
            }

            //e->setRobustKernel(0);
        }

        int nStereoBadObs = 0;
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                //TODO e->setLevel(1);
                nStereoBadObs++;
            }

            //TODO e->setRobustKernel(0);
        }
        //Verbose::PrintMess("LM-LBA: First optimization has " + to_string(nMonoBadObs) + " monocular and " + to_string(nStereoBadObs) + " stereo bad observations", Verbose::VERBOSITY_DEBUG);

        // Optimize again without the outliers
        //Verbose::PrintMess("LM-LBA: second optimization", Verbose::VERBOSITY_DEBUG);
        // 步骤11：排除误差较大的outlier后再次优化，但这里没有去掉，相当于接着优化了10次，如果上面不去掉应该注释掉，浪费了计算时间
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());

    // Check inlier observations
    // 步骤12：在优化后重新计算误差，剔除连接误差比较大的关键帧和MapPoint
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = vpEdgesBody[i];
        MapPoint *pMP = vpMapPointEdgeBody[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    //Verbose::PrintMess("LM-LBA: outlier observations: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);
    bool bRedrawError = false;
    // 误差过大的数量如果超过了总数量的一半，返回，不优化了
    if (vToErase.size() >= (vpMapPointEdgeMono.size() + vpMapPointEdgeStereo.size()) * 0.5)
    {
        Verbose::PrintMess("LM-LBA: ERROR IN THE OPTIMIZATION, MOST OF THE POINTS HAS BECOME OUTLIERS", Verbose::VERBOSITY_NORMAL);

        return;
        bRedrawError = true;
        string folder_name = "test_LBA";
        string name = "_PreLM_LBA";
        //pMap->printReprojectionError(lLocalKeyFrames, pKF, name, folder_name);
        name = "_PreLM_LBA_Fixed";
        //pMap->printReprojectionError(lFixedCameras, pKF, name, folder_name);
        /*for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        Verbose::PrintMess("ERROR CHI2: " + to_string(e->chi2()) + "; DEPTH POSITIVE: " + to_string(e->isDepthPositive()), Verbose::VERBOSITY_NORMAL);
    }*/

        //return;
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // 连接偏差比较大，在关键帧中剔除对该MapPoint的观测
    // 连接偏差比较大，在MapPoint中剔除对该关键帧的观测
    if (!vToErase.empty())
    {
        map<KeyFrame *, int> mspInitialConnectedKFs;
        map<KeyFrame *, int> mspInitialObservationKFs;
        if (bRedrawError)
        {
            for (KeyFrame *pKFi : lLocalKeyFrames)
            {

                mspInitialConnectedKFs[pKFi] = pKFi->GetConnectedKeyFrames().size();
                mspInitialObservationKFs[pKFi] = pKFi->GetNumberMPs();
            }
        }

        //cout << "LM-LBA: There are " << vToErase.size() << " observations whose will be deleted from the map" << endl;
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
        // 后面做了统计输出，但做了更新连接关系
        map<KeyFrame *, int> mspFinalConnectedKFs;
        map<KeyFrame *, int> mspFinalObservationKFs;
        if (bRedrawError)
        {
            ofstream f_lba;
            f_lba.open("test_LBA/LBA_failure_KF" + to_string(pKF->mnId) + ".txt");
            f_lba << "# KF id, Initial Num CovKFs, Final Num CovKFs, Initial Num MPs, Fimal Num MPs" << endl;
            f_lba << fixed;

            for (KeyFrame *pKFi : lLocalKeyFrames)
            {
                pKFi->UpdateConnections();
                int finalNumberCovKFs = pKFi->GetConnectedKeyFrames().size();
                int finalNumberMPs = pKFi->GetNumberMPs();
                f_lba << pKFi->mnId << ", " << mspInitialConnectedKFs[pKFi] << ", " << finalNumberCovKFs << ", " << mspInitialObservationKFs[pKFi] << ", " << finalNumberMPs << endl;

                mspFinalConnectedKFs[pKFi] = finalNumberCovKFs;
                mspFinalObservationKFs[pKFi] = finalNumberMPs;
            }

            f_lba.close();
        }
    }
    // 步骤13：优化后更新关键帧位姿以及MapPoints的位置、平均观测方向等属性
    // Recover optimized data
    //Keyframes
    bool bShowStats = false;
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        cv::Mat Tiw = Converter::toCvMat(SE3quat);
        cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
        cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
        double dist = cv::norm(trasl);
        pKFi->SetPose(Converter::toCvMat(SE3quat));

        // 计算了关键帧优化前后的距离改变，后面也只是统计了一些东西，没有什么实质的改变
        if (dist > 1.0)
        {
            bShowStats = true;
            Verbose::PrintMess("LM-LBA: Too much distance in KF " + to_string(pKFi->mnId) + ", " + to_string(dist) + " meters. Current KF " + to_string(pKF->mnId), Verbose::VERBOSITY_DEBUG);
            Verbose::PrintMess("LM-LBA: Number of connections between the KFs " + to_string(pKF->GetWeight((pKFi))), Verbose::VERBOSITY_DEBUG);

            int numMonoMP = 0, numBadMonoMP = 0;
            int numStereoMP = 0, numBadStereoMP = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                if (vpEdgeKFMono[i] != pKFi)
                    continue;
                ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
                MapPoint *pMP = vpMapPointEdgeMono[i];

                if (pMP->isBad())
                    continue;

                if (e->chi2() > 5.991 || !e->isDepthPositive())
                {
                    numBadMonoMP++;
                }
                else
                {
                    numMonoMP++;
                }
            }

            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
            {
                if (vpEdgeKFStereo[i] != pKFi)
                    continue;
                g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
                MapPoint *pMP = vpMapPointEdgeStereo[i];

                if (pMP->isBad())
                    continue;

                if (e->chi2() > 7.815 || !e->isDepthPositive())
                {
                    numBadStereoMP++;
                }
                else
                {
                    numStereoMP++;
                }
            }
            Verbose::PrintMess("LM-LBA: Good observations in mono " + to_string(numMonoMP) + " and stereo " + to_string(numStereoMP), Verbose::VERBOSITY_DEBUG);
            Verbose::PrintMess("LM-LBA: Bad observations in mono " + to_string(numBadMonoMP) + " and stereo " + to_string(numBadStereoMP), Verbose::VERBOSITY_DEBUG);
        }
    }
    //Verbose::PrintMess("LM-LBA: Num fixed cameras " + to_string(num_fixedKF), Verbose::VERBOSITY_DEBUG);
    //Verbose::PrintMess("LM-LBA: Num Points " + to_string(lLocalMapPoints.size()), Verbose::VERBOSITY_DEBUG);
    //Verbose::PrintMess("LM-LBA: Num optimized cameras " + to_string(lLocalKeyFrames.size()), Verbose::VERBOSITY_DEBUG);
    //Verbose::PrintMess("----------", Verbose::VERBOSITY_DEBUG);

    //Points
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    if (bRedrawError)
    {
        string folder_name = "test_LBA";
        string name = "_PostLM_LBA";
        //pMap->printReprojectionError(lLocalKeyFrames, pKF, name, folder_name);
        name = "_PostLM_LBA_Fixed";
        //pMap->printReprojectionError(lFixedCameras, pKF, name, folder_name);
    }

    // TODO Check this changeindex
    // 当前地图改变次数+1
    pMap->IncreaseChangeIndex();
}

/**
 * @brief Local Bundle Adjustment LoopClosing::MergeLocal() 融合地图时使用，纯视觉 可以理解为跨地图的局部窗口优化
 * 优化目标： 1. vpAdjustKF; 2.vpAdjustKF与vpFixedKF对应的MP点
 * @param pMainKF        mpCurrentKF 当前关键帧
 * @param vpAdjustKF     vpLocalCurrentWindowKFs 待优化的KF
 * @param vpFixedKF      vpMergeConnectedKFs 固定的KF
 * @param pbStopFlag     false
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pMainKF, vector<KeyFrame *> vpAdjustKF, vector<KeyFrame *> vpFixedKF, bool *pbStopFlag)
{
    bool bShowImages = false;

    // 1. 构建g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>(); // 稀疏求解器

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;
    set<KeyFrame *> spKeyFrameBA; // 存放关键帧，包含固定的与不固定的

    Map *pCurrentMap = pMainKF->GetMap();

    //set<MapPoint*> sNumObsMP;
    vector<MapPoint *> vpMPs; // 存放MP

    // Set fixed KeyFrame vertices
    // 2. 构建固定关键帧的节点，并储存对应的MP
    for (KeyFrame *pKFi : vpFixedKF)
    {
        if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
        {
            Verbose::PrintMess("ERROR LBA: KF is bad or is not in the current map", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        pKFi->mnBALocalForMerge = pMainKF->mnId; // 防止重复添加

        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;

        set<MapPoint *> spViewMPs = pKFi->GetMapPoints();
        for (MapPoint *pMPi : spViewMPs)
        {
            if (pMPi)
                if (!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)

                    if (pMPi->mnBALocalForMerge != pMainKF->mnId) // 防止重复添加
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge = pMainKF->mnId;
                    }
            /*if(sNumObsMP.find(pMPi) == sNumObsMP.end())
                {
                    sNumObsMP.insert(pMPi);
                }
                else
                {
                    if(pMPi->mnBALocalForMerge!=pMainKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge=pMainKF->mnId;
                    }
                }*/
        }

        spKeyFrameBA.insert(pKFi);
    }

    //cout << "End to load Fixed KFs" << endl;

    // Set non fixed Keyframe vertices
    // 3. 构建不固定关键帧的节点，并储存对应的MP
    set<KeyFrame *> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
    for (KeyFrame *pKFi : vpAdjustKF)
    {
        if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;

        pKFi->mnBALocalForKF = pMainKF->mnId; // 防止重复添加

        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;

        set<MapPoint *> spViewMPs = pKFi->GetMapPoints();
        for (MapPoint *pMPi : spViewMPs)
        {
            if (pMPi)
            {
                if (!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)
                {
                    /*if(sNumObsMP.find(pMPi) == sNumObsMP.end())
                {
                    sNumObsMP.insert(pMPi);
                }*/
                    if (pMPi->mnBALocalForMerge != pMainKF->mnId) // 防止重复添加
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge = pMainKF->mnId;
                    }
                }
            }
        }

        spKeyFrameBA.insert(pKFi);
    }

    //Verbose::PrintMess("LBA: There are " + to_string(vpMPs.size()) + " MPs to optimize", Verbose::VERBOSITY_NORMAL);

    //cout << "End to load KFs for position adjust" << endl;
    // 准备存放边的vector
    const int nExpectedSize = (vpAdjustKF.size() + vpFixedKF.size()) * vpMPs.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    map<KeyFrame *, int> mpObsKFs;      // 统计每个关键帧对应的MP点数，调试输出用
    map<KeyFrame *, int> mpObsFinalKFs; // 统计每个MP对应的关键帧数，调试输出用
    map<MapPoint *, int> mpObsMPs;      // 统计每个MP被观测的图片数，双目就是两个，调试输出用

    // 4. 确定MP节点与边的连接
    for (unsigned int i = 0; i < vpMPs.size(); ++i)
    {
        MapPoint *pMPi = vpMPs[i];
        if (pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMPi->GetWorldPos()));
        const int id = pMPi->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, tuple<int, int>> observations = pMPi->GetObservations();
        int nEdges = 0;
        //SET EDGES
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            //cout << "--KF view init" << endl;

            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid || pKF->mnBALocalForMerge != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            //cout << "-- KF view exists" << endl;
            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];
            //cout << "-- KeyPoint loads" << endl;

            if (pKF->mvuRight[get<0>(mit->second)] < 0) //Monocular
            {
                mpObsMPs[pMPi]++;
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                //cout << "-- Sigma loads" << endl;

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);

                e->pCamera = pKF->mpCamera;
                //cout << "-- Calibration loads" << endl;

                optimizer.addEdge(e);
                //cout << "-- Edge added" << endl;

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMPi);
                //cout << "-- Added to vector" << endl;

                mpObsKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsMPs[pMPi] += 2;
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber3D);

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
            //cout << "-- End to load point" << endl;
        }
    }
    //Verbose::PrintMess("LBA: number total of edged -> " + to_string(vpEdgeKFMono.size() + vpEdgeKFStereo.size()), Verbose::VERBOSITY_NORMAL);
    // 这段没啥用，调试输出的，暂时不看
    map<int, int> mStatsObs;
    for (map<MapPoint *, int>::iterator it = mpObsMPs.begin(); it != mpObsMPs.end(); ++it)
    {
        MapPoint *pMPi = it->first;
        int numObs = it->second;

        mStatsObs[numObs]++;
        /*if(numObs < 5)
    {
        cout << "LBA: MP " << pMPi->mnId << " has " << numObs << " observations" << endl;
    }*/
    }

    /*for(map<int, int>::iterator it = mStatsObs.begin(); it != mStatsObs.end(); ++it)
{
    cout << "LBA: There are " << it->second << " MPs with " << it->first << " observations" << endl;
}*/

    //cout << "End to load MPs" << endl;

    if (pbStopFlag)
        if (*pbStopFlag)
            return;
    // 5. 优化
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    //cout << "End the first optimization" << endl;

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;
    // 6. 剔除误差大的边
    if (bDoMore)
    {
        // Check inlier observations
        int badMonoMP = 0, badStereoMP = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badMonoMP++;
            }

            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badStereoMP++;
            }

            e->setRobustKernel(0);
        }
        Verbose::PrintMess("LBA: First optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " sterero bad edges", Verbose::VERBOSITY_DEBUG);

        // Optimize again without the outliers

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        //cout << "End the second optimization (without outliers)" << endl;
    }

    // 下面这段代码都是调试用的
    //---------------------------------------------------------------------------------------------------------
    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());
    set<MapPoint *> spErasedMPs;
    set<KeyFrame *> spErasedKFs;

    // Check inlier observations
    int badMonoMP = 0, badStereoMP = 0;
    map<unsigned long int, int> mWrongObsKF;
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
            mWrongObsKF[pKFi->mnId]++;
            badMonoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
            mWrongObsKF[pKFi->mnId]++;
            badStereoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }
    Verbose::PrintMess("LBA: Second optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " sterero bad edges", Verbose::VERBOSITY_DEBUG);

    // Get Map Mutex
    unique_lock<mutex> lock(pMainKF->GetMap()->mMutexMapUpdate);

    if (!vToErase.empty())
    {
        map<KeyFrame *, int> mpMPs_in_KF;
        for (KeyFrame *pKFi : spErasedKFs)
        {
            int num_MPs = pKFi->GetMapPoints().size();
            mpMPs_in_KF[pKFi] = num_MPs;
        }

        Verbose::PrintMess("LBA: There are " + to_string(vToErase.size()) + " observations whose will be deleted from the map", Verbose::VERBOSITY_DEBUG);
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }

        Verbose::PrintMess("LBA: " + to_string(spErasedMPs.size()) + " MPs had deleted observations", Verbose::VERBOSITY_DEBUG);
        Verbose::PrintMess("LBA: Current map is " + to_string(pMainKF->GetMap()->GetId()), Verbose::VERBOSITY_DEBUG);
        int numErasedMP = 0;
        for (MapPoint *pMPi : spErasedMPs)
        {
            if (pMPi->isBad())
            {
                Verbose::PrintMess("LBA: MP " + to_string(pMPi->mnId) + " has lost almost all the observations, its origin map is " + to_string(pMPi->mnOriginMapId), Verbose::VERBOSITY_DEBUG);
                numErasedMP++;
            }
        }
        Verbose::PrintMess("LBA: " + to_string(numErasedMP) + " MPs had deleted from the map", Verbose::VERBOSITY_DEBUG);

        for (KeyFrame *pKFi : spErasedKFs)
        {
            int num_MPs = pKFi->GetMapPoints().size();
            int num_init_MPs = mpMPs_in_KF[pKFi];
            Verbose::PrintMess("LBA: Initially KF " + to_string(pKFi->mnId) + " had " + to_string(num_init_MPs) + ", at the end has " + to_string(num_MPs), Verbose::VERBOSITY_DEBUG);
        }
    }
    for (unsigned int i = 0; i < vpMPs.size(); ++i)
    {
        MapPoint *pMPi = vpMPs[i];
        if (pMPi->isBad())
            continue;

        const map<KeyFrame *, tuple<int, int>> observations = pMPi->GetObservations();
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            //cout << "--KF view init" << endl;

            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid || pKF->mnBALocalForKF != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];
            //cout << "-- KeyPoint loads" << endl;

            if (pKF->mvuRight[get<0>(mit->second)] < 0) //Monocular
            {
                mpObsFinalKFs[pKF]++;
            }
            else // RGBD or Stereo
            {

                mpObsFinalKFs[pKF]++;
            }
            //cout << "-- End to load point" << endl;
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //cout << "End to erase observations" << endl;

    // Recover optimized data
    // 7. 取出结果
    //Keyframes
    for (KeyFrame *pKFi : vpAdjustKF)
    {
        if (pKFi->isBad())
            continue;
        // 7.1 取出对应位姿，并计算t的变化量。
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        cv::Mat Tiw = Converter::toCvMat(SE3quat);
        cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
        cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
        double dist = cv::norm(trasl); // 平方和再开方

        // 统计调试用
        int numMonoBadPoints = 0, numMonoOptPoints = 0;
        int numStereoBadPoints = 0, numStereoOptPoints = 0;

        vector<MapPoint *> vpMonoMPsOpt, vpStereoMPsOpt; // 存放mp内点
        vector<MapPoint *> vpMonoMPsBad, vpStereoMPsBad; // 存放mp外点
        // 7.2 卡方检验
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];
            KeyFrame *pKFedge = vpEdgeKFMono[i];

            if (pKFi != pKFedge)
            {
                continue;
            }

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                numMonoBadPoints++;
                vpMonoMPsBad.push_back(pMP);
            }
            else
            {
                numMonoOptPoints++;
                vpMonoMPsOpt.push_back(pMP);
            }
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];
            KeyFrame *pKFedge = vpEdgeKFMono[i];

            if (pKFi != pKFedge)
            {
                continue;
            }

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                numStereoBadPoints++;
                vpStereoMPsBad.push_back(pMP);
            }
            else
            {
                numStereoOptPoints++;
                vpStereoMPsOpt.push_back(pMP);
            }
        }

        if (numMonoOptPoints + numStereoOptPoints < 50)
        {
            Verbose::PrintMess("LBA ERROR: KF " + to_string(pKFi->mnId) + " has only " + to_string(numMonoOptPoints) + " monocular and " + to_string(numStereoOptPoints) + " stereo points", Verbose::VERBOSITY_DEBUG);
        }
        if (dist > 1.0) // 单位？？？1m？？？ 下面这段是画图输出用的暂时不看
        {
            if (bShowImages)
            {
                string strNameFile = pKFi->mNameFile;
                cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

                cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

                int numPointsMono = 0, numPointsStereo = 0;
                int numPointsMonoBad = 0, numPointsStereoBad = 0;
                for (int i = 0; i < vpMonoMPsOpt.size(); ++i)
                {
                    if (!vpMonoMPsOpt[i] || vpMonoMPsOpt[i]->isBad())
                    {
                        continue;
                    }
                    int index = get<0>(vpMonoMPsOpt[i]->GetIndexInKeyFrame(pKFi));
                    if (index < 0)
                    {
                        //cout << "LBA ERROR: KF has a monocular observation which is not recognized by the MP" << endl;
                        //cout << "LBA: KF " << pKFi->mnId << " and MP " << vpMonoMPsOpt[i]->mnId << " with index " << endl;
                        continue;
                    }

                    //string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                    cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(255, 0, 0));
                    //cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    numPointsMono++;
                }

                for (int i = 0; i < vpStereoMPsOpt.size(); ++i)
                {
                    if (!vpStereoMPsOpt[i] || vpStereoMPsOpt[i]->isBad())
                    {
                        continue;
                    }
                    int index = get<0>(vpStereoMPsOpt[i]->GetIndexInKeyFrame(pKFi));
                    if (index < 0)
                    {
                        //cout << "LBA: KF has a stereo observation which is not recognized by the MP" << endl;
                        //cout << "LBA: KF " << pKFi->mnId << " and MP " << vpStereoMPsOpt[i]->mnId << endl;
                        continue;
                    }

                    //string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                    cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(0, 255, 0));
                    //cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    numPointsStereo++;
                }

                for (int i = 0; i < vpMonoMPsBad.size(); ++i)
                {
                    if (!vpMonoMPsBad[i] || vpMonoMPsBad[i]->isBad())
                    {
                        continue;
                    }
                    int index = get<0>(vpMonoMPsBad[i]->GetIndexInKeyFrame(pKFi));
                    if (index < 0)
                    {
                        //cout << "LBA ERROR: KF has a monocular observation which is not recognized by the MP" << endl;
                        //cout << "LBA: KF " << pKFi->mnId << " and MP " << vpMonoMPsOpt[i]->mnId << " with index " << endl;
                        continue;
                    }

                    //string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                    cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(0, 0, 255));
                    //cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    numPointsMonoBad++;
                }
                for (int i = 0; i < vpStereoMPsBad.size(); ++i)
                {
                    if (!vpStereoMPsBad[i] || vpStereoMPsBad[i]->isBad())
                    {
                        continue;
                    }
                    int index = get<0>(vpStereoMPsBad[i]->GetIndexInKeyFrame(pKFi));
                    if (index < 0)
                    {
                        //cout << "LBA: KF has a stereo observation which is not recognized by the MP" << endl;
                        //cout << "LBA: KF " << pKFi->mnId << " and MP " << vpStereoMPsOpt[i]->mnId << endl;
                        continue;
                    }

                    //string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                    cv::circle(imLeft, pKFi->mvKeys[index].pt, 2, cv::Scalar(0, 0, 255));
                    //cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    numPointsStereoBad++;
                }

                string namefile = "./test_LBA/LBA_KF" + to_string(pKFi->mnId) + "_" + to_string(numPointsMono + numPointsStereo) + "_D" + to_string(dist) + ".png";
                cv::imwrite(namefile, imLeft);

                Verbose::PrintMess("--LBA in KF " + to_string(pKFi->mnId), Verbose::VERBOSITY_DEBUG);
                Verbose::PrintMess("--Distance: " + to_string(dist) + " meters", Verbose::VERBOSITY_DEBUG);
                Verbose::PrintMess("--Number of observations: " + to_string(numMonoOptPoints) + " in mono and " + to_string(numStereoOptPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
                Verbose::PrintMess("--Number of discarded observations: " + to_string(numMonoBadPoints) + " in mono and " + to_string(numStereoBadPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
                Verbose::PrintMess("--To much distance correction in LBA: It has " + to_string(mpObsKFs[pKFi]) + " observated MPs", Verbose::VERBOSITY_DEBUG);
                Verbose::PrintMess("--To much distance correction in LBA: It has " + to_string(mpObsFinalKFs[pKFi]) + " deleted observations", Verbose::VERBOSITY_DEBUG);
                Verbose::PrintMess("--------", Verbose::VERBOSITY_DEBUG);
            }
        }
        pKFi->SetPose(Tiw); // 赋值
    }
    //cout << "End to update the KeyFrames" << endl;

    //Points
    // 7.3 取出MP优化结果
    for (MapPoint *pMPi : vpMPs)
    {
        if (pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMPi->mnId + maxKFid + 1));
        pMPi->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMPi->UpdateNormalAndDepth();
    }
    //cout << "End to update MapPoint" << endl;
}

// TODO
/**
 * @brief  LoopClosing::CorrectLoop() 回环矫正时使用，纯视觉，全局本质图优化，有严重BUG，经测试差了一个尺度！
 * 优化目标： 地图中所有MP与关键帧
 * @param pMap                当前的map
 * @param pLoopKF             mpLoopMatchedKF 与 mpCurrentKF 匹配的关键帧
 * @param pCurKF              mpCurrentKF 当前关键帧
 * @param NonCorrectedSim3    通过pKFi->GetPose()计算的放NonCorrectedSim3也就是回环前的位姿 这里面的帧只是与mpCurrentKF相关联的
 * @param CorrectedSim3       通过mg2oLoopScw计算的放CorrectedSim3 这里面的帧只是与mpCurrentKF相关联的
 * @param LoopConnections     因为回环而建立的新的帧与帧的连接关系，里面的key 全部为pCurKF的共视帧与其本身
 * @param bFixScale           false
 */
void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                        const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                        const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                        const map<KeyFrame *, set<KeyFrame *>> &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    // 1. 构建优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    // 7表示位姿是sim3  3表示三维点坐标维度
    g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
        new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames(); // 所有关键帧
    const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints(); // 所有mp

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);          // 存放每一帧优化前的sim3
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1); // 存放每一帧优化后的sim3，修正mp位姿用
    vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);                           // 存放节点，没用，还占地方

    vector<Eigen::Vector3d> vZvectors(nMaxKFid + 1); // For debugging 调试用的参数
    Eigen::Vector3d z_vec;                           // 调试用，暂时不去管
    z_vec << 0.0, 0.0, 1.0;

    const int minFeat = 100; // MODIFICATION originally was set to 100 本质图的权重

    // Set KeyFrame vertices
    // 2. 关键帧节点
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if (it != CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            // 没有在CorrectedSim3里面找到表示与pCurKF并不关联，也就是离得远，并没有经过计算得到一个初始值，这里直接使用原始的位置
            // TODO 是不是可以给他简单修改一下增加精度？应该做下测试看看这个修改是否多余
            Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw, tcw, 1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }
        // 固定第一帧
        if (pKF->mnId == pMap->GetInitKFid())
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);
        vZvectors[nIDi] = vScw[nIDi].rotation().toRotationMatrix() * z_vec; // For debugging

        vpVertices[nIDi] = VSim3;
    }

    set<pair<long unsigned int, long unsigned int>> sInsertedEdges; // 里面保存的是因为回环而新建立的连接关系

    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

    // Set Loop edges
    // 3. 添加边：LoopConnections是闭环时因为MapPoints调整而出现的新关键帧连接关系（包括当前帧与闭环匹配帧之间的连接关系）
    // TODO 验证下上面的说法
    int count_loop = 0;
    for (map<KeyFrame *, set<KeyFrame *>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
    {
        // 3.1 取出帧与帧们
        KeyFrame *pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame *> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi]; // 优化前的位姿
        const g2o::Sim3 Swi = Siw.inverse();

        for (set<KeyFrame *>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            // 这里的约束有点意思，对于每一个连接，只要是存在pCurKF或者pLoopKF 那这个连接不管共视了多少MP都优化
            // 反之没有的话共视度要大于100 构建本质图
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            // 得到两个pose间的Sim3变换
            const g2o::Sim3 Sji = Sjw * Swi; // 优化前他们的相对位姿

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            // BUG 有点问题，vScw里面包含了两类，其中一类是校正前的位姿，以这个为标准？？这不仅不是优化不出的问题，还引入了大量误差
            // 是不是利用上面的想法给做个赋值
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);
            count_loop++;
            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    int count_spa_tree = 0;
    int count_cov = 0;
    int count_imu = 0;
    int count_kf = 0;
    // Set normal edges
    // 4. 添加跟踪时形成的边、闭环匹配成功形成的边
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        count_kf = 0;
        KeyFrame *pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi; // 校正前的sim3

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);
        // 找到的话说明是关键帧的共视帧，没找到表示非共视帧，非共视帧vScw[nIDi]里面装的都是矫正前的
        // 所以不管怎样说 Swi都是校正前的
        if (iti != NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame *pParentKF = pKF->GetParent();

        // Spanning tree edge
        // 4.1 只添加扩展树的边（有父关键帧）
        if (pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            // 尽可能得到未经过Sim3传播调整的位姿
            if (itj != NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];
            // 又是未校正的结果作为观测值
            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);
            count_kf++;
            count_spa_tree++;
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        // 4.2 添加在CorrectLoop函数中AddLoopEdge函数添加的闭环连接边（当前帧与闭环匹配帧之间的连接关系）
        // 使用经过Sim3调整前关键帧之间的相对关系作为边
        const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
        for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            KeyFrame *pLKF = *sit;
            if (pLKF->mnId < pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                // 尽可能得到未经过Sim3传播调整的位姿
                if (itl != NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                // BUG 根据两个Pose顶点的位姿算出相对位姿作为边，那还存在误差？优化有用？（wubo???）
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
                count_kf++;
                count_loop++;
            }
        }

        // Covisibility graph edges
        // 4.3 最有很好共视关系的关键帧也作为边进行优化
        // 使用经过Sim3调整前关键帧之间的相对关系作为边
        const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            KeyFrame *pKFn = *vit;
            if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn)) // 排除上面添加过的帧
            {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                {
                    if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId)))) // 排除因为回环新添加的连接，因为已经添加过
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    // 尽可能得到未经过Sim3传播调整的位姿
                    if (itn != NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                    count_kf++;
                    count_cov++;
                }
            }
        }

        // Inertial edges if inertial
        // 如果是imu的话还会找前一帧做优化，那不就是所有帧全部搞到一起优化了
        // 不过这个函数不就是非imu么，这代码漏洞真多
        if (pKF->bImu && pKF->mPrevKF)
        {
            g2o::Sim3 Spw;
            LoopClosing::KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
            if (itp != NonCorrectedSim3.end())
                Spw = itp->second;
            else
                Spw = vScw[pKF->mPrevKF->mnId];

            g2o::Sim3 Spi = Spw * Swi;
            g2o::EdgeSim3 *ep = new g2o::EdgeSim3();
            ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mPrevKF->mnId)));
            ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            ep->setMeasurement(Spi);
            ep->information() = matLambda;
            optimizer.addEdge(ep);
            count_kf++;
            count_imu++;
        }
        /*if(count_kf<3)
        cout << "EG: kf with " << count_kf << " edges!!    ID: " << pKF->mnId << endl;*/
    }

    //cout << "EG: Number of KFs: " << vpKFs.size() << endl;
    //cout << "EG: spaning tree edges: " << count_spa_tree << endl;
    //cout << "EG: Loop edges: " << count_loop << endl;
    //cout << "EG: covisible edges: " << count_cov << endl;
    //cout << "EG: imu edges: " << count_imu << endl;
    // Optimize!
    // 5. 开始g2o优化
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err0 = optimizer.activeRobustChi2();
    optimizer.optimize(20);
    optimizer.computeActiveErrors();
    float errEnd = optimizer.activeRobustChi2();
    //cout << "err_0/err_end: " << err0 << "/" << errEnd << endl;
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 6. 设定优化后的位姿
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw = VSim3->estimate();
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *= (1. / s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

        pKFi->SetPose(Tiw);
        // cout << "angle KF " << nIDi << ": " << (180.0/3.1415)*acos(vZvectors[nIDi].dot(eigR*z_vec)) << endl;
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    // 7. 步骤5和步骤6优化得到关键帧的位姿后，MapPoints根据参考帧优化前后的相对关系调整自己的位置
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMPs[i];

        if (pMP->isBad())
            continue;

        int nIDr;
        // 该MapPoint经过Sim3调整过，(LoopClosing.cpp，CorrectLoop函数，步骤2.2_
        if (pMP->mnCorrectedByKF == pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            // 通过情况下MapPoint的参考关键帧就是创建该MapPoint的那个关键帧
            KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        // 得到MapPoint参考关键帧步骤5优化前的位姿
        g2o::Sim3 Srw = vScw[nIDr];
        // 得到MapPoint参考关键帧优化后的位姿
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }

    // TODO Check this changeindex
    pMap->IncreaseChangeIndex();
}

/**
 * @brief  LoopClosing::CorrectLoop() 回环矫正时使用，IMU加视觉，全局本质图优化，流程基本与上面OptimizeEssentialGraph一模一样，同样有严重BUG
 * 优化目标： 地图中所有MP与关键帧
 * @param pMap                当前的map
 * @param pLoopKF             mpLoopMatchedKF 与 mpCurrentKF 匹配的关键帧
 * @param pCurKF              mpCurrentKF 当前关键帧
 * @param NonCorrectedSim3    通过pKFi->GetPose()计算的放NonCorrectedSim3也就是回环前的位姿 这里面的帧只是与mpCurrentKF相关联的
 * @param CorrectedSim3       通过mg2oLoopScw计算的放CorrectedSim3 这里面的帧只是与mpCurrentKF相关联的
 * @param LoopConnections     因为回环而建立的新的帧与帧的连接关系，里面的key 全部为pCurKF的共视帧与其本身
 */
void Optimizer::OptimizeEssentialGraph4DoF(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                            const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                            const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                            const map<KeyFrame *, set<KeyFrame *>> &LoopConnections)
{
    // typedef g2o::BlockSolver< g2o::BlockSolverTraits<4, 4> > BlockSolver_4_4;  // 没用到，注释了

    // Setup optimizer
    // 1. 构建优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolverX::LinearSolverType *linearSolver =
        new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames(); // 所有关键帧
    const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints(); // 所有mp

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);          // 存放每一帧优化前的sim3
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1); // 存放每一帧优化后的sim3，修正mp位姿用

    vector<VertexPose4DoF *> vpVertices(nMaxKFid + 1);

    const int minFeat = 100; // 100 本质图的权重
    // Set KeyFrame vertices
    // 2. 关键帧节点
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        // 自定义的一个优化4自由度的节点
        VertexPose4DoF *V4DoF;

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if (it != CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            const g2o::Sim3 Swc = it->second.inverse();
            Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
            Eigen::Vector3d twc = Swc.translation();
            V4DoF = new VertexPose4DoF(Rwc, twc, pKF);
        }
        else
        {
            // 没有在CorrectedSim3里面找到表示与pCurKF并不关联，也就是离得远，并没有经过计算得到一个初始值，这里直接使用原始的位置
            // TODO 是不是可以给他简单修改一下增加精度？应该做下测试看看这个修改是否多余
            Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw, tcw, 1.0);
            vScw[nIDi] = Siw;
            V4DoF = new VertexPose4DoF(pKF);
        }

        // 固定回环帧
        if (pKF == pLoopKF)
            V4DoF->setFixed(true);

        V4DoF->setId(nIDi);
        V4DoF->setMarginalized(false);

        optimizer.addVertex(V4DoF);
        vpVertices[nIDi] = V4DoF;
    }
    cout << "PoseGraph4DoF: KFs loaded" << endl;

    set<pair<long unsigned int, long unsigned int>> sInsertedEdges; // 里面保存的是因为回环而新建立的连接关系

    // Edge used in posegraph has still 6Dof, even if updates of camera poses are just in 4DoF
    Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();
    matLambda(0, 0) = 1e3;
    matLambda(1, 1) = 1e3;
    matLambda(0, 0) = 1e3;

    // Set Loop edges
    // 3. 添加边：LoopConnections是闭环时因为MapPoints调整而出现的新关键帧连接关系（包括当前帧与闭环匹配帧之间的连接关系）
    // TODO 验证下上面的说法
    Edge4DoF *e_loop;
    for (map<KeyFrame *, set<KeyFrame *>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
    {
        // 3.1 取出帧与帧们
        KeyFrame *pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame *> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi]; // 优化前的位姿
        const g2o::Sim3 Swi = Siw.inverse();

        for (set<KeyFrame *>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            // 这里的约束有点意思，对于每一个连接，只要是存在pCurKF或者pLoopKF 那这个连接不管共视了多少MP都优化
            // 反之没有的话共视度要大于100 构建本质图
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            // 得到两个pose间的Sim3变换
            const g2o::Sim3 Sij = Siw * Sjw.inverse();
            Eigen::Matrix4d Tij;
            Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
            Tij.block<3, 1>(0, 3) = Sij.translation();
            Tij(3, 3) = 1.;

            // BUG 有点问题，vScw里面包含了两类，其中一类是校正前的位姿，以这个为标准？？这不仅不是优化不出的问题，还引入了大量误差
            // 是不是利用上面的想法给做个赋值
            Edge4DoF *e = new Edge4DoF(Tij);
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));

            e->information() = matLambda;
            e_loop = e;
            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }
    cout << "PoseGraph4DoF: Loop edges loaded" << endl;

    // 1. Set normal edges
    // 4. 添加跟踪时形成的边、闭环匹配成功形成的边
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame *pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Siw;

        // Use noncorrected poses for posegraph edges
        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);
        // 找到的话说明是关键帧的共视帧，没找到表示非共视帧，非共视帧vScw[nIDi]里面装的都是矫正前的
        // 所以不管怎样说 Swi都是校正前的
        if (iti != NonCorrectedSim3.end())
            Siw = iti->second;
        else
            Siw = vScw[nIDi];

        // 1.1.0 Spanning tree edge
        // 4.1 只添加扩展树的边（有父关键帧） 这里并没有父帧
        KeyFrame *pParentKF = static_cast<KeyFrame *>(NULL);
        if (pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            // 尽可能得到未经过Sim3传播调整的位姿
            if (itj != NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj = vScw[nIDj].inverse();

            // 又是未校正的结果作为观测值
            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
            Tij.block<3, 1>(0, 3) = Sij.translation();
            Tij(3, 3) = 1.;

            Edge4DoF *e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.1.1 Inertial edges
        // 代替父帧的是利用mPrevKF，流程与上面一样
        KeyFrame *prevKF = pKF->mPrevKF;
        if (prevKF)
        {
            int nIDj = prevKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(prevKF);

            if (itj != NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj = vScw[nIDj].inverse();

            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
            Tij.block<3, 1>(0, 3) = Sij.translation();
            Tij(3, 3) = 1.;

            Edge4DoF *e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.2 Loop edges
        // 4.2 添加在CorrectLoop函数中AddLoopEdge函数添加的闭环连接边（当前帧与闭环匹配帧之间的连接关系）
        // 使用经过Sim3调整前关键帧之间的相对关系作为边
        const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
        for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            KeyFrame *pLKF = *sit;
            if (pLKF->mnId < pKF->mnId)
            {
                g2o::Sim3 Swl;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if (itl != NonCorrectedSim3.end())
                    Swl = itl->second.inverse();
                else
                    Swl = vScw[pLKF->mnId].inverse();

                g2o::Sim3 Sil = Siw * Swl;
                Eigen::Matrix4d Til;
                Til.block<3, 3>(0, 0) = Sil.rotation().toRotationMatrix();
                Til.block<3, 1>(0, 3) = Sil.translation();
                Til(3, 3) = 1.;

                // BUG 根据两个Pose顶点的位姿算出相对位姿作为边，那还存在误差？优化有用？
                Edge4DoF *e = new Edge4DoF(Til);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                e->information() = matLambda;
                optimizer.addEdge(e);
            }
        }

        // 1.3 Covisibility graph edges
        // 4.3 最有很好共视关系的关键帧也作为边进行优化
        // 使用经过Sim3调整前关键帧之间的相对关系作为边
        const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            KeyFrame *pKFn = *vit;
            if (pKFn && pKFn != pParentKF && pKFn != prevKF && pKFn != pKF->mNextKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                {
                    if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId))))
                        continue;

                    g2o::Sim3 Swn;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if (itn != NonCorrectedSim3.end())
                        Swn = itn->second.inverse();
                    else
                        Swn = vScw[pKFn->mnId].inverse();

                    g2o::Sim3 Sin = Siw * Swn;
                    Eigen::Matrix4d Tin;
                    Tin.block<3, 3>(0, 0) = Sin.rotation().toRotationMatrix();
                    Tin.block<3, 1>(0, 3) = Sin.translation();
                    Tin(3, 3) = 1.;
                    Edge4DoF *e = new Edge4DoF(Tin);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                    e->information() = matLambda;
                    optimizer.addEdge(e);
                }
            }
        }
    }
    cout << "PoseGraph4DoF: Covisibility edges loaded" << endl;
    // 5. 开始g2o优化
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 6. 设定优化后的位姿
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        VertexPose4DoF *Vi = static_cast<VertexPose4DoF *>(optimizer.vertex(nIDi));
        Eigen::Matrix3d Ri = Vi->estimate().Rcw[0];
        Eigen::Vector3d ti = Vi->estimate().tcw[0];

        g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri, ti, 1.);
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();

        cv::Mat Tiw = Converter::toCvSE3(Ri, ti);
        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    // 7. 步骤5和步骤6优化得到关键帧的位姿后，MapPoints根据参考帧优化前后的相对关系调整自己的位置
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMPs[i];

        if (pMP->isBad())
            continue;

        int nIDr;

        KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
        nIDr = pRefKF->mnId;

        // 得到MapPoint参考关键帧步骤5优化前的位姿
        g2o::Sim3 Srw = vScw[nIDr];
        // 得到MapPoint参考关键帧优化后的位姿
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
    pMap->IncreaseChangeIndex();
}

/**
 * @brief 没有使用，暂时不看
 */
void Optimizer::OptimizeEssentialGraph6DoF(KeyFrame *pCurKF, vector<KeyFrame *> &vpFixedKFs, vector<KeyFrame *> &vpFixedCorrectedKFs,
                                            vector<KeyFrame *> &vpNonFixedKFs, vector<MapPoint *> &vpNonCorrectedMPs, double scale)
{
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedKFs.size()) + " KFs fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map", Verbose::VERBOSITY_DEBUG);

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver =
        new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    Map *pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vScw(nMaxKFid + 1);
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vScw_bef(nMaxKFid + 1);
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vCorrectedSwc(nMaxKFid + 1);
    vector<g2o::VertexSE3Expmap *> vpVertices(nMaxKFid + 1);
    vector<bool> vbFromOtherMap(nMaxKFid + 1);

    const int minFeat = 100;

    for (KeyFrame *pKFi : vpFixedKFs)
    {
        if (pKFi->isBad())
            continue;

        g2o::VertexSE3Expmap *VSE3 = new g2o::VertexSE3Expmap();

        const int nIDi = pKFi->mnId;

        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
        g2o::SE3Quat Siw(Rcw, tcw);
        vScw[nIDi] = Siw;
        vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
        VSE3->setEstimate(Siw);

        VSE3->setFixed(true);

        VSE3->setId(nIDi);
        VSE3->setMarginalized(false);
        //VSim3->_fix_scale = true; //TODO
        vbFromOtherMap[nIDi] = false;

        optimizer.addVertex(VSE3);

        vpVertices[nIDi] = VSE3;
    }
    cout << "Opt_Essential: vpFixedKFs loaded" << endl;

    set<unsigned long> sIdKF;
    for (KeyFrame *pKFi : vpFixedCorrectedKFs)
    {
        if (pKFi->isBad())
            continue;

        g2o::VertexSE3Expmap *VSE3 = new g2o::VertexSE3Expmap();

        const int nIDi = pKFi->mnId;

        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
        g2o::SE3Quat Siw(Rcw, tcw);
        vScw[nIDi] = Siw;
        vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
        VSE3->setEstimate(Siw);

        cv::Mat Tcw_bef = pKFi->mTcwBefMerge;
        Eigen::Matrix<double, 3, 3> Rcw_bef = Converter::toMatrix3d(Tcw_bef.rowRange(0, 3).colRange(0, 3));
        Eigen::Matrix<double, 3, 1> tcw_bef = Converter::toVector3d(Tcw_bef.rowRange(0, 3).col(3)) / scale;
        vScw_bef[nIDi] = g2o::SE3Quat(Rcw_bef, tcw_bef);

        VSE3->setFixed(true);

        VSE3->setId(nIDi);
        VSE3->setMarginalized(false);
        //VSim3->_fix_scale = true;
        vbFromOtherMap[nIDi] = true;

        optimizer.addVertex(VSE3);

        vpVertices[nIDi] = VSE3;

        sIdKF.insert(nIDi);
    }
    Verbose::PrintMess("Opt_Essential: vpFixedCorrectedKFs loaded", Verbose::VERBOSITY_DEBUG);

    for (KeyFrame *pKFi : vpNonFixedKFs)
    {
        if (pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        if (sIdKF.count(nIDi)) // It has already added in the corrected merge KFs
            continue;

        g2o::VertexSE3Expmap *VSE3 = new g2o::VertexSE3Expmap();

        //cv::Mat Tcw = pKFi->mTcwBefMerge;
        //Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(Tcw.rowRange(0, 3).colRange(0, 3));
        //Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(Tcw.rowRange(0, 3).col(3));
        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation()) / scale;
        g2o::SE3Quat Siw(Rcw, tcw);
        vScw_bef[nIDi] = Siw;
        VSE3->setEstimate(Siw);

        VSE3->setFixed(false);

        VSE3->setId(nIDi);
        VSE3->setMarginalized(false);
        //VSim3->_fix_scale = true;
        vbFromOtherMap[nIDi] = true;

        optimizer.addVertex(VSE3);

        vpVertices[nIDi] = VSE3;

        sIdKF.insert(nIDi);
    }
    Verbose::PrintMess("Opt_Essential: vpNonFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

    vector<KeyFrame *> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(), vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
    set<KeyFrame *> spKFs(vpKFs.begin(), vpKFs.end());

    Verbose::PrintMess("Opt_Essential: List of KF loaded", Verbose::VERBOSITY_DEBUG);

    const Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();

    for (KeyFrame *pKFi : vpKFs)
    {
        int num_connections = 0;
        const int nIDi = pKFi->mnId;

        g2o::SE3Quat Swi = vScw[nIDi].inverse();
        g2o::SE3Quat Swi_bef;
        if (vbFromOtherMap[nIDi])
        {
            Swi_bef = vScw_bef[nIDi].inverse();
        }
        /*if(pKFi->mnMergeCorrectedForKF == pCurKF->mnId)
    {
            Swi = vScw[nIDi].inverse();
    }
    else
    {
        cv::Mat Twi = pKFi->mTwcBefMerge;
        Swi = g2o::Sim3(Converter::toMatrix3d(Twi.rowRange(0, 3).colRange(0, 3)), 
                        Converter::toVector3d(Twi.rowRange(0, 3).col(3)), 1.0);
    }*/

        KeyFrame *pParentKFi = pKFi->GetParent();

        // Spanning tree edge
        if (pParentKFi && spKFs.find(pParentKFi) != spKFs.end())
        {
            int nIDj = pParentKFi->mnId;

            g2o::SE3Quat Sjw = vScw[nIDj];
            g2o::SE3Quat Sjw_bef;
            if (vbFromOtherMap[nIDj])
            {
                Sjw_bef = vScw_bef[nIDj];
            }

            /*if(pParentKFi->mnMergeCorrectedForKF == pCurKF->mnId)
        {
                Sjw =  vScw[nIDj];
        }
        else
        {
            cv::Mat Tjw = pParentKFi->mTcwBefMerge;
            Sjw = g2o::Sim3(Converter::toMatrix3d(Tjw.rowRange(0, 3).colRange(0, 3)), 
                            Converter::toVector3d(Tjw.rowRange(0, 3).col(3)), 1.0);
        }*/

            g2o::SE3Quat Sji;

            if (vbFromOtherMap[nIDi] && vbFromOtherMap[nIDj])
            {
                Sji = Sjw_bef * Swi_bef;
            }
            else
            {
                Sji = Sjw * Swi;
            }

            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
            num_connections++;
        }

        // Loop edges
        const set<KeyFrame *> sLoopEdges = pKFi->GetLoopEdges();
        for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            KeyFrame *pLKF = *sit;
            if (spKFs.find(pLKF) != spKFs.end() && pLKF->mnId < pKFi->mnId)
            {
                g2o::SE3Quat Slw = vScw[pLKF->mnId];
                g2o::SE3Quat Slw_bef;
                if (vbFromOtherMap[pLKF->mnId])
                {
                    Slw_bef = vScw_bef[pLKF->mnId];
                }

                /*if(pLKF->mnMergeCorrectedForKF == pCurKF->mnId)
            {
                    Slw = vScw[pLKF->mnId];
            }
            else
            {
                cv::Mat Tlw = pLKF->mTcwBefMerge;
                Slw = g2o::Sim3(Converter::toMatrix3d(Tlw.rowRange(0, 3).colRange(0, 3)), 
                                Converter::toVector3d(Tlw.rowRange(0, 3).col(3)), 1.0);
            }*/

                g2o::SE3Quat Sli;

                if (vbFromOtherMap[nIDi] && vbFromOtherMap[pLKF->mnId])
                {
                    Sli = Slw_bef * Swi_bef;
                }
                else
                {
                    Sli = Slw * Swi;
                }

                g2o::EdgeSE3 *el = new g2o::EdgeSE3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
                num_connections++;
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame *> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            KeyFrame *pKFn = *vit;
            if (pKFn && pKFn != pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn) && spKFs.find(pKFn) != spKFs.end())
            {
                if (!pKFn->isBad() && pKFn->mnId < pKFi->mnId)
                {
                    g2o::SE3Quat Snw = vScw[pKFn->mnId];

                    g2o::SE3Quat Snw_bef;
                    if (vbFromOtherMap[pKFn->mnId])
                    {
                        Snw_bef = vScw_bef[pKFn->mnId];
                    }
                    /*if(pKFn->mnMergeCorrectedForKF == pCurKF->mnId)
                {
                    Snw = vScw[pKFn->mnId];
                }
                else
                {
                    cv::Mat Tnw = pKFn->mTcwBefMerge;
                    Snw = g2o::Sim3(Converter::toMatrix3d(Tnw.rowRange(0, 3).colRange(0, 3)), 
                                    Converter::toVector3d(Tnw.rowRange(0, 3).col(3)), 1.0);
                }*/

                    g2o::SE3Quat Sni;

                    if (vbFromOtherMap[nIDi] && vbFromOtherMap[pKFn->mnId])
                    {
                        Sni = Snw_bef * Swi_bef;
                    }
                    else
                    {
                        Sni = Snw * Swi;
                    }

                    g2o::EdgeSE3 *en = new g2o::EdgeSE3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                    num_connections++;
                }
            }
        }

        if (num_connections == 0)
        {
            Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections", Verbose::VERBOSITY_DEBUG);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    Verbose::PrintMess("Opt_Essential: Finish the optimization", Verbose::VERBOSITY_DEBUG);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    Verbose::PrintMess("Opt_Essential: Apply the new pose to the KFs", Verbose::VERBOSITY_DEBUG);
    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (KeyFrame *pKFi : vpNonFixedKFs)
    {
        if (pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        g2o::VertexSE3Expmap *VSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(nIDi));
        g2o::SE3Quat CorrectedSiw = VSE3->estimate();
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        //double s = CorrectedSiw.scale();

        //eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

        /*{
        cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
        cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
        double dist = cv::norm(trasl);
        if(dist > 1.0)
        {
            cout << "--Distance: " << dist << " meters" << endl;
            cout << "--To much distance correction in EssentGraph: It has connected " << pKFi->GetVectorCovisibleKeyFrames().size() << " KFs" << endl;
        }

        string strNameFile = pKFi->mNameFile;
        cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

        cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();
        for(int j=0; j<vpMapPointsKFi.size(); ++j)
        {
            if(!vpMapPointsKFi[j] || vpMapPointsKFi[j]->isBad())
            {
                continue;
            }
            string strNumOBs = to_string(vpMapPointsKFi[j]->Observations());
            cv::circle(imLeft, pKFi->mvKeys[j].pt, 2, cv::Scalar(0, 255, 0));
            cv::putText(imLeft, strNumOBs, pKFi->mvKeys[j].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
        }

        string namefile = "./test_OptEssent/Essent_" + to_string(pCurKF->mnId) + "_KF" + to_string(pKFi->mnId) +"_D" + to_string(dist) +".png";
        cv::imwrite(namefile, imLeft);
    }*/

        pKFi->mTcwBefMerge = pKFi->GetPose();
        pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
        pKFi->SetPose(Tiw);
    }

    Verbose::PrintMess("Opt_Essential: Apply the new pose to the MPs", Verbose::VERBOSITY_DEBUG);
    cout << "Opt_Essential: number of points -> " << vpNonCorrectedMPs.size() << endl;
    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for (MapPoint *pMPi : vpNonCorrectedMPs)
    {
        if (pMPi->isBad())
            continue;

        //Verbose::PrintMess("Opt_Essential: MP id " + to_string(pMPi->mnId), Verbose::VERBOSITY_DEBUG);
        /*int nIDr;
    if(pMPi->mnCorrectedByKF==pCurKF->mnId)
    {
        nIDr = pMPi->mnCorrectedReference;
    }
    else
    {

    }*/
        KeyFrame *pRefKF = pMPi->GetReferenceKeyFrame();
        g2o::SE3Quat Srw;
        g2o::SE3Quat correctedSwr;
        while (pRefKF->isBad())
        {
            if (!pRefKF)
            {
                Verbose::PrintMess("MP " + to_string(pMPi->mnId) + " without a valid reference KF", Verbose::VERBOSITY_DEBUG);
                break;
            }

            pMPi->EraseObservation(pRefKF);
            pRefKF = pMPi->GetReferenceKeyFrame();
        }
        /*if(pRefKF->mnMergeCorrectedForKF == pCurKF->mnId)
    {
        int nIDr = pRefKF->mnId;

        Srw = vScw[nIDr];
        correctedSwr = vCorrectedSwc[nIDr];
    }
    else
    {*/
        //cv::Mat TNonCorrectedwr = pRefKF->mTwcBefMerge;
        //Eigen::Matrix<double, 3, 3> RNonCorrectedwr = Converter::toMatrix3d(TNonCorrectedwr.rowRange(0, 3).colRange(0, 3));
        //Eigen::Matrix<double, 3, 1> tNonCorrectedwr = Converter::toVector3d(TNonCorrectedwr.rowRange(0, 3).col(3));
        Srw = vScw_bef[pRefKF->mnId]; //g2o::SE3Quat(RNonCorrectedwr, tNonCorrectedwr).inverse();

        cv::Mat Twr = pRefKF->GetPoseInverse();
        Eigen::Matrix<double, 3, 3> Rwr = Converter::toMatrix3d(Twr.rowRange(0, 3).colRange(0, 3));
        Eigen::Matrix<double, 3, 1> twr = Converter::toVector3d(Twr.rowRange(0, 3).col(3));
        correctedSwr = g2o::SE3Quat(Rwr, twr);
        //}
        //cout << "Opt_Essential: Loaded the KF reference position" << endl;

        cv::Mat P3Dw = pMPi->GetWorldPos() / scale;
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        //cout << "Opt_Essential: Calculated the new MP position" << endl;
        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        //cout << "Opt_Essential: Converted the position to the OpenCV format" << endl;
        pMPi->SetWorldPos(cvCorrectedP3Dw);
        //cout << "Opt_Essential: Loaded the corrected position in the MP object" << endl;

        pMPi->UpdateNormalAndDepth();
    }

    Verbose::PrintMess("Opt_Essential: End of the optimization", Verbose::VERBOSITY_DEBUG);
}

/**
 * @brief  LoopClosing::MergeLocal() 融合地图时使用，优化当前帧没有参与融合的元素，可以理解为局部的本质图优化，有严重BUG
 * 优化目标： 1. vpNonFixedKFs; 2.vpNonCorrectedMPs
 * @param pCurKF                 mpCurrentKF 融合时当前关键帧
 * @param vpFixedKFs             vpMergeConnectedKFs 融合地图中的关键帧
 * @param vpFixedCorrectedKFs    vpLocalCurrentWindowKFs 当前地图中经过矫正的关键帧
 * @param vpNonFixedKFs          vpCurrentMapKFs 当前地图中剩余的关键帧，待优化
 * @param vpNonCorrectedMPs      vpCurrentMapMPs 当前地图中剩余的MP点，待优化
 */
void Optimizer::OptimizeEssentialGraph(KeyFrame *pCurKF, vector<KeyFrame *> &vpFixedKFs, vector<KeyFrame *> &vpFixedCorrectedKFs,
                                        vector<KeyFrame *> &vpNonFixedKFs, vector<MapPoint *> &vpNonCorrectedMPs)
{
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedKFs.size()) + " KFs fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map", Verbose::VERBOSITY_DEBUG);

    // 1. 优化器构建
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
        new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    Map *pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);          // 存放每一帧优化前的sim3
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1); // 存放每一帧优化后的sim3，调试输出用
    vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);                           // 存放节点，没用，还占地方

    const int minFeat = 100; // pKFi->GetCovisiblesByWeight(minFeat);  essentialgraph 阈值就是共视大于100

    // 2. 确定固定关键帧的节点
    for (KeyFrame *pKFi : vpFixedKFs)
    {
        if (pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
        g2o::Sim3 Siw(Rcw, tcw, 1.0);
        vScw[nIDi] = Siw;
        vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
        VSim3->setEstimate(Siw);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = true; // TODO

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;
    }
    Verbose::PrintMess("Opt_Essential: vpFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

    set<unsigned long> sIdKF;
    for (KeyFrame *pKFi : vpFixedCorrectedKFs)
    {
        if (pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
        g2o::Sim3 Siw(Rcw, tcw, 1.0);
        //vScw[nIDi] = Siw;
        vCorrectedSwc[nIDi] = Siw.inverse(); // This KFs mustn't be corrected
        VSim3->setEstimate(Siw);

        cv::Mat Tcw_bef = pKFi->mTcwBefMerge;
        Eigen::Matrix<double, 3, 3> Rcw_bef = Converter::toMatrix3d(Tcw_bef.rowRange(0, 3).colRange(0, 3));
        Eigen::Matrix<double, 3, 1> tcw_bef = Converter::toVector3d(Tcw_bef.rowRange(0, 3).col(3));
        vScw[nIDi] = g2o::Sim3(Rcw_bef, tcw_bef, 1.0);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;

        sIdKF.insert(nIDi);
    }
    Verbose::PrintMess("Opt_Essential: vpFixedCorrectedKFs loaded", Verbose::VERBOSITY_DEBUG);

    // 3. 确定待优化的关键帧节点
    for (KeyFrame *pKFi : vpNonFixedKFs)
    {
        if (pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        if (sIdKF.count(nIDi)) // It has already added in the corrected merge KFs
            continue;

        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        //cv::Mat Tcw = pKFi->mTcwBefMerge;
        //Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(Tcw.rowRange(0, 3).colRange(0, 3));
        //Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(Tcw.rowRange(0, 3).col(3));
        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKFi->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKFi->GetTranslation());
        g2o::Sim3 Siw(Rcw, tcw, 1.0);
        vScw[nIDi] = Siw;
        // BUG !!!!!!!!!! 用未矫正的作为观测值？？？ 还优化个蛋蛋
        VSim3->setEstimate(Siw);

        VSim3->setFixed(false);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;

        sIdKF.insert(nIDi);
    }
    Verbose::PrintMess("Opt_Essential: vpNonFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

    vector<KeyFrame *> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(), vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
    set<KeyFrame *> spKFs(vpKFs.begin(), vpKFs.end());

    Verbose::PrintMess("Opt_Essential: List of KF loaded", Verbose::VERBOSITY_DEBUG);

    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();
    // 4. 遍历所有帧
    for (KeyFrame *pKFi : vpKFs)
    {
        int num_connections = 0; // 统计与pKFi连接的数量
        const int nIDi = pKFi->mnId;

        g2o::Sim3 Swi = vScw[nIDi].inverse();
        /*if(pKFi->mnMergeCorrectedForKF == pCurKF->mnId)
    {
            Swi = vScw[nIDi].inverse();
    }
    else
    {
        cv::Mat Twi = pKFi->mTwcBefMerge;
        Swi = g2o::Sim3(Converter::toMatrix3d(Twi.rowRange(0, 3).colRange(0, 3)), 
                        Converter::toVector3d(Twi.rowRange(0, 3).col(3)), 1.0);
    }*/

        KeyFrame *pParentKFi = pKFi->GetParent();

        // Spanning tree edge
        // 4.1 找到pKFi的父帧且在这批关键帧里面，添加与其关联的sim3边
        if (pParentKFi && spKFs.find(pParentKFi) != spKFs.end())
        {
            int nIDj = pParentKFi->mnId;

            g2o::Sim3 Sjw = vScw[nIDj];

            /*if(pParentKFi->mnMergeCorrectedForKF == pCurKF->mnId)
        {
                Sjw =  vScw[nIDj];
        }
        else
        {
            cv::Mat Tjw = pParentKFi->mTcwBefMerge;
            Sjw = g2o::Sim3(Converter::toMatrix3d(Tjw.rowRange(0, 3).colRange(0, 3)), 
                            Converter::toVector3d(Tjw.rowRange(0, 3).col(3)), 1.0);
        }*/

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda; // 单位阵
            optimizer.addEdge(e);
            num_connections++;
        }

        // Loop edges
        // 4.2 添加回环的边（极大概率没有）
        const set<KeyFrame *> sLoopEdges = pKFi->GetLoopEdges(); // LoopClosing::CorrectLoop 执行时会与回环帧相互关联这个变量
        for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            KeyFrame *pLKF = *sit;
            // 保证回环帧也在这批关键帧里面，且id上小于pKFi（目的是防止添加两次）
            if (spKFs.find(pLKF) != spKFs.end() && pLKF->mnId < pKFi->mnId)
            {
                g2o::Sim3 Slw = vScw[pLKF->mnId];

                /*if(pLKF->mnMergeCorrectedForKF == pCurKF->mnId)
            {
                    Slw = vScw[pLKF->mnId];
            }
            else
            {
                cv::Mat Tlw = pLKF->mTcwBefMerge;
                Slw = g2o::Sim3(Converter::toMatrix3d(Tlw.rowRange(0, 3).colRange(0, 3)), 
                                Converter::toVector3d(Tlw.rowRange(0, 3).col(3)), 1.0);
            }*/

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
                num_connections++;
            }
        }

        // Covisibility graph edges
        // 4.3 建立essentialgraph
        const vector<KeyFrame *> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            KeyFrame *pKFn = *vit;
            // 1.这个帧存在且不是pKFi的父帧，防止重复添加
            // 2.pKFn不为pKFi的子帧
            // 3.pKFn不为回环帧，防止重复添加
            // 4.pKFn要在这批关键帧里面
            if (pKFn && pKFn != pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn) && spKFs.find(pKFn) != spKFs.end())
            {
                if (!pKFn->isBad() && pKFn->mnId < pKFi->mnId)
                {

                    g2o::Sim3 Snw = vScw[pKFn->mnId];
                    /*if(pKFn->mnMergeCorrectedForKF == pCurKF->mnId)
                {
                    Snw = vScw[pKFn->mnId];
                }
                else
                {
                    cv::Mat Tnw = pKFn->mTcwBefMerge;
                    Snw = g2o::Sim3(Converter::toMatrix3d(Tnw.rowRange(0, 3).colRange(0, 3)), 
                                    Converter::toVector3d(Tnw.rowRange(0, 3).col(3)), 1.0);
                }*/

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                    num_connections++;
                }
            }
        }

        if (num_connections == 0)
        {
            Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections", Verbose::VERBOSITY_DEBUG);
        }
    }

    // Optimize!
    // 5. 开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    Verbose::PrintMess("Opt_Essential: Finish the optimization", Verbose::VERBOSITY_DEBUG);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    Verbose::PrintMess("Opt_Essential: Apply the new pose to the KFs", Verbose::VERBOSITY_DEBUG);
    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 6. 取出结果
    for (KeyFrame *pKFi : vpNonFixedKFs)
    {
        if (pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw = VSim3->estimate();
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *= (1. / s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

        /*{
        cv::Mat Tco_cn = pKFi->GetPose() * Tiw.inv();
        cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
        double dist = cv::norm(trasl);
        if(dist > 1.0)
        {
            cout << "--Distance: " << dist << " meters" << endl;
            cout << "--To much distance correction in EssentGraph: It has connected " << pKFi->GetVectorCovisibleKeyFrames().size() << " KFs" << endl;
        }
        string strNameFile = pKFi->mNameFile;
        cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);
        cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);
        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();
        for(int j=0; j<vpMapPointsKFi.size(); ++j)
        {
            if(!vpMapPointsKFi[j] || vpMapPointsKFi[j]->isBad())
            {
                continue;
            }
            string strNumOBs = to_string(vpMapPointsKFi[j]->Observations());
            cv::circle(imLeft, pKFi->mvKeys[j].pt, 2, cv::Scalar(0, 255, 0));
            cv::putText(imLeft, strNumOBs, pKFi->mvKeys[j].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
        }
        string namefile = "./test_OptEssent/Essent_" + to_string(pCurKF->mnId) + "_KF" + to_string(pKFi->mnId) +"_D" + to_string(dist) +".png";
        cv::imwrite(namefile, imLeft);
    }*/

        pKFi->mTcwBefMerge = pKFi->GetPose();
        pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
        pKFi->SetPose(Tiw);
    }

    Verbose::PrintMess("Opt_Essential: Apply the new pose to the MPs", Verbose::VERBOSITY_DEBUG);
    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for (MapPoint *pMPi : vpNonCorrectedMPs)
    {
        if (pMPi->isBad())
            continue;

        Verbose::PrintMess("Opt_Essential: MP id " + to_string(pMPi->mnId), Verbose::VERBOSITY_DEBUG);
        /*int nIDr;
    if(pMPi->mnCorrectedByKF==pCurKF->mnId)
    {
        nIDr = pMPi->mnCorrectedReference;
    }
    else
    {
    }*/
        KeyFrame *pRefKF = pMPi->GetReferenceKeyFrame();
        g2o::Sim3 Srw;
        g2o::Sim3 correctedSwr;
        while (pRefKF->isBad())
        {
            if (!pRefKF)
            {
                Verbose::PrintMess("MP " + to_string(pMPi->mnId) + " without a valid reference KF", Verbose::VERBOSITY_DEBUG);
                break;
            }

            pMPi->EraseObservation(pRefKF);
            pRefKF = pMPi->GetReferenceKeyFrame();
        }
        /*if(pRefKF->mnMergeCorrectedForKF == pCurKF->mnId)
    {
        int nIDr = pRefKF->mnId;
        Srw = vScw[nIDr];
        correctedSwr = vCorrectedSwc[nIDr];
    }
    else
    {*/
        cv::Mat TNonCorrectedwr = pRefKF->mTwcBefMerge;
        Eigen::Matrix<double, 3, 3> RNonCorrectedwr = Converter::toMatrix3d(TNonCorrectedwr.rowRange(0, 3).colRange(0, 3));
        Eigen::Matrix<double, 3, 1> tNonCorrectedwr = Converter::toVector3d(TNonCorrectedwr.rowRange(0, 3).col(3));
        Srw = g2o::Sim3(RNonCorrectedwr, tNonCorrectedwr, 1.0).inverse();

        cv::Mat Twr = pRefKF->GetPoseInverse();
        Eigen::Matrix<double, 3, 3> Rwr = Converter::toMatrix3d(Twr.rowRange(0, 3).colRange(0, 3));
        Eigen::Matrix<double, 3, 1> twr = Converter::toVector3d(Twr.rowRange(0, 3).col(3));
        correctedSwr = g2o::Sim3(Rwr, twr, 1.0);
        //}
        //cout << "Opt_Essential: Loaded the KF reference position" << endl;

        cv::Mat P3Dw = pMPi->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        //cout << "Opt_Essential: Calculated the new MP position" << endl;
        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        //cout << "Opt_Essential: Converted the position to the OpenCV format" << endl;
        pMPi->SetWorldPos(cvCorrectedP3Dw);
        //cout << "Opt_Essential: Loaded the corrected position in the MP object" << endl;

        pMPi->UpdateNormalAndDepth();
    }

    Verbose::PrintMess("Opt_Essential: End of the optimization", Verbose::VERBOSITY_DEBUG);
}

/**
 * @brief 没有使用，暂时不看
 */
void Optimizer::OptimizeEssentialGraph(KeyFrame *pCurKF,
                                        const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                        const LoopClosing::KeyFrameAndPose &CorrectedSim3)
{
    // Setup optimizer
    Map *pMap = pCurKF->GetMap();
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
        new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
    vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

    const int minFeat = 100; // TODO Check. originally 100

    // Set KeyFrame vertices
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
        Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
        g2o::Sim3 Siw(Rcw, tcw, 1.0);
        vScw[nIDi] = Siw;
        VSim3->setEstimate(Siw);

        if (pKF->mnBALocalForKF == pCurKF->mnId || pKF->mnBAFixedForKF == pCurKF->mnId)
        {
            cout << "fixed fk: " << pKF->mnId << endl;
            VSim3->setFixed(true);
        }
        else
            VSim3->setFixed(false);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        // TODO Check
        // VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;
    }

    set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

    int count_edges[3] = {0, 0, 0};
    // Set normal edges
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame *pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if (iti != NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame *pParentKF = pKF->GetParent();

        // Spanning tree edge
        if (pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;
            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if (itj != NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
            count_edges[0]++;
        }

        // Loop edges
        const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
        for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            KeyFrame *pLKF = *sit;
            if (pLKF->mnId < pKF->mnId)
            {
                g2o::Sim3 Slw;
                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if (itl != NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
                count_edges[1]++;
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            KeyFrame *pKFn = *vit;
            if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                {
                    // just one edge between frames
                    if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if (itn != NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                    count_edges[2]++;
                }
            }
        }
    }

    Verbose::PrintMess("edges pose graph: " + to_string(count_edges[0]) + ", " + to_string(count_edges[1]) + ", " + to_string(count_edges[2]), Verbose::VERBOSITY_DEBUG);
    // Optimize!
    optimizer.initializeOptimization();
    optimizer.setVerbose(false);
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw = VSim3->estimate();
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *= (1. / s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMPs[i];

        if (pMP->isBad())
            continue;

        int nIDr;
        if (pMP->mnCorrectedByKF == pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }

    // TODO Check this changeindex
    pMap->IncreaseChangeIndex();
}

/**
 * @brief 没有使用，暂时不看
 */
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale = bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0, 2);
    vSim3->_principle_point1[1] = K1.at<float>(1, 2);
    vSim3->_focal_length1[0] = K1.at<float>(0, 0);
    vSim3->_focal_length1[1] = K1.at<float>(1, 1);
    vSim3->_principle_point2[0] = K2.at<float>(0, 2);
    vSim3->_principle_point2[1] = K2.at<float>(1, 2);
    vSim3->_focal_length2[0] = K2.at<float>(0, 0);
    vSim3->_focal_length2[1] = K2.at<float>(1, 1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2 * N);
    vpEdges12.reserve(2 * N);
    vpEdges21.reserve(2 * N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for (int i = 0; i < N; i++)
    {
        if (!vpMatches1[i])
            continue;

        MapPoint *pMP1 = vpMapPoints1[i];
        MapPoint *pMP2 = vpMatches1[i];

        const int id1 = 2 * i + 1;
        const int id2 = 2 * (i + 1);

        const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKF2));

        if (pMP1 && pMP2)
        {
            if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0)
            {
                g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w * P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double, 2, 1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ *e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double, 2, 1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ *e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(NULL);
            vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if (nBad > 0)
        nMoreIterations = 10;
    else
        nMoreIterations = 5;

    if (nCorrespondences - nBad < 10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();

    return nIn;
}

/**
 * @brief loop closing使用，1投2， 2投1这么来
 * @param pKF1         当前关键帧
 * @param pKF2         候选关键帧
 * @param vpMatches1   当前关键帧与地图匹配上的MP，中间会有NULL，与当前关键帧的特征点一一对应
 * @param g2oS12       相似变换矩阵
 * @param th2          误差上限的平方
 * @param bFixScale    是否固定尺度
 * @param mAcumHessian 计算累计海森矩阵（没啥用）
 * @param bAllPoints   是否计算所有点（都为true）
 * 
 * 总结下与ORBSLAM2的不同
 * 前面操作基本一样，这里面当KF1特征点没有对应的自己的MP，却有回环的MP时
 */
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2,
                            const bool bFixScale, Eigen::Matrix<double, 7, 7> &mAcumHessian, const bool bAllPoints)
{
    bool bShowImages = false;

    // 步骤1：初始化g2o优化器
    // 先构造求解器
    g2o::SparseOptimizer optimizer;
    // 构造线性方程求解器，Hx = -b的求解器
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    // 使用dense的求解器，（常见非dense求解器有cholmod线性求解器和shur补线性求解器）
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
    // 使用L-M迭代
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    //const cv::Mat &DistCoeff2 = pKF2->mDistCoef;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    // 步骤2.1 添加Sim3顶点
    ORB_SLAM3::VertexSim3Expmap *vSim3 = new ORB_SLAM3::VertexSim3Expmap();
    vSim3->_fix_scale = bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->pCamera1 = pKF1->mpCamera;
    vSim3->pCamera2 = pKF2->mpCamera;
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<ORB_SLAM3::EdgeSim3ProjectXYZ *> vpEdges12;        //pKF2对应的MapPoints到pKF1的投影
    vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ *> vpEdges21; //pKF1对应的MapPoints到pKF2的投影
    vector<size_t> vnIndexEdge;
    vector<bool> vbIsInKF2;

    vnIndexEdge.reserve(2 * N);
    vpEdges12.reserve(2 * N);
    vpEdges21.reserve(2 * N);
    vbIsInKF2.reserve(2 * N);

    float cx1 = K1.at<float>(0, 2);
    float cy1 = K1.at<float>(1, 2);
    float fx1 = K1.at<float>(0, 0);
    float fy1 = K1.at<float>(1, 1);

    float cx2 = K2.at<float>(0, 2);
    float cy2 = K2.at<float>(1, 2);
    float fx2 = K2.at<float>(0, 0);
    float fy2 = K2.at<float>(1, 1);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;
    int nBadMPs = 0;         // 没有实际用处，没有输出信息
    int nInKF2 = 0;          // 输出信息用
    int nOutKF2 = 0;         // 输出信息用
    int nMatchWithoutMP = 0; // 输出信息用

    cv::Mat img1 = cv::imread(pKF1->mNameFile, CV_LOAD_IMAGE_UNCHANGED);
    cv::cvtColor(img1, img1, CV_GRAY2BGR);
    cv::Mat img2 = cv::imread(pKF2->mNameFile, CV_LOAD_IMAGE_UNCHANGED);
    cv::cvtColor(img2, img2, CV_GRAY2BGR);

    vector<int> vIdsOnlyInKF2; // 统计点仅在没有实际用处
    // 有可能在当前帧特征点中没有对应的原始mp，却有对应的回环mp，反之同理
    // 优化目标g2oS12
    // 添加边与三维点定点，KF2 pMP2点投到 KF1 里面，再做KF1点 pMP1投到KF2里面，其中obs2是KF2对应特征点坐标，如果没有，通过投影pMP2到KF2作为特征点
    for (int i = 0; i < N; i++)
    {
        if (!vpMatches1[i])
            continue;

        // pMP1和pMP2是匹配的MapPoints，pMP1表示当前帧正常对应的mp，pMP2表示对应的回环的mp
        MapPoint *pMP1 = vpMapPoints1[i];
        MapPoint *pMP2 = vpMatches1[i];

        // (1, 2) (3, 4) (5, 6)
        const int id1 = 2 * i + 1;
        const int id2 = 2 * (i + 1);

        const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKF2));
        /*if(i2 < 0)
        cout << "Sim3-OPT: Error, there is a matched which is not find it" << endl;*/

        cv::Mat P3D1c; // 点1在当前关键帧相机坐标系下的坐标
        cv::Mat P3D2c; // 点2在候选关键帧相机坐标系下的坐标

        if (pMP1 && pMP2)
        {
            //if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            if (!pMP1->isBad() && !pMP2->isBad())
            {
                // 步骤2.2 添加PointXYZ顶点， 且设为了固定
                g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                P3D1c = R1w * P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
            {
                nBadMPs++;
                continue;
            }
        }
        else
        {
            nMatchWithoutMP++;

            //TODO The 3D position in KF1 doesn't exist
            if (!pMP2->isBad())
            {
                // 执行到这里意味着特征点没有对应的原始MP，却有回环MP，将其投到候选帧里面
                g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);

                vIdsOnlyInKF2.push_back(id2);
            }

            cv::circle(img1, pKF1->mvKeys[i].pt, 1, cv::Scalar(0, 0, 255));

            continue;
        }

        if (i2 < 0 && !bAllPoints)
        {
            Verbose::PrintMess("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints), Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if (P3D2c.at<float>(2) < 0)
        {
            Verbose::PrintMess("Sim3: Z coordinate is negative", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        nCorrespondences++;

        // 步骤2.3 添加两个顶点（3D点）到相机投影的边
        // Set edge x1 = S12*X2
        Eigen::Matrix<double, 2, 1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ();
        /*bool inKF1;
    if(pMP1)
    {
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        inKF1 = true;
    }
    else
    {
        float invz = 1/P3D1c.at<float>(2);
        float x = P3D1c.at<float>(0)*invz;
        float y = P3D1c.at<float>(1)*invz;

        obs1 << x, y;
        kpUn1 = cv::KeyPoint(cv::Point2f(x, y), pMP1->mnTrackScaleLevel);

        inKF1 = false;
    }*/

        if (bShowImages) //TODO test to project the matched points in the image
        {
            cv::circle(img1, pKF1->mvKeys[i].pt, 1, cv::Scalar(0, 255, 0));

            Eigen::Matrix<double, 3, 1> eigP3D2c = Converter::toVector3d(P3D2c);
            Eigen::Matrix<double, 3, 1> eigP3D1c = g2oS12.map(eigP3D2c);
            cv::Mat cvP3D1c = Converter::toCvMat(eigP3D1c);

            float invz = 1 / cvP3D1c.at<float>(2);
            float x = fx1 * cvP3D1c.at<float>(0) * invz + cx1;
            float y = fy1 * cvP3D1c.at<float>(1) * invz + cy1;

            cv::Point2f ptProjPoint(x, y);
            cv::line(img1, pKF1->mvKeys[i].pt, ptProjPoint, cv::Scalar(255, 0, 0), 1);
        }

        // 2相机坐标系下的三维点经过g2oS12投影到kf1下计算重投影误差
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2))); // 2相机坐标系下的三维点
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));   // g2oS12
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber); // sqrt(th2)
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        // 步骤2.4 另一个边
        Eigen::Matrix<double, 2, 1> obs2;
        cv::KeyPoint kpUn2;
        bool inKF2;
        if (i2 >= 0)
        {
            kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;
            inKF2 = true;

            nInKF2++; // 输出信息，表示在kf2中找到MP2的点数

            if (bShowImages)
            {
                cv::circle(img2, pKF2->mvKeys[i2].pt, 1, cv::Scalar(0, 255, 0));
            }
        }
        else // BUG 如果没找到，使用三维点投影到KF2中，表示并没有特征点与之对应（把这个结果当做obs2是不是会带来一些误差，而且还不通过内参吗？？？，重大bug）
        {
            float invz = 1 / P3D2c.at<float>(2);
            float x = P3D2c.at<float>(0) * invz;
            float y = P3D2c.at<float>(1) * invz;

            /*cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = x;
        mat.at<float>(0, 1) = y;
        mat=mat.reshape(2);
        cv::undistortPoints(mat, mat, K2, DistCoeff2, cv::Mat(), K2);
        mat=mat.reshape(1);

        x = mat.at<float>(0, 0);
        y = mat.at<float>(0, 1);*/

            obs2 << x, y;
            kpUn2 = cv::KeyPoint(cv::Point2f(x, y), pMP2->mnTrackScaleLevel); // 金字塔层数

            inKF2 = false;
            nOutKF2++; // 输出信息，表示在kf2中未找到MP2的点数

            if (bShowImages)
            {
                cv::circle(img2, kpUn2.pt, 1, cv::Scalar(0, 0, 255));
            }

            //TODO print projection, because all of them become in outliers

            // Project in image and check it is not outside
            //float u = pKF2->fx * x + pKF2->cx;
            //float v = pKF2->fy * y + pKF2->cy;
            //obs2 << u, v;
            //kpUn2 = cv::KeyPoint(cv::Point2f(u, v), pMP2->mnTrackScaleLevel);
        }
        // 画图使用的
        {
            Eigen::Matrix<double, 3, 1> eigP3D1c = Converter::toVector3d(P3D1c);
            Eigen::Matrix<double, 3, 1> eigP3D2c = g2oS12.inverse().map(eigP3D1c);
            cv::Mat cvP3D2c = Converter::toCvMat(eigP3D2c);

            float invz = 1 / cvP3D2c.at<float>(2);
            float x = fx2 * cvP3D2c.at<float>(0) * invz + cx2;
            float y = fy2 * cvP3D2c.at<float>(1) * invz + cy2;

            if (bShowImages)
            {
                cv::Point2f ptProjPoint(x, y);
                cv::line(img2, kpUn2.pt, ptProjPoint, cv::Scalar(255, 0, 0), 1);
            }
        }
        // 1相机坐标系下的三维点经过g2oS12投影到kf2下计算重投影误差
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = new ORB_SLAM3::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);

        vbIsInKF2.push_back(inKF2);
    }

    Verbose::PrintMess("Sim3: There are " + to_string(nCorrespondences) + " matches, " + to_string(nInKF2) + " are in the KF and " + to_string(nOutKF2) + " are in the connected KFs. There are " + to_string(nMatchWithoutMP) + " matches which have not an associate MP", Verbose::VERBOSITY_DEBUG);

    // Optimize!
    // 步骤3：g2o开始优化，先迭代5次
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    // 步骤4：剔除一些误差大的边，因为e12与e21对应的是同一个三维点，所以只要有一个误差太大就直接搞掉
    // Check inliers
    // 进行卡方检验，大于阈值的边剔除，同时删除鲁棒核函数
    int nBad = 0;
    int nBadOutKF2 = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<ORB_SLAM3::EdgeSim3ProjectXYZ *>(NULL);
            vpEdges21[i] = static_cast<ORB_SLAM3::EdgeInverseSim3ProjectXYZ *>(NULL);
            nBad++;

            if (!vbIsInKF2[i])
            {
                nBadOutKF2++;
            }
            continue;
        }

        //Check if remove the robust adjustment improve the result
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }
    if (bShowImages)
    {
        string pathImg1 = "./test_OptSim3/KF_" + to_string(pKF1->mnId) + "_Main.jpg";
        cv::imwrite(pathImg1, img1);
        string pathImg2 = "./test_OptSim3/KF_" + to_string(pKF1->mnId) + "_Matched.jpg";
        cv::imwrite(pathImg2, img2);
    }

    Verbose::PrintMess("Sim3: First Opt -> Correspondences: " + to_string(nCorrespondences) + "; nBad: " + to_string(nBad) + "; nBadOutKF2: " + to_string(nBadOutKF2), Verbose::VERBOSITY_DEBUG);

    // 如果有坏点，迭代次数更多
    int nMoreIterations;
    if (nBad > 0)
        nMoreIterations = 10;
    else
        nMoreIterations = 5;

    if (nCorrespondences - nBad < 10)
        return 0;

    // Optimize again only with inliers
    // 步骤5：再次g2o优化剔除后剩下的边
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
    // 更新vpMatches1，删除外点，统计内点数量
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
        }
        else
        {
            nIn++;
            //mAcumHessian += e12->GetHessian();
        }
    }

    // Recover optimized Sim3
    //Verbose::PrintMess("Sim3: Initial seed " + g2oS12, Verbose::VERBOSITY_DEBUG);
    // 步骤6：得到优化后的结果
    g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();

    //Verbose::PrintMess("Sim3: Optimized solution " + g2oS12, Verbose::VERBOSITY_DEBUG);

    return nIn;
}

/**
 * @brief 没有使用，暂时不看
 */
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, vector<KeyFrame *> &vpMatches1KF, g2o::Sim3 &g2oS12, const float th2,
                            const bool bFixScale, Eigen::Matrix<double, 7, 7> &mAcumHessian, const bool bAllPoints)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    Verbose::PrintMess("Extracted rotation and traslation from the first KF ", Verbose::VERBOSITY_DEBUG);
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();
    Verbose::PrintMess("Extracted rotation and traslation from the second KF ", Verbose::VERBOSITY_DEBUG);

    // Set Sim3 vertex
    g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale = bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0, 2);
    vSim3->_principle_point1[1] = K1.at<float>(1, 2);
    vSim3->_focal_length1[0] = K1.at<float>(0, 0);
    vSim3->_focal_length1[1] = K1.at<float>(1, 1);
    vSim3->_principle_point2[0] = K2.at<float>(0, 2);
    vSim3->_principle_point2[1] = K2.at<float>(1, 2);
    vSim3->_focal_length2[0] = K2.at<float>(0, 0);
    vSim3->_focal_length2[1] = K2.at<float>(1, 1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<ORB_SLAM3::EdgeSim3ProjectXYZ *> vpEdges12;
    vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ *> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2 * N);
    vpEdges12.reserve(2 * N);
    vpEdges21.reserve(2 * N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    KeyFrame *pKFm = pKF2;
    for (int i = 0; i < N; i++)
    {
        if (!vpMatches1[i])
            continue;

        MapPoint *pMP1 = vpMapPoints1[i];
        MapPoint *pMP2 = vpMatches1[i];

        const int id1 = 2 * i + 1;
        const int id2 = 2 * (i + 1);

        pKFm = vpMatches1KF[i];
        const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKFm));
        if (i2 < 0)
            Verbose::PrintMess("Sim3-OPT: Error, there is a matched which is not find it", Verbose::VERBOSITY_DEBUG);

        cv::Mat P3D2c;

        if (pMP1 && pMP2)
        {
            //if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            if (!pMP1->isBad() && !pMP2->isBad())
            {
                g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w * P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        if (i2 < 0 && !bAllPoints)
        {
            Verbose::PrintMess("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints), Verbose::VERBOSITY_DEBUG);
            continue;
        }

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double, 2, 1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double, 2, 1> obs2;
        cv::KeyPoint kpUn2;
        if (i2 >= 0 && pKFm == pKF2)
        {
            kpUn2 = pKFm->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;
        }
        else
        {
            float invz = 1 / P3D2c.at<float>(2);
            float x = P3D2c.at<float>(0) * invz;
            float y = P3D2c.at<float>(1) * invz;

            // Project in image and check it is not outside
            float u = pKF2->fx * x + pKFm->cx;
            float v = pKF2->fy * y + pKFm->cy;
            obs2 << u, v;
            kpUn2 = cv::KeyPoint(cv::Point2f(u, v), pMP2->mnTrackScaleLevel);
        }

        ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = new ORB_SLAM3::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKFm->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<ORB_SLAM3::EdgeSim3ProjectXYZ *>(NULL);
            vpEdges21[i] = static_cast<ORB_SLAM3::EdgeInverseSim3ProjectXYZ *>(NULL);
            nBad++;
            continue;
        }

        //Check if remove the robust adjustment improve the result
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    //cout << "Sim3 -> Correspondences: " << nCorrespondences << "; nBad: " << nBad << endl;

    int nMoreIterations;
    if (nBad > 0)
        nMoreIterations = 10;
    else
        nMoreIterations = 5;

    if (nCorrespondences - nBad < 10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
        }
        else
        {
            nIn++;
            //mAcumHessian += e12->GetHessian();
        }
    }

    // Recover optimized Sim3
    ORB_SLAM3::VertexSim3Expmap *vSim3_recov = static_cast<ORB_SLAM3::VertexSim3Expmap *>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();

    return nIn;
}

/** 
 * @brief 局部地图＋惯导BA LocalMapping IMU中使用，地图经过imu初始化时用这个函数代替LocalBundleAdjustment
 * @param pKF 关键帧
 * @param pbStopFlag 是否停止的标志
 * @param pMap 地图
 * @param bLarge 匹配点是否足够多
 * @param bRecInit !GetIniertialBA2()
 */
void Optimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, bool bLarge, bool bRecInit)
{
    // 1. 确定待优化的关键帧
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Map *pCurrentMap = pKF->GetMap();

    int maxOpt = 10;
    int opt_it = 10;
    if (bLarge)
    {
        maxOpt = 25;
        opt_it = 4;
    }

    // 预计待优化的关键帧数，bLarge越大说明了附近关键帧比较多，所以maxOpt就大一些，如果当前地图关键帧数量很多，直接等于maxOpt
    const int Nd = std::min((int)pCurrentMap->KeyFramesInMap() - 2, maxOpt);
    const unsigned long maxKFid = pKF->mnId;

    vector<KeyFrame *> vpOptimizableKFs;
    vpOptimizableKFs.reserve(Nd);
    vpOptimizableKFs.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    for (int i = 1; i < Nd; i++)
    {
        if (vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by temporal optimizable keyframes
    // 2. 确定这些关键帧对应的MP，存入lLocalMapPoints
    list<MapPoint *> lLocalMapPoints;
    for (int i = 0; i < N; i++)
    {
        vector<MapPoint *> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    // Fixed Keyframe: First frame previous KF to optimization window)
    // 3. 固定一帧，为vpOptimizableKFs中最早的那一关键帧的上一关键帧，如果没有上一关键帧了就用最早的那一帧，毕竟目前得到的地图虽然有尺度但并不是绝对的位置
    list<KeyFrame *> lFixedKeyFrames;
    if (vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF = pKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF = 0;
        vpOptimizableKFs.back()->mnBAFixedForKF = pKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Optimizable visual KFs
    // 4. 做了一系列操作发现最后lpOptVisKFs为空。这段应该是调试遗留代码，如果实现的话其实就是把共视图中在前面没有加过的关键帧们加进来，但作者可能发现之前就把共视图的全部帧加进来了，也有可能发现优化的效果不好浪费时间
    // 获得与当前关键帧有共视关系的一些关键帧，大于15个点，排序为从小到大
    const vector<KeyFrame *> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
    list<KeyFrame *> lpOptVisKFs;
    const int maxCovKF = 0;
    for (int i = 0, iend = vpNeighsKFs.size(); i < iend; i++)
    {
        if (lpOptVisKFs.size() >= maxCovKF)
            break;

        KeyFrame *pKFi = vpNeighsKFs[i];
        if (pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
            continue;
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            lpOptVisKFs.push_back(pKFi);

            vector<MapPoint *> vpMPs = pKFi->GetMapPointMatches();
            for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                MapPoint *pMP = *vit;
                if (pMP)
                    if (!pMP->isBad())
                        if (pMP->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
            }
        }
    }

    // Fixed KFs which are not covisible optimizable
    // 5. 将所有mp点对应的关键帧（除了前面加过的）放入到固定组里面，后面优化时不改变
    const int maxFixKF = 200;

    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map<KeyFrame *, tuple<int, int>> observations = (*lit)->GetObservations();
        for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                {
                    lFixedKeyFrames.push_back(pKFi);
                    break;
                }
            }
        }
        if (lFixedKeyFrames.size() >= maxFixKF)
            break;
    }

    bool bNonFixed = (lFixedKeyFrames.size() == 0);

    // 6. 构造优化器，正式开始优化
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    if (bLarge)
    {
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e-2); // 避免迭代寻找最优lambda to avoid iterating for finding optimal lambda
        optimizer.setAlgorithm(solver);
    }
    else
    {
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e0);
        optimizer.setAlgorithm(solver);
    }

    // Set Local temporal KeyFrame vertices
    // 7. 建立关于关键帧的节点，其中包括，位姿，速度，以及两个偏置
    N = vpOptimizableKFs.size();
    for (int i = 0; i < N; i++)
    {
        KeyFrame *pKFi = vpOptimizableKFs[i];

        VertexPose *VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        // bImu = pMap->isImuInitialized()
        if (pKFi->bImu)
        {
            VertexVelocity *VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias *VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias *VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local visual KeyFrame vertices
    // 8. 建立关于共视关键帧的节点，但这里为空
    for (list<KeyFrame *>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++)
    {
        KeyFrame *pKFi = *it;
        VertexPose *VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);
    }

    // Set Fixed KeyFrame vertices
    // 8. 建立关于固定关键帧的节点，其中包括，位姿，速度，以及两个偏置
    for (list<KeyFrame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        VertexPose *VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if (pKFi->bImu) // This should be done only for keyframe just before temporal window
        {
            VertexVelocity *VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias *VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias *VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    // 暂时没看到有什么用
    vector<EdgeInertial *> vei(N, (EdgeInertial *)NULL);
    vector<EdgeGyroRW *> vegr(N, (EdgeGyroRW *)NULL);
    vector<EdgeAccRW *> vear(N, (EdgeAccRW *)NULL);
    // 9. 建立边，没有imu跳过
    for (int i = 0; i < N; i++)
    {
        KeyFrame *pKFi = vpOptimizableKFs[i];

        if (!pKFi->mPrevKF)
        {
            cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
            continue;
        }
        if (pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1);
            g2o::HyperGraph::Vertex *VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2);
            g2o::HyperGraph::Vertex *VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3);
            g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex *VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2);
            g2o::HyperGraph::Vertex *VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3);

            if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", " << VV1 << ", " << VG1 << ", " << VA1 << ", " << VP2 << ", " << VV2 << ", " << VG2 << ", " << VA2 << endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            vei[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            vei[i]->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG1));
            vei[i]->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA1));
            vei[i]->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            vei[i]->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));

            if (i == N - 1 || bRecInit)
            {
                // All inertial residuals are included without robust cost function, but not that one linking the
                // last optimizable keyframe inside of the local window and the first fixed keyframe out. The
                // information matrix for this measurement is also downweighted. This is done to avoid accumulating
                // error due to fixing variables.
                g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
                vei[i]->setRobustKernel(rki);
                if (i == N - 1)
                    vei[i]->setInformation(vei[i]->information() * 1e-2);
                rki->setDelta(sqrt(16.92));
            }
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0, VG1);
            vegr[i]->setVertex(1, VG2);
            cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
            Eigen::Matrix3d InfoG;

            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    InfoG(r, c) = cvInfoG.at<float>(r, c);
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            // cout << "b";
            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0, VA1);
            vear[i]->setVertex(1, VA2);
            cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
            Eigen::Matrix3d InfoA;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    InfoA(r, c) = cvInfoA.at<float>(r, c);
            vear[i]->setInformation(InfoA);

            optimizer.addEdge(vear[i]);
        }
        else
            cout << "ERROR building inertial edge" << endl;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (N + lFixedKeyFrames.size()) * lLocalMapPoints.size();

    // Mono
    vector<EdgeMono *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<EdgeStereo *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid * 5;

    map<int, int> mVisEdges;
    for (int i = 0; i < N; i++)
    {
        KeyFrame *pKFi = vpOptimizableKFs[i];
        mVisEdges[pKFi->mnId] = 0;
    }
    for (list<KeyFrame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
    {
        mVisEdges[(*lit)->mnId] = 0;
    }

    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

        unsigned long id = pMP->mnId + iniMPid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

        // Create visual constraints
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                continue;

            if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                cv::KeyPoint kpUn;

                // Monocular left observation
                if (leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0)
                {
                    mVisEdges[pKFi->mnId]++;

                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono *e = new EdgeMono(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                // Stereo-observation
                else if (leftIndex != -1) // Stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    mVisEdges[pKFi->mnId]++;

                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo *e = new EdgeStereo(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }

                // Monocular right observation
                if (pKFi->mpCamera2)
                {
                    int rightIndex = get<1>(mit->second);

                    if (rightIndex != -1)
                    {
                        rightIndex -= pKFi->NLeft;
                        mVisEdges[pKFi->mnId]++;

                        Eigen::Matrix<double, 2, 1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        EdgeMono *e = new EdgeMono(1);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                }
            }
        }
    }

    //cout << "Total map points: " << lLocalMapPoints.size() << endl;
    for (map<int, int>::iterator mit = mVisEdges.begin(), mend = mVisEdges.end(); mit != mend; mit++)
    {
        assert(mit->second >= 3);
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(opt_it); // Originally to 2
    float err_end = optimizer.activeRobustChi2();
    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        EdgeMono *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];
        bool bClose = pMP->mTrackDepth < 10.f;

        if (pMP->isBad())
            continue;

        if ((e->chi2() > chi2Mono2 && !bClose) || (e->chi2() > 1.5f * chi2Mono2 && bClose) || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Stereo
    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        EdgeStereo *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > chi2Stereo2)
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // TODO: Some convergence problems have been detected here
    //cout << "err0 = " << err << endl;
    //cout << "err_end = " << err_end << endl;
    if ((2 * err < err_end || isnan(err) || isnan(err_end)) && !bLarge) //bGN)
    {
        cout << "FAIL LOCAL-INERTIAL BA!!!!" << endl;
        return;
    }

    if (!vToErase.empty())
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Display main statistcis of optimization
    Verbose::PrintMess("LIBA KFs: " + to_string(N), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("LIBA bNonFixed?: " + to_string(bNonFixed), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("LIBA KFs visual outliers: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);

    for (list<KeyFrame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
        (*lit)->mnBAFixedForKF = 0;

    // Recover optimized data
    // Local temporal Keyframes
    N = vpOptimizableKFs.size();
    for (int i = 0; i < N; i++)
    {
        KeyFrame *pKFi = vpOptimizableKFs[i];

        VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
        cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF = 0;

        if (pKFi->bImu)
        {
            VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
            pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
            VertexGyroBias *VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
            VertexAccBias *VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]));
        }
    }

    // Local visual KeyFrame
    for (list<KeyFrame *>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++)
    {
        KeyFrame *pKFi = *it;
        VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
        cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF = 0;
    }

    //Points
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + iniMPid + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    /*double t_const = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
double t_opt = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
double t_rec = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
/*std::cout << " Construction time: " << t_const << std::endl;
std::cout << " Optimization time: " << t_opt << std::endl;
std::cout << " Recovery time: " << t_rec << std::endl;
std::cout << " Total time: " << t_const+t_opt+t_rec << std::endl;
std::cout << " Optimization iterations: " << opt_it << std::endl;*/
}

/** 
 * @brief PoseInertialOptimizationLastFrame 中使用 Marginalize(H, 0, 14); 
 * 使用舒尔补的方式边缘化海森矩阵，边缘化。
 * 列数 6            3                    3                            3                         6           3             3              3   
 * --------------------------------------------------------------------------------------------------------------------------------------------------- 行数
 * |  Jp1.t * Jp1  Jp1.t * Jv1         Jp1.t * Jg1                 Jp1.t * Ja1            |  Jp1.t * Jp2  Jp1.t * Jv2        0              0        |  6
 * |  Jv1.t * Jp1  Jv1.t * Jv1         Jv1.t * Jg1                 Jv1.t * Ja1            |  Jv1.t * Jp2  Jv1.t * Jv2        0              0        |  3
 * |  Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1 + Jgr1.t * Jgr1        Jg1.t * Ja1            |  Jg1.t * Jp2  Jg1.t * Jv2  Jgr1.t * Jgr2        0        |  3
 * |  Ja1.t * Jp1  Ja1.t * Jv1         Ja1.t * Jg1           Ja1.t * Ja1 + Jar1.t * Jar1  |  Ja1.t * Jp2  Ja1.t * Jv2  Jar1.t * Jar2        0        |  3
 * |--------------------------------------------------------------------------------------------------------------------------------------------------
 * |  Jp2.t * Jp1  Jp2.t * Jv1         Jp2.t * Jg1                 Jp2.t * Ja1            |  Jp2.t * Jp2  Jp2.t * Jv2        0              0        |  6
 * |  Jv2.t * Jp1  Jv2.t * Jv1         Jv2.t * Jg1                 Jv2.t * Ja1            |  Jv2.t * Jp2  Jv2.t * Jv2        0              0        |  3
 * |      0            0              Jgr2.t * Jgr1                      0                |        0           0       Jgr2.t * Jgr2        0        |  3
 * |      0            0                    0                     Jar2.t * Jar1           |        0           0             0        Jar2.t * Jar2  |  3
 * ---------------------------------------------------------------------------------------------------------------------------------------------------
 * @param H 30*30的海森矩阵
 * @param start 开始位置
 * @param end 结束位置
 */
Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    // Goal
    // a  | ab | ac       a*  | 0 | ac*
    // ba | b  | bc  -->  0   | 0 | 0
    // ca | cb | c        ca* | 0 | c*

    // Size of block before block to marginalize
    const int a = start; // 0
    // Size of block to marginalize
    const int b = end - start + 1; // 15
    // Size of block after block to marginalize
    const int c = H.cols() - (end + 1); // 15

    // Reorder as follows:
    // a  | ab | ac       a  | ac | ab
    // ba | b  | bc  -->  ca | c  | cb
    // ca | cb | c        ba | bc | b
    // 1. 调换矩阵块的位置
    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    //block(a, b, c, d) 意思是矩阵的起点在 a b  终点在  a+c-1 b+d-1
    if (a > 0)
    {
        Hn.block(0, 0, a, a) = H.block(0, 0, a, a);
        Hn.block(0, a + c, a, b) = H.block(0, a, a, b);
        Hn.block(a + c, 0, b, a) = H.block(a, 0, b, a);
    }
    if (a > 0 && c > 0)
    {
        Hn.block(0, a, a, c) = H.block(0, a + b, a, c);
        Hn.block(a, 0, c, a) = H.block(a + b, 0, c, a);
    }
    if (c > 0)
    {
        // 把H矩阵分成 2*2 的块矩阵 那么Hn为
        // |A B|   ---->  |D C|
        // |C D|          |B A|
        Hn.block(a, a, c, c) = H.block(a + b, a + b, c, c);
        Hn.block(a, a + c, c, b) = H.block(a + b, a, c, b);
        Hn.block(a + c, a, b, c) = H.block(a, a + b, b, c);
    }
    Hn.block(a + c, a + c, b, b) = H.block(a, a, b, b);

    // Perform marginalization (Schur complement)
    // 2. 边缘化（舒尔补）
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a + c, a + c, b, b), Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv = svd.singularValues(); // 奇异值
    // 奇异值求倒数，对于过小的倒数赋值成0
    for (int i = 0; i < b; ++i)
    {
        if (singularValues_inv(i) > 1e-6)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else
            singularValues_inv(i) = 0;
    }
    // 求得A的逆矩阵, A =  U * ∑ * V.t          A(-1) = V * ∑(-1) * U.t

    Eigen::MatrixXd invHb = svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose();
    // 做舒尔补，且其他地方设置为0，因为要传递给下一帧，所以其他位置都含有上一帧的信息，所以全部为0，只保留仅当前帧的信息
    // Hn = |D C|    ------>|D - CA^-1B  0|
    //      |B A|           |0           0|
    Hn.block(0, 0, a + c, a + c) = Hn.block(0, 0, a + c, a + c) - Hn.block(0, a + c, a + c, b) * invHb * Hn.block(a + c, 0, b, a + c);
    Hn.block(a + c, a + c, b, b) = Eigen::MatrixXd::Zero(b, b);
    Hn.block(0, a + c, a + c, b) = Eigen::MatrixXd::Zero(a + c, b);
    Hn.block(a + c, 0, b, a + c) = Eigen::MatrixXd::Zero(b, a + c);

    // Inverse reorder
    // a*  | ac* | 0       a*  | 0 | ac*
    // ca* | c*  | 0  -->  0   | 0 | 0
    // 0   | 0   | 0       ca* | 0 | c*
    // 3. 再返回去
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    if (a > 0)
    {
        res.block(0, 0, a, a) = Hn.block(0, 0, a, a);
        res.block(0, a, a, b) = Hn.block(0, a + c, a, b);
        res.block(a, 0, b, a) = Hn.block(a + c, 0, b, a);
    }
    if (a > 0 && c > 0)
    {
        res.block(0, a + b, a, c) = Hn.block(0, a, a, c);
        res.block(a + b, 0, c, a) = Hn.block(a, 0, c, a);
    }
    if (c > 0)
    {
        // 把H矩阵分成 2*2 的块矩阵 那么Hn为
        // |A B|   <----  |D C|
        // |C D|          |B A|
        res.block(a + b, a + b, c, c) = Hn.block(a, a, c, c);
        res.block(a + b, a, c, b) = Hn.block(a, a + c, c, b);
        res.block(a, a + b, b, c) = Hn.block(a + c, a, b, c);
    }

    res.block(a, a, b, b) = Hn.block(a + c, a + c, b, b);

    return res;
}

Eigen::MatrixXd Optimizer::Condition(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    // Size of block before block to condition
    const int a = start;
    // Size of block to condition
    const int b = end + 1 - start;

    // Set to zero elements related to block b(start:end, start:end)
    // a  | ab | ac       a  | 0 | ac
    // ba | b  | bc  -->  0  | 0 | 0
    // ca | cb | c        ca | 0 | c

    Eigen::MatrixXd Hn = H;

    Hn.block(a, 0, b, H.cols()) = Eigen::MatrixXd::Zero(b, H.cols());
    Hn.block(0, a, H.rows(), b) = Eigen::MatrixXd::Zero(H.rows(), b);

    return Hn;
}

Eigen::MatrixXd Optimizer::Sparsify(const Eigen::MatrixXd &H, const int &start1, const int &end1, const int &start2, const int &end2)
{
    // Goal: remove link between a and b
    // p(a, b, c) ~ p(a, b, c)*p(a|c)/p(a|b, c) => H' = H + H1 - H2
    // H1: marginalize b and condition c
    // H2: condition b and c
    Eigen::MatrixXd Hac = Marginalize(H, start2, end2);
    Eigen::MatrixXd Hbc = Marginalize(H, start1, end1);
    Eigen::MatrixXd Hc = Marginalize(Hac, start1, end1);

    return Hac + Hbc - Hc;
}

/** 
 * @brief imu初始化优化，LocalMapping::InitializeIMU中使用，其中kf的位姿是固定不变的
 * @param pMap 地图
 * @param Rwg 重力方向到速度方向的转角
 * @param scale 尺度（输出cout用）
 * @param bg 陀螺仪偏置（输出cout用）
 * @param ba 加速度计偏置（输出cout用）
 * @param bMono 是否为单目
 * @param covInertial 惯导协方差矩阵(暂时没用，9*9的0矩阵)
 * @param bFixedVel 是否固定速度不优化
 * @param bGauss  没用
 * @param priorG 陀螺仪偏置的信息矩阵系数
 * @param priorA 加速度计偏置的信息矩阵系数
 */
void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba,
                                        bool bMono, Eigen::MatrixXd &covInertial, bool bFixedVel, bool bGauss, float priorG, float priorA)
{
    Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
    int its = 200; // Check number of iterations
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames(); // 获取所有关键帧

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    if (priorG != 0.f)
        solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    // 1. 确定关键帧节点（锁住的位姿及可优化的速度）
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        // 跳过id大于当前地图最大id的关键帧
        if (pKFi->mnId > maxKFid)
            continue;
        VertexPose *VP = new VertexPose(pKFi); // 继承于public g2o::BaseVertex<6, ImuCamPose>
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity *VV = new VertexVelocity(pKFi); // 继承于public g2o::BaseVertex<3, Eigen::Vector3d>
        VV->setId(maxKFid + (pKFi->mnId) + 1);
        if (bFixedVel)
            VV->setFixed(true);
        else
            VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    // 2. 确定偏置节点，陀螺仪与加速度计
    VertexGyroBias *VG = new VertexGyroBias(vpKFs.front()); // 继承于public g2o::BaseVertex<3, Eigen::Vector3d>
    VG->setId(maxKFid * 2 + 2);
    if (bFixedVel)
        VG->setFixed(true);
    else
        VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias *VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid * 2 + 3);
    if (bFixedVel)
        VA->setFixed(true);
    else
        VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    // 3. 添加关于加速度计与陀螺仪偏置的边，这两个边加入是保证第一帧的偏置为0
    EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F)); // 继承于public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexGyroBias>
    epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);

    EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F)); // 继承于public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexAccBias>
    epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    // 4. 添加关于重力方向与尺度的节点
    VertexGDir *VGDir = new VertexGDir(Rwg); // 继承于public g2o::BaseVertex<2, GDirection>
    VGDir->setId(maxKFid * 2 + 4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale *VS = new VertexScale(scale);
    VS->setId(maxKFid * 2 + 5);
    VS->setFixed(!bMono); // Fixed for stereo case(双目就固定尺度)
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    // 5. imu信息链接重力方向与尺度信息
    vector<EdgeInertialGS *> vpei; // 后面虽然加入了边，但是没有用到，应该调试用的
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
    vppUsedKF.reserve(vpKFs.size()); // 后面虽然加入了关键帧，但是没有用到，应该调试用的

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];

        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;
            // 到这里的条件是pKFi是好的，并且它有上一个关键帧，且他们的id要小于最大id
            // 5.1 检查节点指针是否为空
            // 将pKFi偏置设定为上一关键帧的偏置
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
            g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid * 2 + 2);
            g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
            g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
            g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
            if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", " << VV1 << ", " << VG << ", " << VA << ", " << VP2 << ", " << VV2 << ", " << VGDir << ", " << VS << endl;

                continue;
            }
            // 5.2 这是一个大边。。。。包含了上面所有信息，注意到前面的两个偏置也做了两个一元边加入
            EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    std::set<g2o::HyperGraph::Edge *> setEdges = optimizer.edges();

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // 取数
    scale = VS->estimate();
    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid * 2 + 2));
    VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid * 2 + 3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();
    scale = VS->estimate();

    IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);
    Rwg = VGDir->estimate().Rwg;

    cv::Mat cvbg = Converter::toCvMat(bg);

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for (size_t i = 0; i < N; i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;

        VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
        Eigen::Vector3d Vw = VV->estimate(); // Velocity is scaled after
        pKFi->SetVelocity(Converter::toCvMat(Vw));

        if (cv::norm(pKFi->GetGyroBias() - cvbg) > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);
    }
}

/** 
 * @brief 跟参数最多的那个同名函数不同的地方在于很多节点不可选是否固定，优化的目标有：
 * 速度，偏置
 * @param pMap 地图
 * @param bg 陀螺仪偏置（输出cout用）
 * @param ba 加速度计偏置（输出cout用）
 * @param priorG 陀螺仪偏置的信息矩阵系数
 * @param priorA 加速度计偏置的信息矩阵系数
 */
void Optimizer::InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
{
    int its = 200; // Check number of iterations
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e3); // 是否是公式中的λ？

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    // 1. 确定关键帧节点（锁住的位姿及可优化的速度）
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        // 跳过id大于当前地图最大id的关键帧
        if (pKFi->mnId > maxKFid)
            continue;
        VertexPose *VP = new VertexPose(pKFi); // 继承于public g2o::BaseVertex<6, ImuCamPose>
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity *VV = new VertexVelocity(pKFi); // 继承于public g2o::BaseVertex<3, Eigen::Vector3d>
        VV->setId(maxKFid + (pKFi->mnId) + 1);
        VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    // 2. 确定偏置节点，陀螺仪与加速度计
    VertexGyroBias *VG = new VertexGyroBias(vpKFs.front()); // 继承于public g2o::BaseVertex<3, Eigen::Vector3d>
    VG->setId(maxKFid * 2 + 2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias *VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid * 2 + 3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // prior acc bias
    // 3. 添加关于加速度计与陀螺仪偏置的边
    EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F)); // 继承于public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexGyroBias>
    epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F)); // 继承于public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexAccBias>
    epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    // 4. 添加关于重力方向与尺度的节点，这里固定了这两个变量
    VertexGDir *VGDir = new VertexGDir(Eigen::Matrix3d::Identity()); // 继承于public g2o::BaseVertex<2, GDirection>
    VGDir->setId(maxKFid * 2 + 4);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);
    VertexScale *VS = new VertexScale(1.0);
    VS->setId(maxKFid * 2 + 5);
    VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    // 5. imu信息链接重力方向与尺度信息
    vector<EdgeInertialGS *> vpei; // 后面虽然加入了边，但是没有用到，应该调试用的
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
    vppUsedKF.reserve(vpKFs.size()); // 后面虽然加入了关键帧，但是没有用到，应该调试用的

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];

        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {

            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;
            // 到这里的条件是pKFi是好的，并且它有上一个关键帧，且他们的id要小于最大id
            // 5.1 检查节点指针是否为空
            // 将pKFi偏置设定为上一关键帧的偏置
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
            g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid * 2 + 2);
            g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
            g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
            g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
            if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", " << VV1 << ", " << VG << ", " << VA << ", " << VP2 << ", " << VV2 << ", " << VGDir << ", " << VS << endl;

                continue;
            }
            // 5.2 这是一个大边。。。。包含了上面所有信息，注意到前面的两个偏置也做了两个一元边加入
            EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // Recover optimized data
    // Biases
    // 取数
    VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid * 2 + 2));
    VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid * 2 + 3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();

    IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);

    cv::Mat cvbg = Converter::toCvMat(bg);

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for (size_t i = 0; i < N; i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;

        VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
        Eigen::Vector3d Vw = VV->estimate();
        pKFi->SetVelocity(Converter::toCvMat(Vw));

        if (cv::norm(pKFi->GetGyroBias() - cvbg) > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);
    }
}

/** 
 * @brief 跟同名函数不同的地方在于输入是kfs，而不是地图，优化的目标有：
 * 速度，偏置
 * @param vpKFs 要更新的帧
 * @param bg 陀螺仪偏置（输出cout用）
 * @param ba 加速度计偏置（输出cout用）
 * @param priorG 陀螺仪偏置的信息矩阵系数
 * @param priorA 加速度计偏置的信息矩阵系数
 */
void Optimizer::InertialOptimization(vector<KeyFrame *> vpKFs, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
{
    int its = 200; // Check number of iterations
    long unsigned int maxKFid = vpKFs[0]->GetMap()->GetMaxKFid();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        //if(pKFi->mnId>maxKFid)
        //    continue;
        VertexPose *VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity *VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid + (pKFi->mnId) + 1);
        VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    VertexGyroBias *VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid * 2 + 2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias *VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid * 2 + 3);
    VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    EdgePriorAcc *epa = new EdgePriorAcc(cv::Mat::zeros(3, 1, CV_32F));
    epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro *epg = new EdgePriorGyro(cv::Mat::zeros(3, 1, CV_32F));
    epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    VertexGDir *VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
    VGDir->setId(maxKFid * 2 + 4);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);
    VertexScale *VS = new VertexScale(1.0);
    VS->setId(maxKFid * 2 + 5);
    VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<EdgeInertialGS *> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame *, KeyFrame *>> vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];

        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + (pKFi->mPrevKF->mnId) + 1);
            g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex *VG = optimizer.vertex(maxKFid * 2 + 2);
            g2o::HyperGraph::Vertex *VA = optimizer.vertex(maxKFid * 2 + 3);
            g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
            g2o::HyperGraph::Vertex *VS = optimizer.vertex(maxKFid * 2 + 5);
            if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", " << VV1 << ", " << VG << ", " << VA << ", " << VP2 << ", " << VV2 << ", " << VGDir << ", " << VS << endl;

                continue;
            }
            EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF, pKFi));
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid * 2 + 2));
    VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid * 2 + 3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();

    IMU::Bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);

    cv::Mat cvbg = Converter::toCvMat(bg);

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for (size_t i = 0; i < N; i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;

        VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + (pKFi->mnId) + 1));
        Eigen::Vector3d Vw = VV->estimate();
        pKFi->SetVelocity(Converter::toCvMat(Vw));

        if (cv::norm(pKFi->GetGyroBias() - cvbg) > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);
    }
}

/** 
 * @brief 优化重力方向与尺度，LocalMapping::ScaleRefinement()中使用，优化目标有：
 * 重力方向与尺度
 * @param pMap 地图
 * @param Rwg 重力方向到速度方向的转角
 * @param scale 尺度
 */
void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale)
{
    int its = 10;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (all variables are fixed)
    // 2. 添加帧节点，其中包括位姿，速度，两个偏置
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];
        if (pKFi->mnId > maxKFid)
            continue;
        VertexPose *VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity *VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid + 1 + (pKFi->mnId));
        VV->setFixed(true);
        optimizer.addVertex(VV);

        // Vertex of fixed biases
        VertexGyroBias *VG = new VertexGyroBias(vpKFs.front());
        VG->setId(2 * (maxKFid + 1) + (pKFi->mnId));
        VG->setFixed(true);
        optimizer.addVertex(VG);
        VertexAccBias *VA = new VertexAccBias(vpKFs.front());
        VA->setId(3 * (maxKFid + 1) + (pKFi->mnId));
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }
    // 3. 添加重力方向与尺度的节点，为优化对象
    // Gravity and scale
    VertexGDir *VGDir = new VertexGDir(Rwg);
    VGDir->setId(4 * (maxKFid + 1));
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale *VS = new VertexScale(scale);
    VS->setId(4 * (maxKFid + 1) + 1);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    // Graph edges
    // 4. 添加边
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];

        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;

            g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex((maxKFid + 1) + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex((maxKFid + 1) + pKFi->mnId);
            g2o::HyperGraph::Vertex *VG = optimizer.vertex(2 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VA = optimizer.vertex(3 * (maxKFid + 1) + pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(4 * (maxKFid + 1));
            g2o::HyperGraph::Vertex *VS = optimizer.vertex(4 * (maxKFid + 1) + 1);
            if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                Verbose::PrintMess("Error" + to_string(VP1->id()) + ", " + to_string(VV1->id()) + ", " + to_string(VG->id()) + ", " + to_string(VA->id()) + ", " + to_string(VP2->id()) + ", " + to_string(VV2->id()) + ", " + to_string(VGDir->id()) + ", " + to_string(VS->id()), Verbose::VERBOSITY_NORMAL);

                continue;
            }
            EdgeInertialGS *ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // Recover optimized data
    scale = VS->estimate();
    Rwg = VGDir->estimate().Rwg;
}

/** 
 * @brief 没有被用到，暂时不看
 */
void Optimizer::MergeBundleAdjustmentVisual(KeyFrame *pCurrentKF, vector<KeyFrame *> vpWeldingKFs, vector<KeyFrame *> vpFixedKFs, bool *pbStopFlag)
{
    vector<MapPoint *> vpMPs;

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;
    set<KeyFrame *> spKeyFrameBA;

    // Set not fixed KeyFrame vertices
    for (KeyFrame *pKFi : vpWeldingKFs)
    {
        if (pKFi->isBad())
            continue;

        pKFi->mnBALocalForKF = pCurrentKF->mnId;

        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;

        set<MapPoint *> spViewMPs = pKFi->GetMapPoints();
        for (MapPoint *pMPi : spViewMPs)
        {
            if (pMPi)
                if (!pMPi->isBad())
                    if (pMPi->mnBALocalForKF != pCurrentKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForKF = pCurrentKF->mnId;
                    }
        }

        spKeyFrameBA.insert(pKFi);
    }

    // Set fixed KeyFrame vertices
    for (KeyFrame *pKFi : vpFixedKFs)
    {
        if (pKFi->isBad())
            continue;

        pKFi->mnBALocalForKF = pCurrentKF->mnId;

        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;

        set<MapPoint *> spViewMPs = pKFi->GetMapPoints();
        for (MapPoint *pMPi : spViewMPs)
        {
            if (pMPi)
                if (!pMPi->isBad())
                    if (pMPi->mnBALocalForKF != pCurrentKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForKF = pCurrentKF->mnId;
                    }
        }

        spKeyFrameBA.insert(pKFi);
    }

    const int nExpectedSize = (vpWeldingKFs.size() + vpFixedKFs.size()) * vpMPs.size();

    vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for (unsigned int i = 0; i < vpMPs.size(); ++i)
    {
        MapPoint *pMPi = vpMPs[i];
        if (pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMPi->GetWorldPos()));
        const int id = pMPi->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, tuple<int, int>> observations = pMPi->GetObservations();
        int nEdges = 0;
        //SET EDGES
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            //cout << "--KF view init" << endl;

            KeyFrame *pKF = mit->first;
            if (spKeyFrameBA.find(pKF) == spKeyFrameBA.end() || pKF->isBad() || pKF->mnId > maxKFid || pKF->mnBALocalForKF != pCurrentKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            //cout << "-- KF view exists" << endl;
            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];
            //cout << "-- KeyPoint loads" << endl;

            if (pKF->mvuRight[get<0>(mit->second)] < 0) //Monocular
            {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                //cout << "-- Sigma loads" << endl;

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                //cout << "-- Calibration loads" << endl;

                optimizer.addEdge(e);
                //cout << "-- Edge added" << endl;

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMPi);
                //cout << "-- Added to vector" << endl;
            }
            else // RGBD or Stereo
            {
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber3D);

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMPi);
            }
            //cout << "-- End to load point" << endl;
        }
    }

    //cout << "End to load MPs" << endl;

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    //cout << "End the first optimization" << endl;

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore)
    {

        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Optimize again without the outliers

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        //cout << "End the second optimization (without outliers)" << endl;
    }

    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pCurrentKF->GetMap()->mMutexMapUpdate);

    if (!vToErase.empty())
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    //cout << "End to erase observations" << endl;

    // Recover optimized data

    //Keyframes
    for (KeyFrame *pKFi : vpWeldingKFs)
    {
        if (pKFi->isBad())
            continue;

        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKFi->SetPose(Converter::toCvMat(SE3quat));
    }
    //cout << "End to update the KeyFrames" << endl;

    //Points
    for (MapPoint *pMPi : vpMPs)
    {
        if (pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMPi->mnId + maxKFid + 1));
        pMPi->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMPi->UpdateNormalAndDepth();
    }
}

/** 
 * @brief 在LoopClosing::MergeLocal() LoopClosing::MergeLocal2()中地图融合使用。
 * 优化目标：相关帧的位姿，速度，偏置，还有涉及点的坐标，可以理解为跨地图的局部窗口优化
 * @param pCurrKF 与mpCurrentKF不同的是mpCurrentKF是在回环线程迭代的那个关键帧，运行到此处有可能有新的关键帧，pCurrKF就是最新的关键帧
 * @param pMergeKF 与mpCurrentKF匹配的候选关键帧
 * @param pbStopFlag false
 * @param pMap mpCurrentKF->GetMap()
 * @param corrPoses map<KeyFrame *, g2o::Sim3, std::less<KeyFrame *>, Eigen::aligned_allocator<std::pair<const KeyFrame *, g2o::Sim3>>> 空的
 */
void Optimizer::MergeInertialBA(KeyFrame *pCurrKF, KeyFrame *pMergeKF, bool *pbStopFlag, Map *pMap, LoopClosing::KeyFrameAndPose &corrPoses)
{
    const int Nd = 6;
    // 理论上这是最大的id
    const unsigned long maxKFid = pCurrKF->mnId;

    // For cov KFS, inertial parameters are not optimized
    // 这里面的关键帧不更新惯导参数（更正：别看上面英文瞎说。。。vpOptimizableCovKFs这里面的帧正常优化，只不过不统计他们的mp点做优化）
    const int maxCovKF = 30;
    vector<KeyFrame *> vpOptimizableCovKFs;
    vpOptimizableCovKFs.reserve(maxCovKF);

    // Add sliding window for current KF
    // 弄个滑动窗口
    vector<KeyFrame *> vpOptimizableKFs;
    vpOptimizableKFs.reserve(2 * Nd);
    vpOptimizableKFs.push_back(pCurrKF);
    pCurrKF->mnBALocalForKF = pCurrKF->mnId;
    // 1. 优化前的处理
    // 1.1 一直放一直放，一直放到没有上一个关键帧，这里面包含了pCurrKF最近的6帧，从晚到早排列，不过有可能存不满
    for (int i = 1; i < Nd; i++)
    {
        if (vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId; // 添加标识，避免重复添加
        }
        else
            break;
    }
    // 1.2 如果vpOptimizableKFs中最早的一帧前面还有，往不更新惯导参数的序列中添加
    // 否则把最后一个取出来放到不更新惯导参数的序列中
    if (vpOptimizableKFs.back()->mPrevKF)
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBALocalForKF = pCurrKF->mnId;
    }
    else
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }
    // 取出固定的帧的Twc（这两行我注释掉了，没用）
    // KeyFrame* pKF0 = vpOptimizableCovKFs.back();
    // cv::Mat Twc0 = pKF0->GetPoseInverse();

    // Add temporal neighbours to merge KF (previous and next KFs)
    // 1.3 把匹配的融合关键帧也放进来准备一起优化
    vpOptimizableKFs.push_back(pMergeKF);
    pMergeKF->mnBALocalForKF = pCurrKF->mnId;

    // Previous KFs
    // 1.4 再放进来3个pMergeKF附近的帧，有可能放不满
    for (int i = 1; i < (Nd / 2); i++)
    {
        if (vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    // We fix just once the old map
    // 1.5 类似于上面的做法如果有前一个关键帧放入lFixedKeyFrames，否则从vpOptimizableKFs取出，注意这里防止重复添加的标识又多了一个变量
    list<KeyFrame *> lFixedKeyFrames;
    if (vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF = pCurrKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF = 0;
        vpOptimizableKFs.back()->mnBAFixedForKF = pCurrKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Next KFs
    // 1.6 再添加一个pMergeKF的下一个关键帧
    if (pMergeKF->mNextKF)
    {
        vpOptimizableKFs.push_back(pMergeKF->mNextKF);
        vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;

        while (vpOptimizableKFs.size() < (2 * Nd))
        {
            if (vpOptimizableKFs.back()->mNextKF)
            {
                vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mNextKF);
                vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
            }
            else
                break;
        }
    }
    // 1.7 数量不够时，添加最后一个的下一帧。这里有问题，如果1.6添加失败相当于上一个又添加下一个，就有重复添加了，这里修复一下
    // 这里原本的目的是想添加pMergeKF后面的，如果没有就不添加了，所以应该吧这段代码移至1.6 里面
    // while(vpOptimizableKFs.size()<(2*Nd))
    // {
    //     if(vpOptimizableKFs.back()->mNextKF)
    //     {
    //         vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mNextKF);
    //         vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
    //     }
    //     else
    //         break;
    // }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by optimizable keyframes
    // 2. 帧弄完了该弄点了，将优化的帧的点存入lLocalMapPoints
    list<MapPoint *> lLocalMapPoints;
    map<MapPoint *, int> mLocalObs; // 统计了在这些帧中点被观测的次数
    for (int i = 0; i < N; i++)
    {
        vector<MapPoint *> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            // Using mnBALocalForKF we avoid redundance here, one MP can not be added several times to lLocalMapPoints
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pCurrKF->mnId)
                    {
                        mLocalObs[pMP] = 1;
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pCurrKF->mnId; // 防止重复添加
                    }
                    else
                        mLocalObs[pMP]++;
        }
    }

    std::vector<std::pair<MapPoint *, int>> pairs;
    pairs.reserve(mLocalObs.size());
    // MARK 还能这么添加，map里的元素直接添加到pair里了
    for (auto itr = mLocalObs.begin(); itr != mLocalObs.end(); ++itr)
        pairs.push_back(*itr);
    // 按照观测数从小到大排列，这里细节了，下面想要的是除了局部帧之外的可以看到局部mp的帧，点被观测次数越低表示这个点越边缘，得到的帧越是想要的
    sort(pairs.begin(), pairs.end(), sortByVal);

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // 2.1 继续添加帧
    int i = 0;
    for (vector<pair<MapPoint *, int>>::iterator lit = pairs.begin(), lend = pairs.end(); lit != lend; lit++, i++)
    {
        // 上限，添加30个
        if (i >= maxCovKF)
            break;
        map<KeyFrame *, tuple<int, int>> observations = lit->first->GetObservations();
        // 每个点对应的观测帧中最多添加一个
        for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;
            // 在前面没添加过
            if (pKFi->mnBALocalForKF != pCurrKF->mnId && pKFi->mnBAFixedForKF != pCurrKF->mnId) // If optimizable or already included...
            {
                pKFi->mnBALocalForKF = pCurrKF->mnId;
                if (!pKFi->isBad())
                {
                    // 固定惯导参数序列中加入
                    vpOptimizableCovKFs.push_back(pKFi);
                    break;
                }
            }
        }
    }
    // 3. 总算添加完了。。。开始构建优化了
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e3); // TODO uncomment

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // 4. 做一些节点，为优化做准备
    // Set Local KeyFrame vertices
    N = vpOptimizableKFs.size();
    for (int i = 0; i < N; i++)
    {
        KeyFrame *pKFi = vpOptimizableKFs[i];

        VertexPose *VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if (pKFi->bImu)
        {
            VertexVelocity *VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias *VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias *VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local cov keyframes vertices
    int Ncov = vpOptimizableCovKFs.size();
    for (int i = 0; i < Ncov; i++)
    {
        KeyFrame *pKFi = vpOptimizableCovKFs[i];

        VertexPose *VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if (pKFi->bImu)
        {
            VertexVelocity *VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias *VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias *VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Fixed KeyFrame vertices
    for (list<KeyFrame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        VertexPose *VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if (pKFi->bImu)
        {
            VertexVelocity *VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->mnId) + 1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias *VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->mnId) + 2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias *VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->mnId) + 3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<EdgeInertial *> vei(N, (EdgeInertial *)NULL);
    vector<EdgeGyroRW *> vegr(N, (EdgeGyroRW *)NULL);
    vector<EdgeAccRW *> vear(N, (EdgeAccRW *)NULL);
    // 5. 第一阶段优化vpOptimizableKFs里面的帧
    for (int i = 0; i < N; i++)
    {
        //cout << "inserting inertial edge " << i << endl;
        KeyFrame *pKFi = vpOptimizableKFs[i];
        // 没有上一个还做个毛积分
        if (!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!!!!", Verbose::VERBOSITY_NORMAL);
            continue;
        }
        if (pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 1);
            g2o::HyperGraph::Vertex *VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 2);
            g2o::HyperGraph::Vertex *VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->mPrevKF->mnId) + 3);
            g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1);
            g2o::HyperGraph::Vertex *VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2);
            g2o::HyperGraph::Vertex *VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3);

            if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", " << VV1 << ", " << VG1 << ", " << VA1 << ", " << VP2 << ", " << VV2 << ", " << VG2 << ", " << VA2 << endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            vei[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            vei[i]->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG1));
            vei[i]->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA1));
            vei[i]->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            vei[i]->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));

            // TODO Uncomment
            g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
            vei[i]->setRobustKernel(rki);
            // 6自由度卡方检验
            rki->setDelta(sqrt(16.92));
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0, VG1);
            vegr[i]->setVertex(1, VG2);
            cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9, 12).colRange(9, 12).inv(cv::DECOMP_SVD);
            Eigen::Matrix3d InfoG;

            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    InfoG(r, c) = cvInfoG.at<float>(r, c);
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0, VA1);
            vear[i]->setVertex(1, VA2);
            cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12, 15).colRange(12, 15).inv(cv::DECOMP_SVD);
            Eigen::Matrix3d InfoA;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    InfoA(r, c) = cvInfoA.at<float>(r, c);
            vear[i]->setInformation(InfoA);
            optimizer.addEdge(vear[i]);
        }
        else
            Verbose::PrintMess("ERROR building inertial edge", Verbose::VERBOSITY_NORMAL);
    }

    Verbose::PrintMess("end inserting inertial edges", Verbose::VERBOSITY_NORMAL);

    // 6. 添加MP的节点
    // Set MapPoint vertices
    const int nExpectedSize = (N + Ncov + lFixedKeyFrames.size()) * lLocalMapPoints.size();

    // Mono
    vector<EdgeMono *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<EdgeStereo *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid * 5; // TODO: should be  maxKFid*4;

    Verbose::PrintMess("start inserting MPs", Verbose::VERBOSITY_NORMAL);
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        if (!pMP)
            continue;

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        // 添加节点
        unsigned long id = pMP->mnId + iniMPid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

        // Create visual constraints
        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (!pKFi)
                continue;

            if ((pKFi->mnBALocalForKF != pCurrKF->mnId) && (pKFi->mnBAFixedForKF != pCurrKF->mnId))
                continue;

            if (pKFi->mnId > maxKFid)
            {
                Verbose::PrintMess("ID greater than current KF is", Verbose::VERBOSITY_NORMAL);
                continue;
            }

            // 如果MP或者KF节点不存在
            if (optimizer.vertex(id) == NULL || optimizer.vertex(pKFi->mnId) == NULL)
                continue;

            if (!pKFi->isBad())
            {
                // 7. 添加视觉重投影误差的边
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[get<0>(mit->second)];

                if (pKFi->mvuRight[get<0>(mit->second)] < 0) // Monocular observation
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono *e = new EdgeMono();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // stereo observation
                {
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo *e = new EdgeStereo();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }
    // 8. 优化！！！
    if (pbStopFlag)
        if (*pbStopFlag)
            return;
    optimizer.initializeOptimization();
    optimizer.optimize(3);
    if (pbStopFlag)
        if (!*pbStopFlag)
            optimizer.optimize(5);

    optimizer.setForceStopFlag(pbStopFlag);
    // 记录下哪些被干掉
    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        EdgeMono *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > chi2Mono2)
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Stereo
    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        EdgeStereo *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > chi2Stereo2)
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex and erase outliers
    // 9. 对于误差大的边直接解除联系，被认为是匹配错的点
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    if (!vToErase.empty())
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // 10. 取出数据，特别的向corrPoses保存结果
    // Recover optimized data
    //Keyframes
    for (int i = 0; i < N; i++)
    {
        KeyFrame *pKFi = vpOptimizableKFs[i];

        VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
        cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
        pKFi->SetPose(Tcw);

        cv::Mat Tiw = pKFi->GetPose();
        cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
        g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
        corrPoses[pKFi] = g2oSiw;

        if (pKFi->bImu)
        {
            VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
            pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
            VertexGyroBias *VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
            VertexAccBias *VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]));
        }
    }

    for (int i = 0; i < Ncov; i++)
    {
        KeyFrame *pKFi = vpOptimizableCovKFs[i];

        VertexPose *VP = static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
        cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
        pKFi->SetPose(Tcw);

        cv::Mat Tiw = pKFi->GetPose();
        cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
        g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
        corrPoses[pKFi] = g2oSiw;

        if (pKFi->bImu)
        {
            VertexVelocity *VV = static_cast<VertexVelocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 1));
            pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
            VertexGyroBias *VG = static_cast<VertexGyroBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 2));
            VertexAccBias *VA = static_cast<VertexAccBias *>(optimizer.vertex(maxKFid + 3 * (pKFi->mnId) + 3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3], b[4], b[5], b[0], b[1], b[2]));
        }
    }

    //Points
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + iniMPid + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();
}

} // namespace ORB_SLAM3
