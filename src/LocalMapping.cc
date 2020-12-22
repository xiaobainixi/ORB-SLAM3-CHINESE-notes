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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"

#include <mutex>
#include <chrono>

namespace ORB_SLAM3
{

LocalMapping::LocalMapping(System *pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName) :
                        mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), mbResetRequestedActiveMap(false),
                        mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false), mbAbortBA(false), mbStopped(false),
                        mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true), mbNewInit(false), mIdxInit(0), mScale(1.0), mInitSect(0),
                        mbNotBA1(true), mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9, 9))
{
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;

    mNumLM = 0;
    mNumKFCulling = 0;

    //DEBUG: times and data from LocalMapping in each frame

    strSequence = ""; //_strSeqName;

    //f_lm.open("localMapping_times" + strSequence + ".txt");
    /*f_lm.open("localMapping_times.txt");

f_lm << "# Timestamp KF, Num CovKFs, Num KFs, Num RecentMPs, Num MPs, processKF, MPCulling, CreateMP, SearchNeigh, BA, KFCulling, [numFixKF_LBA]" << endl;
f_lm << fixed;*/
}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker = pTracker;
}

// 开始整吧～整个类就靠这个活着呢
void LocalMapping::Run()
{

    mbFinished = false;

    while (1)
    {
        // Tracking will see that Local Mapping is busy
        // 告诉Tracking，LocalMapping正处于繁忙状态，
        // LocalMapping线程处理的关键帧都是Tracking线程发过的
        // 在LocalMapping线程还没有处理完关键帧之前Tracking线程最好不要发送太快
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if (CheckNewKeyFrames() && !mbBadImu)
        {
            // std::cout << "LM" << std::endl;
            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

            // BoW conversion and insertion in Map
            // 1.计算关键帧特征点的BoW映射，将关键帧插入地图
            ProcessNewKeyFrame();
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            // Check recent MapPoints
            // 2.剔除ProcessNewKeyFrame函数中引入的不合格MapPoints
            MapPointCulling();
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

            // Triangulate new MapPoints
            // 3.相机运动过程中与相邻关键帧通过三角化恢复出一些MapPoints
            // orbslam2里面23步骤是反过来的，感觉现在这么做更好，因为MapPointCulling处理是根据最近几帧的情况，新建立的MP仅限于当前帧，基本上属于白白扫描浪费循环所用的时间
            CreateNewMapPoints();
            std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

            // Save here:
            // # Cov KFs
            // # tot Kfs
            // # recent added MPs
            // # tot MPs
            // # localMPs in LBA
            // # fixedKFs in LBA

            mbAbortBA = false;
            // 已经处理完队列中的最后的一个关键帧
            if (!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                // 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints，来回融合。
                // 先完成相邻关键帧与当前关键帧的MP的融合（在相邻关键帧中查找当前关键帧的MP），再完成当前关键帧与相邻关键帧的MP的融合（在当前关键帧中查找当前相邻关键帧的MP）
                SearchInNeighbors();
            }

            std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point t5 = t4, t6 = t4;
            // mbAbortBA = false;  // orbslam2在这里，但是函数SearchInNeighbors里面用到了mbAbortBA，所以放到上面更合适

            //DEBUG--
            /*f_lm << setprecision(0);
            f_lm << mpCurrentKeyFrame->mTimeStamp*1e9 << ", ";
            f_lm << mpCurrentKeyFrame->GetVectorCovisibleKeyFrames().size() << ", ";
            f_lm << mpCurrentKeyFrame->GetMap()->GetAllKeyFrames().size() << ", ";
            f_lm << mlpRecentAddedMapPoints.size() << ", ";
            f_lm << mpCurrentKeyFrame->GetMap()->GetAllMapPoints().size() << ", ";*/
            //--
            int num_FixedKF_BA = 0; // 应该是调试用的，不影响实际使用
            // 从处理完新的关键帧到现在如果没有新关键帧进来，且没有停止
            if (!CheckNewKeyFrames() && !stopRequested())
            {
                // 当前地图的关键帧数要大于2
                if (mpAtlas->KeyFramesInMap() > 2)
                {
                    // imu模式且IMU初始化之后
                    if (mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
                    {
                        float dist = cv::norm(mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()) +
                                        cv::norm(mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter());
                        // mTinit自初始化距当前帧的时间
                        if (dist > 0.05)
                            mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
                        if (!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            if ((mTinit < 10.f) && (dist < 0.02))
                            {
                                cout << "Not enough motion for initializing. Reseting..." << endl;
                                unique_lock<mutex> lock(mMutexReset);
                                mbResetRequestedActiveMap = true;
                                mpMapToReset = mpCurrentKeyFrame->GetMap();
                                mbBadImu = true;
                            }
                        }
                        // 与localmap匹配的点数
                        bool bLarge = ((mpTracker->GetMatchesInliers() > 75) && mbMonocular) || ((mpTracker->GetMatchesInliers() > 100) && !mbMonocular);
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(), bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                    }
                    else // IMU初始化之前或者正常单目时
                    {
                        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                        // 局部地图BA
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(), num_FixedKF_BA);
                        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    }
                }

                t5 = std::chrono::steady_clock::now();

                // Initialize IMU here
                // 初始化IMU（一阶段）
                if (!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
                {
                    if (mbMonocular)
                        InitializeIMU(1e2, 1e10, true);
                    else
                        InitializeIMU(1e2, 1e5, true);
                }

                // Check redundant local Keyframes
                // 检测并剔除当前帧相邻的关键帧中冗余的关键帧
                // 剔除的标准是：该关键帧的90%的MapPoints可以被其它关键帧观测到
                // trick!
                // Tracking中先把关键帧交给LocalMapping线程
                // 并且在Tracking中InsertKeyFrame函数的条件比较松，交给LocalMapping线程的关键帧会比较密
                // 在这里再删除冗余的关键帧
                KeyFrameCulling();

                t6 = std::chrono::steady_clock::now();
                // IMU相关 mTinit距第一次初始化成功的时间
                // 当时间小于100s时，再多进行两次优化，输入priorG与priorA越来越小直到为0
                if ((mTinit < 100.0f) && mbInertial)
                {
                    if (mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState == Tracking::OK) // Enter here everytime local-mapping is called
                    {
                        // 初始化imu（二阶段）
                        if (!mpCurrentKeyFrame->GetMap()->GetIniertialBA1())
                        {
                            if (mTinit > 5.0f)
                            {
                                cout << "start VIBA 1" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA1();
                                if (mbMonocular)
                                    InitializeIMU(1.f, 1e5, true); // 1.f, 1e5
                                else
                                    InitializeIMU(1.f, 1e5, true); // 1.f, 1e5

                                cout << "end VIBA 1" << endl;
                            }
                        }
                        //else if (mbNotBA2){  初始化imu（三阶段）
                        else if (!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            if (mTinit > 15.0f)
                            {
                                // 15.0f
                                cout << "start VIBA 2" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA2();
                                if (mbMonocular)
                                    InitializeIMU(0.f, 0.f, true); // 0.f, 0.f
                                else
                                    InitializeIMU(0.f, 0.f, true);

                                cout << "end VIBA 2" << endl;
                            }
                        }

                        // scale refinement
                        // 关键帧小于100，在这里的时间段内时多次进行尺度更新
                        if (((mpAtlas->KeyFramesInMap()) <= 100) &&
                            ((mTinit > 25.0f && mTinit < 25.5f) ||
                                (mTinit > 35.0f && mTinit < 35.5f) ||
                                (mTinit > 45.0f && mTinit < 45.5f) ||
                                (mTinit > 55.0f && mTinit < 55.5f) ||
                                (mTinit > 65.0f && mTinit < 65.5f) ||
                                (mTinit > 75.0f && mTinit < 75.5f)))
                        {
                            cout << "start scale ref" << endl;
                            if (mbMonocular)
                                ScaleRefinement();
                            cout << "end scale ref" << endl;
                        }
                    }
                }
            }

            std::chrono::steady_clock::time_point t7 = std::chrono::steady_clock::now();
            // 将当前帧加入到闭环检测队列中
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
            std::chrono::steady_clock::time_point t8 = std::chrono::steady_clock::now();

            double t_procKF = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
            double t_MPcull = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
            double t_CheckMP = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t3 - t2).count();
            double t_searchNeigh = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t4 - t3).count();
            double t_Opt = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t5 - t4).count();
            double t_KF_cull = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t6 - t5).count();
            double t_Insert = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t8 - t7).count();

            //DEBUG--
            /*f_lm << setprecision(6);
            f_lm << t_procKF << ", ";
            f_lm << t_MPcull << ", ";
            f_lm << t_CheckMP << ", ";
            f_lm << t_searchNeigh << ", ";
            f_lm << t_Opt << ", ";
            f_lm << t_KF_cull << ", ";
            f_lm << setprecision(0) << num_FixedKF_BA << "\n";*/
            //--
        }
        else if (Stop() && !mbBadImu)
        {
            // Safe area to stop
            while (isStopped() && !CheckFinish())
            {
                // cout << "LM: usleep if is stopped" << endl;
                usleep(3000);
            }
            if (CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if (CheckFinish())
            break;

        // cout << "LM: normal usleep" << endl;
        usleep(3000);
    }

    //f_lm.close();

    SetFinish();
}

/**
 * @brief 插入关键帧
 * @param  pKF 关键帧
 */
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

/**
 * @brief 查看是否有未处理的关键帧
 * @return 结果
 */
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

/**
 * @brief 处理新的关键帧
 */
void LocalMapping::ProcessNewKeyFrame()
{
    //cout << "ProcessNewKeyFrame: " << mlNewKeyFrames.size() << endl;
    // 步骤1：从缓冲队列中取出一帧关键帧
    // Tracking线程向LocalMapping中插入关键帧存在该队列中
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        // 从列表中获得一个等待被插入的关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    // 步骤2：计算该关键帧特征点的Bow映射关系
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    // 步骤3：跟踪局部地图过程中新匹配上的MapPoints和当前关键帧绑定
    // 在TrackLocalMap函数中将局部地图中的MapPoints与当前帧进行了匹配，
    // 但没有对这些匹配上的MapPoints与当前帧进行关联
    const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->isBad())
            {
                // 非当前帧生成的MapPoints
                // 为当前帧在tracking过程跟踪到的MapPoints更新属性
                if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    // 添加观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    // 获得该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();
                    // 加入关键帧后，更新3d点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    // 当前帧生成的MapPoints
                    // 将双目或RGBD跟踪过程中新插入的MapPoints放入mlpRecentAddedMapPoints，等待检查
                    // CreateNewMapPoints函数中通过三角化也会生成MapPoints
                    // 这些MapPoints都会经过MapPointCulling函数的检验
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    // 步骤4：更新关键帧间的连接关系，Covisibility图和Essential图(tree)
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    // 步骤5：将该关键帧插入到地图中
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 * @brief 处理新的关键帧，使队列为空，注意这里只是处理了关键帧，并没有生成MP
 */
void LocalMapping::EmptyQueue()
{
    while (CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

/**
 * @brief 剔除ProcessNewKeyFrame和CreateNewMapPoints函数中引入的质量不好的MapPoints
 * @see VI-B recent map points culling
 */
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if (mbMonocular)
        nThObs = 2;
    else
        nThObs = 3; // MODIFICATION_STEREO_IMU here 3
    const int cnThObs = nThObs;

    int borrar = mlpRecentAddedMapPoints.size(); // 没有用到，应该是调试用的

    while (lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint *pMP = *lit;
        // 步骤1：已经是坏点的MapPoints直接从检查链表中删除
        if (pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if (pMP->GetFoundRatio() < 0.25f) // 被找到/被观测 < 0.25
        {
            // 步骤2：将不满足VI-B条件的MapPoint剔除
            // VI-B 条件1：
            // 跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例需大于25%
            // IncreaseFound / IncreaseVisible < 25%，注意不一定是关键帧。
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs) // 当前帧id-点第一次被观测的id>=2 && mObs次数小于cnThObs
        {
            // 步骤3：将不满足VI-B条件的MapPoint剔除
            // VI-B 条件2：从该点建立开始，到现在已经过了不小于2个关键帧
            // 但是观测到该点的关键帧数却不超过cnThObs帧，那么该点检验不合格
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
        {
            // 步骤4：从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
            // 因此没有SetBadFlag()，仅从队列中删除，放弃继续对该MapPoint的检测
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            lit++;
            borrar--;
        }
    }
    //cout << "erase MP: " << borrar << endl;
}

/**
 * @brief 相机运动过程中和共视程度比较高的关键帧们通过三角化恢复出一些MapPoints，找匹配关系，三角化，验证结果，生成MP
 */
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    // For stereo inertial case
    if (mbMonocular)
        nn = 20;
    // 步骤1：在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻帧vpNeighKFs
    vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    // imu模式下在附近添加更多的帧进来
    if (mbInertial)
    {
        KeyFrame *pKF = mpCurrentKeyFrame;
        int count = 0;
        // 在总数不够且上一关键帧存在，且添加的帧没有超过总数时
        while ((vpNeighKFs.size() <= nn) && (pKF->mPrevKF) && (count++ < nn))
        {
            vector<KeyFrame *>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
            if (it == vpNeighKFs.end())
                vpNeighKFs.push_back(pKF->mPrevKF);
            pKF = pKF->mPrevKF;
        }
    }

    float th = 0.6f;

    ORBmatcher matcher(th, false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));
    // 得到当前关键帧在世界坐标系中的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

    // Search matches with epipolar restriction and triangulate
    // 步骤2：遍历相邻关键帧vpNeighKFs
    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        if (i > 0 && CheckNewKeyFrames()) // && (mnMatchesInliers>50))
            return;

        KeyFrame *pKF2 = vpNeighKFs[i];

        GeometricCamera *pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        // 邻接的关键帧在世界坐标系中的坐标
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        // 基线向量，两个关键帧间的相机位移
        cv::Mat vBaseline = Ow2 - Ow1;
        // 基线长度
        const float baseline = cv::norm(vBaseline);
        // 步骤3：判断相机运动的基线是不是足够长
        if (!mbMonocular)
        {
            // 如果是立体相机，关键帧间距太小时不生成3D点
            if (baseline < pKF2->mb)
                continue;
        }
        else
        {
            // 邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // baseline与景深的比例
            const float ratioBaselineDepth = baseline / medianDepthKF2;
            // 如果特别远(比例特别小)，那么不考虑当前邻接的关键帧，不生成3D点
            if (ratioBaselineDepth < 0.01)
                continue;
        }

        // Compute Fundamental Matrix
        // 步骤4：根据两个关键帧的位姿计算它们之间的基本矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fullfil epipolar constraint
        // 步骤5：通过极线约束限制匹配时的搜索范围，进行特征点匹配
        vector<pair<size_t, size_t>> vMatchedIndices;
        // imu相关，非imu时为false
        bool bCoarse = mbInertial &&
                        ((!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && mpCurrentKeyFrame->GetMap()->GetIniertialBA1()) ||
                        mpTracker->mState == Tracking::RECENTLY_LOST);
        // 通过极线约束的方式找到匹配点（且该点还没有成为MP，注意非单目已经生成的MP这里直接跳过不做匹配，所以最后并不会覆盖掉特征点对应的MP）
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false, bCoarse);

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
        // 步骤6：对每对匹配通过三角化生成3D点,he Triangulate函数差不多
        const int nmatches = vMatchedIndices.size();
        for (int ikp = 0; ikp < nmatches; ikp++)
        {
            // 步骤6.1：取出匹配特征点

            // 当前匹配对在当前关键帧中的索引
            const int &idx1 = vMatchedIndices[ikp].first;
            // 当前匹配对在邻接关键帧中的索引
            const int &idx2 = vMatchedIndices[ikp].second;

            // 当前匹配在当前关键帧中的特征点
            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame->NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                        : (idx1 < mpCurrentKeyFrame->NLeft) ? mpCurrentKeyFrame->mvKeys[idx1]
                                                                                                            : mpCurrentKeyFrame->mvKeysRight[idx1 - mpCurrentKeyFrame->NLeft];
            // mvuRight中存放着极限校准后双目特征点在右目对应的像素横坐标，如果不是基线校准的双目或者没有找到匹配点，其值将为-1（或者rgbd）
            const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur >= 0);  // 前面的判断有些多余
            // 查看点idx1是否为右目的点
            const bool bRight1 = (mpCurrentKeyFrame->NLeft == -1 || idx1 < mpCurrentKeyFrame->NLeft) ? false : true;

            const cv::KeyPoint &kp2 = (pKF2->NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2->NLeft) ? pKF2->mvKeys[idx2]
                                                                                    : pKF2->mvKeysRight[idx2 - pKF2->NLeft];
            // 当前匹配在邻接关键帧中的特征点
            // mvuRight中存放着极限校准后双目特征点在右目对应的像素横坐标，如果不是基线校准的双目或者没有找到匹配点，其值将为-1（或者rgbd）
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur >= 0);
            // 查看点idx2是否为右目的点
            const bool bRight2 = (pKF2->NLeft == -1 || idx2 < pKF2->NLeft) ? false : true;
            // 类似于ORBmatcher::SearchForTriangulation里面的判断
            // 当目前为左右目时，确定两个点所在相机之间的位姿关系
            if (mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2)
            {
                if (bRight1 && bRight2)
                {
                    Rcw1 = mpCurrentKeyFrame->GetRightRotation();
                    Rwc1 = Rcw1.t();
                    tcw1 = mpCurrentKeyFrame->GetRightTranslation();
                    Tcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    Rcw2 = pKF2->GetRightRotation();
                    Rwc2 = Rcw2.t();
                    tcw2 = pKF2->GetRightTranslation();
                    Tcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if (bRight1 && !bRight2)
                {
                    Rcw1 = mpCurrentKeyFrame->GetRightRotation();
                    Rwc1 = Rcw1.t();
                    tcw1 = mpCurrentKeyFrame->GetRightTranslation();
                    Tcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    Rcw2 = pKF2->GetRotation();
                    Rwc2 = Rcw2.t();
                    tcw2 = pKF2->GetTranslation();
                    Tcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if (!bRight1 && bRight2)
                {
                    Rcw1 = mpCurrentKeyFrame->GetRotation();
                    Rwc1 = Rcw1.t();
                    tcw1 = mpCurrentKeyFrame->GetTranslation();
                    Tcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    Rcw2 = pKF2->GetRightRotation();
                    Rwc2 = Rcw2.t();
                    tcw2 = pKF2->GetRightTranslation();
                    Tcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else
                {
                    Rcw1 = mpCurrentKeyFrame->GetRotation();
                    Rwc1 = Rcw1.t();
                    tcw1 = mpCurrentKeyFrame->GetTranslation();
                    Tcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    Rcw2 = pKF2->GetRotation();
                    Rwc2 = Rcw2.t();
                    tcw2 = pKF2->GetTranslation();
                    Tcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
            }

            // Check parallax between rays
            // 步骤6.2：利用匹配点反投影得到视差角
            // 特征点反投影，得到归一化相平面的坐标
            cv::Mat xn1 = pCamera1->unprojectMat(kp1.pt);
            cv::Mat xn2 = pCamera2->unprojectMat(kp2.pt);

            // 由相机坐标系转到世界坐标系，得到视差角余弦值
            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

            // 加1是为了让cosParallaxStereo随便初始化为一个很大的值
            float cosParallaxStereo = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;
            // 步骤6.3：对于双目，利用双目得到视差角，这里有点近似的感觉，默认点在两个相机中央，计算到两个相机的两个射线的夹角
            if (bStereo1)
                cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2, mpCurrentKeyFrame->mvDepth[idx1]));
            else if (bStereo2)
                cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

            // 步骤6.4：三角化恢复3D点
            cv::Mat x3D;
            // 
            // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)表明视线角正常
            // cosParallaxRays<cosParallaxStereo表明前后帧视线角比双目视线角大，所以用前后帧三角化而来，反之使用双目的，如果没有双目则跳过
            // 视差角度小时用三角法恢复3D点，视差角大时（离相机近）用双目恢复3D点（双目以及深度有效）
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
                (bStereo1 || bStereo2 || (cosParallaxRays < 0.9998 && mbInertial) || (cosParallaxRays < 0.9998 && !mbInertial)))
            {
                // Linear Triangulation Method
                // 见Initializer.cpp的Triangulate函数,A矩阵构建的方式类似，不同的是乘的反对称矩阵那个是像素坐标构成的，而这个是归一化坐标构成的
                // Pc = Tcw*Pw， 此处Tcw行数默认为3，因为不会计算第4行所以没有去掉，
                // 左右两面乘Pc的反对称矩阵 [Pc]x * Tcw *Pw = 0 构成了A矩阵，中间涉及一个尺度a，因为都是归一化平面，但右面是0所以直接可以约掉不影响最后的尺度
                //  0 -1 y    Tcw.row(0)     -Tcw.row(1) + y*Tcw.row(2)
                //  1 0 -x *  Tcw.row(1)  =   Tcw.row(0) - x*Tcw.row(2)
                // -y x  0    Tcw.row(2)    x*Tcw.row(1) - y*Tcw.row(0)
                // 发现上述矩阵线性相关，所以取前两维，两个点构成了4行的矩阵，就是如下的操作，求出的是4维的结果[X,Y,Z,A]，所以需要除以最后一维使之为1，就成了[X,Y,Z,1]这种齐次形式
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
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
            }
            else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
            {
                continue; //No stereo and very low parallax
            }
            // 转为行向量
            cv::Mat x3Dt = x3D.t();

            if (x3Dt.empty())
                continue;
            //Check triangulation in front of cameras
            // 步骤6.5：检测生成的3D点是否在相机前方
            float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
            if (z1 <= 0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
            if (z2 <= 0)
                continue;

            //Check reprojection error in first keyframe
            // 步骤6.6：计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
            const float invz1 = 1.0 / z1;

            if (!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1, y1, z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;
                // 2自由度卡方检验
                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1; // 在右目中通过计算所在的位置
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur; // 在右目中实际的位置
                // 3自由度卡方检验
                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            // 计算3D点在另一个关键帧下的重投影误差，操作与上相同
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
            const float invz2 = 1.0 / z2;
            if (!bStereo2)
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2, y2, z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            // 步骤6.7：检查尺度连续性
            // 世界坐标系下，3D点与相机间的向量，方向由相机指向3D点
            cv::Mat normal1 = x3D - Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D - Ow2;
            float dist2 = cv::norm(normal2);

            if (dist1 == 0 || dist2 == 0)
                continue;

            if (mbFarPoints && (dist1 >= mThFarPoints || dist2 >= mThFarPoints)) // MODIFICATION
                continue;

            // ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2 / dist1;
            // 金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];
            // ratioFactor = 1.5*1.2 1.2为配置文件中的尺度系数 ratioDist 与 ratioOctave应该是近似的，如果两个数相差过大会得到下面的结果
            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            // Triangulation is succesfull
            // 步骤6.8：三角化生成3D点成功，构造成MapPoint
            MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());

            pMP->AddObservation(mpCurrentKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }
}

/**
 * @brief 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints，先完成相邻关键帧与当前关键帧的MP的融合，再完成当前关键帧与相邻关键帧的MP的融合
 */
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    // 步骤1：获得当前关键帧在covisibility图中权重排名前nn的邻接关键帧
    // 找到当前帧一级相邻与二级相邻关键帧，也就是找到与当前帧共视的共视帧
    int nn = 10;
    if (mbMonocular)
        nn = 20;
    const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame *> vpTargetKFs;
    for (vector<KeyFrame *>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);                       // 加入一级相邻帧
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId; // 并标记已经加入
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for (int i = 0, imax = vpTargetKFs.size(); i < imax; i++)
    {
        const vector<KeyFrame *> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        for (vector<KeyFrame *>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
        {
            KeyFrame *pKFi2 = *vit2;
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);                       // 存入二级相邻帧
            pKFi2->mnFuseTargetForKF = mpCurrentKeyFrame->mnId; // 并标记已经加入
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    // IMU模式下往前找关联关键帧，保证数值达到20或者找到了第一个关键帧
    if (mbInertial)
    {
        KeyFrame *pKFi = mpCurrentKeyFrame->mPrevKF;
        while (vpTargetKFs.size() < 20 && pKFi)
        {
            if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
            pKFi = pKFi->mPrevKF;
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    // 步骤2：将当前帧的MapPoints分别与一级二级相邻帧(的MapPoints)进行融合
    vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (vector<KeyFrame *>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;

        // 投影当前帧的MapPoints到相邻关键帧pKFi中，并判断是否有重复的MapPoints
        // 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
        // 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
        matcher.Fuse(pKFi, vpMapPointMatches);
        if (pKFi->NLeft != -1)
            matcher.Fuse(pKFi, vpMapPointMatches, true);
    }

    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    // 用于存储一级邻接和二级邻接关键帧所有MapPoints的集合，也就是存放了这些帧中所有有效的MP
    vector<MapPoint *> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());
    // 步骤3：将一级二级相邻帧的MapPoints分别与当前帧（的MapPoints）进行融合
    // 遍历每一个一级邻接和二级邻接关键帧
    for (vector<KeyFrame *>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
    {
        KeyFrame *pKFi = *vitKF;

        vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for (vector<MapPoint *>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
        {
            MapPoint *pMP = *vitMP;
            if (!pMP)
                continue;
            if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);
    if (mpCurrentKeyFrame->NLeft != -1)
        matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates, true);

    // Update points
    // 步骤4：更新当前帧MapPoints的描述子，深度，观测主方向等属性
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    // 步骤5：更新当前帧的MapPoints后更新与其它帧的连接关系
    // 更新covisibility图
    mpCurrentKeyFrame->UpdateConnections();
}

/**
 * @brief 根据两关键帧的姿态计算两个关键帧之间的基本矩阵
 * @param  pKF1 关键帧1
 * @param  pKF2 关键帧2
 * @return      基本矩阵
 */
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mpCamera->toK();
    const cv::Mat &K2 = pKF2->mpCamera->toK();

    return K1.t().inv() * t12x * R12 * K2.inv();
}

/**
 * @brief 切换模式以及回环时会停止局部地图线程，正在做局部BA也会停止
 */
void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

/**
 * @brief 停止局部地图
 */
bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

/**
 * @brief 查看是否已经停止
 */
bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

/**
 * @brief 查看是否有停止信号
 */
bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

/**
 * @brief 继续运行线程
 */
void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if (mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

/**
 * @brief 查看是否接收关键帧，也就是当前线程是否在处理数据，当然tracking线程也不会全看这个值，他会根据队列阻塞情况
 */
bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

/**
 * @brief 每次循环开始设定为false，结束设定为true
 */
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

/**
 * @brief 如果不让它暂停，即使发出了暂停信号也不暂停
 */
bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if (flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

/**
 * @brief 放弃这次BA
 */
void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

/**
 * @brief 关键帧剔除
 * 在Covisibility Graph中的关键帧，其90%以上的MapPoints能被其他关键帧（至少3个）观测到，则认为该关键帧为冗余关键帧。
 */
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21; // MODIFICATION_STEREO_IMU 20 This should be the same than that one from LIBA
    mpCurrentKeyFrame->UpdateBestCovisibles();  // 更新共视关系
    // 1. 根据Covisibility Graph提取当前帧的共视关键帧
    vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    // 非IMU时
    if (!mbInertial)
        redundant_th = 0.9;
    else if (mbMonocular) // imu 且单目时
        redundant_th = 0.9;
    else  // 其他imu时
        redundant_th = 0.5;

    // Compoute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial)
    {
        int count = 0;
        KeyFrame *aux_KF = mpCurrentKeyFrame;
        // 找到第前21个关键帧的关键帧id
        while (count < Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }
    int count = 0;
    const bool bInitImu = mpAtlas->isImuInitialized();
    // 对所有的局部关键帧进行遍历
    for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
    {
        count++;
        KeyFrame *pKF = *vit;
        // 跳过没用的帧及bad的帧
        if ((pKF->mnId == pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        
        // 2. 提取每个共视关键帧的MapPoints
        const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs = nObs;
        int nRedundantObservations = 0;
        int nMPs = 0;
        // 3. 遍历该局部关键帧的MapPoints，判断是否90%以上的MapPoints能被其它关键帧（至少3个）观测到
        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    if (!mbMonocular)
                    {
                        // 对于非单目，仅考虑一定范围内的
                        if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
                            continue;
                    }

                    nMPs++;
                    // MapPoints至少被三个关键帧观测到
                    if (pMP->Observations() > thObs)
                    {
                        // 返回这个MP在共视关键帧的金字塔层数
                        const int &scaleLevel = (pKF->NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                    : (i < pKF->NLeft) ? pKF->mvKeys[i].octave
                                                                                        : pKF->mvKeysRight[i].octave;
                        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                        // 统计被观测次数，且要符合尺度要求
                        int nObs = 0;
                        for (map<KeyFrame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                        {
                            KeyFrame *pKFi = mit->first;
                            if (pKFi == pKF)
                                continue;
                            tuple<int, int> indexes = mit->second;
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = -1;
                            if (pKFi->NLeft == -1)
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                            else
                            {
                                if (leftIndex != -1)
                                {
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                                }
                                if (rightIndex != -1)
                                {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                    : scaleLeveli;
                                }
                            }
                            // 其他看到这个MP的关键帧对应的层数如果小于等于共视关键帧MP所在的层数，算上一个
                            // 个人理解作者本意想保留层数较低的关键帧，这样点如果多的话就要删除共视关键帧了，这样以来MP少了一个层数较高的关键帧
                            if (scaleLeveli <= scaleLevel + 1)
                            {
                                nObs++;
                                if (nObs > thObs)
                                    break;
                            }
                        }
                        if (nObs > thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }
        // 这样的MP数量占总数一定比例后就会干掉这个关键帧
        if (nRedundantObservations > redundant_th * nMPs)
        {
            // imu模式下需要更改前后关键帧的连续性，且预积分要叠加起来
            if (mbInertial)
            {
                // 关键帧少于Nd个，跳过不删
                if (mpAtlas->KeyFramesInMap() <= Nd)
                    continue;
                // 关键帧与当前关键帧id差一个，跳过不删
                if (pKF->mnId > (mpCurrentKeyFrame->mnId - 2))
                    continue;
                // 关键帧具有前后关键帧
                if (pKF->mPrevKF && pKF->mNextKF)
                {
                    const float t = pKF->mNextKF->mTimeStamp - pKF->mPrevKF->mTimeStamp;
                    // 下面两个括号里的内容一模一样
                    // imu初始化了，且距当前帧的ID超过21，且前后两个关键帧时间间隔小于3s
                    // 或者时间间隔小于0.5s
                    if ((bInitImu && (pKF->mnId < last_ID) && t < 3.) || (t < 0.5))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                    // 没经过imu初始化的第三阶段，且关键帧与其前一个关键帧的距离小于0.02m，且前后两个关键帧时间间隔小于3s
                    else if (!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && (cv::norm(pKF->GetImuPosition() - pKF->mPrevKF->GetImuPosition()) < 0.02) && (t < 3))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                }
            }
            // 非imu就没那么多逼事儿了，直接干掉
            else
            {
                pKF->SetBadFlag();
            }
        }
        // 遍历共视关键帧个数超过一定，就不弄了
        if ((count > 20 && mbAbortBA) || count > 100) // MODIFICATION originally 20 for mbabortBA check just 10 keyframes
        {
            break;
        }
    }
}

/**
 * @brief 返回反对称矩阵
 * @param v 三维向量
 * @return v的反对称矩阵
 */
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
                                     v.at<float>(2), 0, -v.at<float>(0),
                                    -v.at<float>(1), v.at<float>(0), 0);
}

/**
 * @brief 接收重置信号
 */
void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    cout << "LM: Map reset, waiting..." << endl;

    while (1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

/**
 * @brief 接收重置当前地图的信号
 */
void LocalMapping::RequestResetActiveMap(Map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
    cout << "LM: Active map reset, waiting..." << endl;

    while (1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Active map reset, Done!!!" << endl;
}

/**
 * @brief 重置
 */
void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbResetRequested)
        {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu = false;

            mIdxInit = 0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if (mbResetRequestedActiveMap)
        {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu = false;

            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if (executed_reset)
        cout << "LM: Reset free the mutex" << endl;
}

/**
 * @brief 接收完成信号
 */
void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

/**
 * @brief 查看完成信号，跳出while循环
 */
bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

/**
 * @brief 置为完成
 */
void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

/**
 * @brief 查看是否完成
 */
bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

/** 
 * @brief imu初始化
 * @param priorG 陀螺仪偏置的信息矩阵系数，主动设置时一般bInit为true，也就是只优化最后一帧的偏置，这个数会作为计算信息矩阵时使用
 * @param priorA 加速度计偏置的信息矩阵系数
 * @param bFIBA 是否做BA优化
 */
void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    // 1. 将所有关键帧放入列表及向量里，且查看是否满足初始化条件
    if (mbResetRequested)
        return;

    float minTime;
    int nMinKF;
    // 从时间及帧数上限制初始化，不满足下面的不进行初始化
    if (mbMonocular)
    {
        minTime = 2.0;
        nMinKF = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF = 10;
    }

    // 当前地图大于10帧才进行初始化
    if (mpAtlas->KeyFramesInMap() < nMinKF)
        return;

    // Retrieve all keyframe in temporal order
    // 按照顺序存放目前地图里的关键帧，顺序按照前后顺序来，包括当前关键帧
    list<KeyFrame *> lpKF;
    KeyFrame *pKF = mpCurrentKeyFrame;
    while (pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    // 以相同内容再构建一个vector
    vector<KeyFrame *> vpKF(lpKF.begin(), lpKF.end());

    // TODO 跟上面重复？
    if (vpKF.size() < nMinKF)
        return;

    mFirstTs = vpKF.front()->mTimeStamp;
    if (mpCurrentKeyFrame->mTimeStamp - mFirstTs < minTime)
        return;

    bInitializing = true;

    // 先处理新关键帧，防止堆积且保证数据量充足
    while (CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }
    // 2. 正式IMU初始化
    const int N = vpKF.size();
    IMU::Bias b(0, 0, 0, 0, 0, 0);

    // Compute and KF velocities mRwg estimation
    // 在地图没有初始化情况下
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        cv::Mat cvRwg;
        cv::Mat dirG = cv::Mat::zeros(3, 1, CV_32F);
        for (vector<KeyFrame *>::iterator itKF = vpKF.begin(); itKF != vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;
            // Rwb（imu坐标转到初始化前世界坐标系下的坐标）*更新偏置后的速度，可以理解为在世界坐标系下的速度矢量
            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            // 求取实际的速度，位移/时间
            cv::Mat _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition()) / (*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }
        // 归一化
        dirG = dirG / cv::norm(dirG);
        // 原本的重力方向
        cv::Mat gI = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, -1.0f);
        // 求速度方向与重力方向的角轴
        cv::Mat v = gI.cross(dirG);
        // 求角轴长度
        const float nv = cv::norm(v);
        // 求转角大小
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        // 先计算旋转向量，在除去角轴大小
        cv::Mat vzg = v * ang / nv;
        // 获得重力方向到当前速度方向的旋转向量
        cvRwg = IMU::ExpSO3(vzg);
        mRwg = Converter::toMatrix3d(cvRwg);
        mTinit = mpCurrentKeyFrame->mTimeStamp - mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = Converter::toVector3d(mpCurrentKeyFrame->GetGyroBias());
        mba = Converter::toVector3d(mpCurrentKeyFrame->GetAccBias());
    }

    mScale = 1.0;

    // 暂时没发现在别的地方出现过
    mInitTime = mpTracker->mLastFrame.mTimeStamp - vpKF.front()->mTimeStamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    // 计算残差及偏置差，优化尺度重力方向及Ri Rj Vi Vj Pi Pj
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba,
                                    mbMonocular, infoInertial, false, false, priorG, priorA);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    /*cout << "scale after inertial-only optimization: " << mScale << endl;
cout << "bg after inertial-only optimization: " << mbg << endl;
cout << "ba after inertial-only optimization: " << mba << endl;*/

    // 尺度太小的话初始化认为失败
    if (mScale < 1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing = false;
        return;
    }

    // 到此时为止，前面做的东西没有改变map
    // Before this line we are not changing the map

    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // 尺度变化超过设定值，或者非单目时（无论带不带imu，但这个函数只在带imu时才执行，所以这个可以理解为双目imu）
    if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular)
    {
        // 恢复重力方向与尺度信息
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Converter::toCvMat(mRwg).t(), mScale, true);
        // 更新普通帧的位姿，主要是当前帧与上一帧
        mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    // Check if initialization OK
    // 即使初始化成功后面还会执行这个函数重新初始化
    // 在之前没有初始化成功情况下（此时刚刚初始化成功）对每一帧都标记，后面的kf全部都在tracking里面标记为true
    // 也就是初始化之前的那些关键帧即使有imu信息也不算
    if (!mpAtlas->isImuInitialized())
        for (int i = 0; i < N; i++)
        {
            KeyFrame *pKF2 = vpKF[i];
            pKF2->bImu = true;
        }

    /*cout << "Before GIBA: " << endl;
cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;*/

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    // 代码里都为tue
    if (bFIBA)
    {
        // 承接上一步纯imu优化，按照之前的结果更新了尺度信息及适应重力方向，所以要结合地图进行一次视觉加imu的全局优化，这次带了MP等信息
        if (priorA != 0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, 0, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, 0, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    // If initialization is OK
    mpTracker->UpdateFrameIMU(1.0, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
    if (!mpAtlas->isImuInitialized())
    {
        cout << "IMU in Map " << mpAtlas->GetCurrentMap()->GetId() << " is initialized" << endl;
        // 标记初始化成功
        mpAtlas->SetImuInitialized();
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
        mpCurrentKeyFrame->bImu = true;
    }

    mbNewInit = true;
    mnKFs = vpKF.size();
    mIdxInit++;

    for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    mpTracker->mState = Tracking::OK;
    bInitializing = false;

    /*cout << "After GIBA: " << endl;
cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;
double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
double t_update = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
double t_viba = std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count();
cout << t_inertial_only << ", " << t_update << ", " << t_viba << endl;*/

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

/**
 * @brief 通过BA优化进行尺度更新，关键帧小于100，在这里的时间段内时多次进行尺度更新
 */
void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    // 1. 检索所有的关键帧（当前地图）
    list<KeyFrame *> lpKF;
    KeyFrame *pKF = mpCurrentKeyFrame;
    while (pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame *> vpKF(lpKF.begin(), lpKF.end());
    // 加入新添加的帧
    while (CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // const int N = vpKF.size();
    // 2. 更新旋转与尺度
    mRwg = Eigen::Matrix3d::Identity();
    mScale = 1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    // 优化重力方向与尺度
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale < 1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing = false;
        return;
    }

    // Before this line we are not changing the map
    // 3. 开始更新地图
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // 3.1 如果尺度更新较多，或是在双目imu情况下更新地图
    if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular)
    {
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Converter::toCvMat(mRwg).t(), mScale, true);
        mpTracker->UpdateFrameIMU(mScale, mpCurrentKeyFrame->GetImuBias(), mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    // 3.2 优化的这段时间新进来的kf全部清空不要
    for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

/**
 * @brief 返回是否正在做IMU的初始化，在tracking里面使用，如果为true，暂不添加关键帧
 */
bool LocalMapping::IsInitializing()
{
    return bInitializing;
}

/**
 * @brief 获取当前关键帧的时间戳，System::GetTimeFromIMUInit()中调用
 */
double LocalMapping::GetCurrKFTime()
{
    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

/**
 * @brief 获取当前关键帧
 */
KeyFrame *LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

} // namespace ORB_SLAM3
