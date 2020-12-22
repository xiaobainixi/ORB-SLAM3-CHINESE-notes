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


#include "LoopClosing.h"

#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "G2oTypes.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM3
{

LoopClosing::LoopClosing(Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), 
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true), 
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0), mnLoopNumCoincidences(0), mnMergeNumCoincidences(0), 
    mbLoopDetected(false), mbMergeDetected(false), mnLoopNumNotFound(0), mnMergeNumNotFound(0)
{
    mnCovisibilityConsistencyTh = 3;
    mpLastCurrentKF = static_cast<KeyFrame*>(NULL);
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker = pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

/** 
 * @brief 执行函数
 */
void LoopClosing::Run()
{
    mbFinished = false;

    while(1)
    {
        //NEW LOOP AND MERGE DETECTION ALGORITHM
        //----------------------------
        // 一直循环一直读
        if(CheckNewKeyFrames())
        {
            if(mpLastCurrentKF)
            {
                mpLastCurrentKF->mvpLoopCandKFs.clear();  // 该文件中只在这里提到过一次，还不清楚有什么用
                mpLastCurrentKF->mvpMergeCandKFs.clear();  // 该文件中只在这里提到过一次，还不清楚有什么用
            }
            if(NewDetectCommonRegions())
            {
                // 分两类，NewDetectCommonRegions的结果为地图融合时，表示当前关键帧与其他地图有关联
                // 1. 地图融合时
                if(mbMergeDetected)
                {
                    // 1.1 如果设备有imu但地图没有经过imu初始化时，跳过融合
                    if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO) &&
                        (!mpCurrentKF->GetMap()->isImuInitialized()))
                    {
                        Verbose::PrintMess("IMU is not initilized, merge is aborted", Verbose::VERBOSITY_QUIET);
                    }
                    else
                    {
                        Verbose::PrintMess("*Merged detected", Verbose::VERBOSITY_QUIET);
                        Verbose::PrintMess("Number of KFs in the current map: " + to_string(mpCurrentKF->GetMap()->KeyFramesInMap()), Verbose::VERBOSITY_DEBUG);

                        // 匹配上的融合关键帧在它的地图里面的世界坐标到其的位姿
                        cv::Mat mTmw = mpMergeMatchedKF->GetPose();
                        g2o::Sim3 gSmw2(Converter::toMatrix3d(mTmw.rowRange(0, 3).colRange(0, 3)), Converter::toVector3d(mTmw.rowRange(0, 3).col(3)), 1.0);

                        // 当前地图的世界坐标到当前关键帧
                        cv::Mat mTcw = mpCurrentKF->GetPose();
                        g2o::Sim3 gScw1(Converter::toMatrix3d(mTcw.rowRange(0, 3).colRange(0, 3)), Converter::toVector3d(mTcw.rowRange(0, 3).col(3)), 1.0);
                        
                        // mg2oMergeSlw里面存放的是另一个地图的世界坐标到当前关键帧的sim3
                        g2o::Sim3 gSw2c = mg2oMergeSlw.inverse();
                        // g2o::Sim3 gSw1m = mg2oMergeSlw;

                        // 结果为gSw2w1 也就是当前地图的世界坐标到融合地图的世界坐标的位姿
                        mSold_new = (gSw2c * gScw1);

                        // 1.2 如果两个地图都在IMU模式下
                        if(mpCurrentKF->GetMap()->IsInertial() && mpMergeMatchedKF->GetMap()->IsInertial())
                        {
                            // 1.2.1 尺度差的过多则跳过，凡是有异必为妖
                            if(mSold_new.scale()<0.90 || mSold_new.scale()>1.1)
                            {
                                mpMergeLastCurrentKF->SetErase();
                                mpMergeMatchedKF->SetErase();
                                mnMergeNumCoincidences = 0;
                                mvpMergeMatchedMPs.clear();
                                mvpMergeMPs.clear();
                                mnMergeNumNotFound = 0;
                                mbMergeDetected = false;
                                Verbose::PrintMess("scale bad estimated. Abort merging", Verbose::VERBOSITY_NORMAL);
                                continue;
                            }
                            // If inertial, force only yaw
                            // 1.2.2 强制让两个轴为0 直白的说可以理解成两个坐标系都经过了imu初始化，肯定都是水平的，所以不考虑
                            if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO) &&
                                   mpCurrentKF->GetMap()->GetIniertialBA1()) // TODO, maybe with GetIniertialBA1
                            {
                                Eigen::Vector3d phi = LogSO3(mSold_new.rotation().toRotationMatrix());
                                phi(0) = 0;
                                phi(1) = 0;
                                mSold_new = g2o::Sim3(ExpSO3(phi), mSold_new.translation(), 1.0);
                            }
                        }


//                        cout << "tw2w1: " << mSold_new.translation() << endl;
//                        cout << "Rw2w1: " << mSold_new.rotation().toRotationMatrix() << endl;
//                        cout << "Angle Rw2w1: " << 180*LogSO3(mSold_new.rotation().toRotationMatrix())/3.14 << endl;
//                        cout << "scale w2w1: " << mSold_new.scale() << endl;

                        // mg2oMergeSmw = gSmw2 * gSw2c * gScw1;  // 没用

                        mg2oMergeScw = mg2oMergeSlw;

                        // TODO UNCOMMENT
                        // 1.3 当前地图下如果是IMU模式下用MergeLocal2，否则用MergeLocal
                        if (mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO)
                            MergeLocal2();
                        else
                            MergeLocal();  // 前面地图有可能imu的，当前地图可能没有imu
                    }

                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                    vdPR_MatchedTime.push_back(mpMergeMatchedKF->mTimeStamp);
                    vnPR_TypeRecogn.push_back(1);

                    // Reset all variables
                    mpMergeLastCurrentKF->SetErase();
                    mpMergeMatchedKF->SetErase();
                    mnMergeNumCoincidences = 0;
                    mvpMergeMatchedMPs.clear();
                    mvpMergeMPs.clear();
                    mnMergeNumNotFound = 0;
                    mbMergeDetected = false;

                    if(mbLoopDetected)
                    {
                        // Reset Loop variables
                        mpLoopLastCurrentKF->SetErase();
                        mpLoopMatchedKF->SetErase();
                        mnLoopNumCoincidences = 0;
                        mvpLoopMatchedMPs.clear();
                        mvpLoopMPs.clear();
                        mnLoopNumNotFound = 0;
                        mbLoopDetected = false;
                    }

                }

                // 
                if(mbLoopDetected)
                {
                    // 这三行都是调试用的，统计下回环时间
                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                    vdPR_MatchedTime.push_back(mpLoopMatchedKF->mTimeStamp);
                    vnPR_TypeRecogn.push_back(0);


                    Verbose::PrintMess("*Loop detected", Verbose::VERBOSITY_QUIET);
                    // 走到这里时经过NewDetectCommonRegions函数，mg2oLoopSlw已经从上一个变成了当前的了，所以直接赋值就可以
                    mg2oLoopScw = mg2oLoopSlw; //*mvg2oSim3LoopTcw[nCurrentIndex];
                    if(mpCurrentKF->GetMap()->IsInertial())
                    {
                        // 这个是正常跟踪到当前帧的位姿，而mg2oLoopScw是通过回环帧位姿乘上回环帧与当前帧的位姿得到
                        cv::Mat Twc = mpCurrentKF->GetPoseInverse();
                        g2o::Sim3 g2oTwc(Converter::toMatrix3d(Twc.rowRange(0, 3).colRange(0, 3)), Converter::toVector3d(Twc.rowRange(0, 3).col(3)), 1.0);
                        g2o::Sim3 g2oSww_new = g2oTwc*mg2oLoopScw;

                        Eigen::Vector3d phi = LogSO3(g2oSww_new.rotation().toRotationMatrix());
                        //cout << "tw2w1: " << g2oSww_new.translation() << endl;
                        //cout << "Rw2w1: " << g2oSww_new.rotation().toRotationMatrix() << endl;
                        //cout << "Angle Rw2w1: " << 180*phi/3.14 << endl;
                        //cout << "scale w2w1: " << g2oSww_new.scale() << endl;

                        if (fabs(phi(0))<0.008f && fabs(phi(1))<0.008f && fabs(phi(2))<0.349f)
                        {
                            if(mpCurrentKF->GetMap()->IsInertial())
                            {
                                // If inertial, force only yaw
                                if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO) &&
                                        mpCurrentKF->GetMap()->GetIniertialBA2()) // TODO, maybe with GetIniertialBA1
                                {
                                    phi(0)=0;
                                    phi(1)=0;
                                    g2oSww_new = g2o::Sim3(ExpSO3(phi), g2oSww_new.translation(), 1.0);
                                    mg2oLoopScw = g2oTwc.inverse()*g2oSww_new;
                                }
                            }

                            mvpLoopMapPoints = mvpLoopMPs;//*mvvpLoopMapPoints[nCurrentIndex];
                            CorrectLoop();
                        }
                        else
                        {
                            cout << "BAD LOOP!!!" << endl;
                        }
                    }
                    else
                    {
                        mvpLoopMapPoints = mvpLoopMPs;
                        CorrectLoop();
                    }

                    // Reset all variables
                    mpLoopLastCurrentKF->SetErase();
                    mpLoopMatchedKF->SetErase();
                    mnLoopNumCoincidences = 0;
                    mvpLoopMatchedMPs.clear();
                    mvpLoopMPs.clear();
                    mnLoopNumNotFound = 0;
                    mbLoopDetected = false;
                }
            }
            mpLastCurrentKF = mpCurrentKF;
        }

        ResetIfRequested();

        if(CheckFinish())
        {
            // cout << "LC: Finish requested" << endl;
            break;
        }

        usleep(5000);
    }

    //ofstream f_stats;
    //f_stats.open("PlaceRecognition_stats" + mpLocalMapper->strSequence + ".txt");
    //f_stats << "# current_timestamp, matched_timestamp, [0:Loop, 1:Merge]" << endl;
    //f_stats << fixed;
    //for(int i=0; i< vdPR_CurrentTime.size(); ++i)
    //{
    //    f_stats  << 1e9*vdPR_CurrentTime[i] << ", " << 1e9*vdPR_MatchedTime[i] << ", " << vnPR_TypeRecogn[i] << endl;
    //}

    //f_stats.close();

    SetFinish();
}

/** 
 * @brief 向mlpLoopKeyFrameQueue 插入新的关键帧
 */
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

/** 
 * @brief 查看 mlpLoopKeyFrameQueue 队列里面是否有未处理的新关键帧
 */
bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

/** 
 * @brief orbslam2中这个函数的名字是DetectLoop，检测共同区域，如果当前帧检测到回环或者融合，后面关键帧也检测到，这个函数会一直返回true
 */
bool LoopClosing::NewDetectCommonRegions()
{
    // 1. 先根据新关键帧做数据更新
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        // 防止做着做着关键帧被干掉。。。
        mpCurrentKF->SetNotErase();
        mpCurrentKF->mbCurrentPlaceRecognition = true;  // 暂时还不知道有啥用，目前来看是没用，后面被注释掉了

        mpLastMap = mpCurrentKF->GetMap();  // 当前关键帧对应的地图，
    }
    // 2. 经过3个判定是否做回环
    // imu 模式下还没经过第二阶段初始化则不考虑回环或融合
    if(mpLastMap->IsInertial() && !mpLastMap->GetIniertialBA1())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }
    // 纯双目模式下地图的关键帧很少
    if(mpTracker->mSensor == System::STEREO && mpLastMap->GetAllKeyFrames().size() < 5) //12
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }
    // 地图的关键帧很少
    if(mpLastMap->GetAllKeyFrames().size() < 12)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    //Check the last candidates with geometric validation
    // Loop candidates
    bool bLoopDetectedInKF = false;  // 检测到回环
    // bool bCheckSpatial = false;  // 没用

    // 3. 如果上一帧回环检测成功
    // mnLoopNumCoincidences > 0 表明上一关键帧回环成功
    if(mnLoopNumCoincidences > 0)
    {
        // bCheckSpatial = true;
        // Find from the last KF candidates
        // 3.1 上一个连续回环关键帧到当前帧的位姿变换,同时在结合上次回环的结果得到相似变换gScw
        cv::Mat mTcl = mpCurrentKF->GetPose() * mpLoopLastCurrentKF->GetPoseInverse();
        g2o::Sim3 gScl(Converter::toMatrix3d(mTcl.rowRange(0, 3).colRange(0, 3)), Converter::toVector3d(mTcl.rowRange(0, 3).col(3)), 1.0);
        g2o::Sim3 gScw = gScl * mg2oLoopSlw;

        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        // 3.2 通过非线性优化更新gScw
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpLoopMatchedKF, gScw, numProjMatches, mvpLoopMPs, vpMatchedMPs);
        if(bCommonRegion)
        {

            bLoopDetectedInKF = true;

            mnLoopNumCoincidences++;
            mpLoopLastCurrentKF->SetErase();
            mpLoopLastCurrentKF = mpCurrentKF;
            mg2oLoopSlw = gScw;
            mvpLoopMatchedMPs = vpMatchedMPs;


            mbLoopDetected = mnLoopNumCoincidences >= 3;
            mnLoopNumNotFound = 0;

            if(!mbLoopDetected)
            {
                //f_succes_pr << mpCurrentKF->mNameFile << " " << "8"<< endl;
                //f_succes_pr << "% Number of spatial consensous: " << std::to_string(mnLoopNumCoincidences) << endl;
                cout << "PR: Loop detected with Reffine Sim3" << endl;
            }
        }
        else
        {
            bLoopDetectedInKF = false;
            /*f_succes_pr << mpCurrentKF->mNameFile << " " << "8"<< endl;
            f_succes_pr << "% Number of spatial consensous: " << std::to_string(mnLoopNumCoincidences) << endl;*/

            mnLoopNumNotFound++;
            if(mnLoopNumNotFound >= 2)
            {
                /*for(int i=0; i<mvpLoopLastKF.size(); ++i)
                {
                    mvpLoopLastKF[i]->SetErase();
                    mvpLoopCandidateKF[i]->SetErase();
                    mvpLoopLastKF[i]->mbCurrentPlaceRecognition = true;
                }

                mvpLoopCandidateKF.clear();
                mvpLoopLastKF.clear();
                mvg2oSim3LoopTcw.clear();
                mvnLoopNumMatches.clear();
                mvvpLoopMapPoints.clear();
                mvvpLoopMatchedMapPoints.clear();*/

                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                //mg2oLoopScw;
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
            }

        }
    }

    //Merge candidates
    bool bMergeDetectedInKF = false;
    if(mnMergeNumCoincidences > 0)
    {
        // Find from the last KF candidates

        cv::Mat mTcl = mpCurrentKF->GetPose() * mpMergeLastCurrentKF->GetPoseInverse();
        g2o::Sim3 gScl(Converter::toMatrix3d(mTcl.rowRange(0, 3).colRange(0, 3)), Converter::toVector3d(mTcl.rowRange(0, 3).col(3)), 1.0);
        g2o::Sim3 gScw = gScl * mg2oMergeSlw;
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpMergeMatchedKF, gScw, numProjMatches, mvpMergeMPs, vpMatchedMPs);
        if(bCommonRegion)
        {
            //cout << "BoW: Merge reffined Sim3 transformation suscelful" << endl;
            bMergeDetectedInKF = true;

            mnMergeNumCoincidences++;
            mpMergeLastCurrentKF->SetErase();
            mpMergeLastCurrentKF = mpCurrentKF;
            mg2oMergeSlw = gScw;
            mvpMergeMatchedMPs = vpMatchedMPs;

            mbMergeDetected = mnMergeNumCoincidences >= 3;
        }
        else
        {
            //cout << "BoW: Merge reffined Sim3 transformation failed" << endl;
            mbMergeDetected = false;
            bMergeDetectedInKF = false;

            mnMergeNumNotFound++;
            if(mnMergeNumNotFound >= 2)
            {
                /*cout << "+++++++Merge detected failed in KF" << endl;

                for(int i=0; i<mvpMergeLastKF.size(); ++i)
                {
                    mvpMergeLastKF[i]->SetErase();
                    mvpMergeCandidateKF[i]->SetErase();
                    mvpMergeLastKF[i]->mbCurrentPlaceRecognition = true;
                }

                mvpMergeCandidateKF.clear();
                mvpMergeLastKF.clear();
                mvg2oSim3MergeTcw.clear();
                mvnMergeNumMatches.clear();
                mvvpMergeMapPoints.clear();
                mvvpMergeMatchedMapPoints.clear();*/

                mpMergeLastCurrentKF->SetErase();
                mpMergeMatchedKF->SetErase();
                mnMergeNumCoincidences = 0;
                mvpMergeMatchedMPs.clear();
                mvpMergeMPs.clear();
                mnMergeNumNotFound = 0;
            }


        }
    }

    if(mbMergeDetected || mbLoopDetected)
    {
        //f_time_pr << "Geo" << " " << timeGeoKF_ms.count() << endl;
        mpKeyFrameDB->add(mpCurrentKF);
        return true;
    }

    //-------------
    //TODO: This is only necessary if we use a minimun score for pick the best candidates
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    /*float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }*/
    //-------------

    // Extract candidates from the bag of words
    vector<KeyFrame*> vpMergeBowCand, vpLoopBowCand;
    //cout << "LC: bMergeDetectedInKF: " << bMergeDetectedInKF << "   bLoopDetectedInKF: " << bLoopDetectedInKF << endl;
    if(!bMergeDetectedInKF || !bLoopDetectedInKF)
    {
        // Search in BoW
        // 查找最佳的候选关键帧
        mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF, vpLoopBowCand, vpMergeBowCand, 3);
    }

    // Check the BoW candidates if the geometric candidate list is empty
    //Loop candidates
/*#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point timeStartGeoBoW = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point timeStartGeoBoW = std::chrono::monotonic_clock::now();
#endif*/

    if(!bLoopDetectedInKF && !vpLoopBowCand.empty())
    {
        // 第一次执行这个函数的时候除了 vpLoopBowCand 不为空，剩下的不是为0就是为空
        mbLoopDetected = DetectCommonRegionsFromBoW(vpLoopBowCand, mpLoopMatchedKF, mpLoopLastCurrentKF, mg2oLoopSlw, mnLoopNumCoincidences, mvpLoopMPs, mvpLoopMatchedMPs);
    }
    // Merge candidates

    //cout << "LC: Find BoW candidates" << endl;

    if(!bMergeDetectedInKF && !vpMergeBowCand.empty())
    {
        mbMergeDetected = DetectCommonRegionsFromBoW(vpMergeBowCand, mpMergeMatchedKF, mpMergeLastCurrentKF, mg2oMergeSlw, mnMergeNumCoincidences, mvpMergeMPs, mvpMergeMatchedMPs);
    }

    //cout << "LC: add to KFDB" << endl;
    mpKeyFrameDB->add(mpCurrentKF);

    if(mbMergeDetected || mbLoopDetected)
    {
        return true;
    }

    //cout << "LC: erase KF" << endl;
    mpCurrentKF->SetErase();
    mpCurrentKF->mbCurrentPlaceRecognition = false;

    return false;
}

/** 检测和更新sim3
 * @param pCurrentKF 当前关键帧
 * @param pMatchedKF 上一个回环选得的候选关键帧
 * @param gScw 世界坐标系到当前帧的相似变换（通过计算而来）
 * @param nNumProjMatches 相似变换矩阵
 * @param vpMPs 上一个回环选得的候选关键帧及其共视帧组成的所有帧的mp，在这里会被清空重新赋值，赋值成候选关键帧及其共视帧及其共视帧的共视帧的所有mp
 * @param vpMatchedMPs 清空重新赋值，匹配的点
 */
bool LoopClosing::DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, 
                                                 std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs;
    // 1. 重新基于
    // TODO 清空vpMPs是不是有些多余？经过验证并不多余，点数有一定概率有轻微变化，但不大，这里可以做优化
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
    cout << "REFFINE-SIM3: Projection from last KF with " << nNumProjMatches << " matches" << endl;


    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    /*if(mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO)
    {
        nProjMatches = 50;
        nProjOptMatches = 50;
        nProjMatchesRep = 80;
    }*/
    // 2.点数如果不符合返回false
    if(nNumProjMatches >= nProjMatches)
    {
        // 3.1 求得gScm
        cv::Mat mScw = Converter::toCvMat(gScw);
        cv::Mat mTwm = pMatchedKF->GetPoseInverse();
        g2o::Sim3 gSwm(Converter::toMatrix3d(mTwm.rowRange(0, 3).colRange(0, 3)), Converter::toVector3d(mTwm.rowRange(0, 3).col(3)), 1.0);
        g2o::Sim3 gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        bool bFixedScale = mbFixScale;       // TODO CHECK; Solo para el monocular inertial
        if(mpTracker->mSensor==System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2())
            bFixedScale=false;
        // 3.2 优化gScm，mp固定
        int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);
        cout << "REFFINE-SIM3: Optimize Sim3 from last KF with " << numOptMatches << " inliers" << endl;



        if(numOptMatches > nProjOptMatches)
        {
            // BUG 不优化gScw？
            g2o::Sim3 gScw_estimation(Converter::toMatrix3d(mScw.rowRange(0, 3).colRange(0, 3)), 
                           Converter::toVector3d(mScw.rowRange(0, 3).col(3)), 1.0);

            vector<MapPoint*> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));

            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            //cout << "REFFINE-SIM3: Projection with optimize Sim3 from last KF with " << nNumProjMatches << " matches" << endl;
            if(nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    return false;
}

/** 通过BoW查找有共同区域的候选关键帧，选择vpBowCand中最合适的帧及其对应值
 * @param vpBowCand 候选关键帧
 * @param pMatchedKF2 候选关键帧中与当前帧匹配的关键帧
 * @param pLastCurrentKF mpCurrentKF
 * @param g2oScw 相似变换矩阵
 * @param nNumCoincidences 当前帧及共视帧中与候选关键帧匹配的个数
 * @param vpMPs 匹配的候选关键帧及共视帧这些加一起对应的mp
 * @param vpMatchedMPs 当前帧中特征点对应的MP
 */
bool LoopClosing::DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF2, KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw, 
                                             int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    int nBoWMatches = 20;  // 当前帧与候选关键帧及其共视帧匹配的最低点数
    int nBoWInliers = 15;
    int nSim3Inliers = 20;
    int nProjMatches = 50;
    int nProjOptMatches = 80;
    /*if(mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO)
    {
        nBoWMatches = 20;
        nBoWInliers = 15;
        nSim3Inliers = 20;
        nProjMatches = 35;
        nProjOptMatches = 50;
    }*/

    // 1. 获得与当前关键帧相连的关键帧
    set<KeyFrame*> spConnectedKeyFrames = mpCurrentKF->GetConnectedKeyFrames();

    

    ORBmatcher matcherBoW(0.9, true);
    ORBmatcher matcher(0.75, true);
    int nNumGuidedMatching = 0;

    // Varibles to select the best numbe
    KeyFrame* pBestMatchedKF;
    int nBestMatchesReproj = 0;
    int nBestNumCoindicendes = 0;
    g2o::Sim3 g2oBestScw;
    std::vector<MapPoint*> vpBestMapPoints;
    std::vector<MapPoint*> vpBestMatchedMapPoints;

    int numCandidates = vpBowCand.size();
    vector<int> vnStage(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);

    int index = 0;
    int nNumCovisibles = 5;

    // 2. 对每个候选关键帧都进行详细的分析
    for(KeyFrame* pKFi : vpBowCand)
    {
        //cout << endl << "-------------------------------" << endl;
        if(!pKFi || pKFi->isBad())
            continue;

        // Current KF against KF with covisibles version
        // 2.1 vpCovKFi存放了pKFi的共视关键帧及pKFi自己
        std::vector<KeyFrame*> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);
        vpCovKFi.push_back(vpCovKFi[0]);
        vpCovKFi[0] = pKFi;

        int nIndexMostBoWMatchesKF = 0;  // 最大匹配点数对应帧的位置
        int nMostBoWNumMatches = 0;  // 最大匹配点数，后面没有用到
        std::vector<std::vector<MapPoint*> > vvpMatchedMPs;  // 存放当前关键帧与vpCovKFi中每个关键帧的匹配的MP
        vvpMatchedMPs.resize(vpCovKFi.size());

        // 2.2 寻找vpCovKFi中每一帧与当前关键帧的匹配关系
        for(int j=0; j<vpCovKFi.size(); ++j)
        {
            if(!vpCovKFi[j] || vpCovKFi[j]->isBad())
                continue;

            int num = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]);
            //cout << "BoW: " << num << " putative matches with KF " << vpCovKFi[j]->mnId << endl;
            if (num > nMostBoWNumMatches)
            {
                nMostBoWNumMatches = num;
                nIndexMostBoWMatchesKF = j;
            }
        }

        bool bAbortByNearKF = false;
        std::set<MapPoint*> spMatchedMPi;
        int numBoWMatches = 0;  // 与其他帧匹配的点数，这里不是指某一帧，是mpCurrentKF与vpCovKFi里面所有帧的匹配结果
        // 记录了mpCurrentKF中的MP与vpCovKFi这些帧的匹配MP，个别位置有可能为空
        std::vector<MapPoint*> vpMatchedPoints = std::vector<MapPoint*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
        // vpMatchedPoints每一匹配点对应的关键帧，个别位置有可能为空
        std::vector<KeyFrame*> vpKeyFrameMatchedMP = std::vector<KeyFrame*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));

        // 2.3 找到mpCurrentKF中每个MP跟vpCovKFi中每帧匹配的MP
        for(int j=0; j<vpCovKFi.size(); ++j)
        {
            // 防止vpCovKFi中的帧为mpCurrentKF共视帧（相邻），相邻帧不需要做回环
            if(spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end())
            {
                bAbortByNearKF = true;
                //cout << "BoW: Candidate KF aborted by proximity" << endl;
                break;
            }

            //cout << "Matches: " << num << endl;
            for(int k=0; k < vvpMatchedMPs[j].size(); ++k)
            {
                MapPoint* pMPi_j = vvpMatchedMPs[j][k];
                if(!pMPi_j || pMPi_j->isBad())
                    continue;
                // 每个点只占一次，因为有可能出现重复的MP
                if(spMatchedMPi.find(pMPi_j) == spMatchedMPi.end())
                {
                    spMatchedMPi.insert(pMPi_j);
                    numBoWMatches++;

                    vpMatchedPoints[k]= pMPi_j;
                    vpKeyFrameMatchedMP[k] = vpCovKFi[j];
                }
            }
        }

        //cout <<"BoW: " << numBoWMatches << " independent putative matches" << endl;
        // TODO 1. 前面统计了vpCovKFi中每个帧与当前帧匹配点的位置，可否用点数高的代替
        //      2. 有可能作者认为在DetectNBestCandidates已经找到共视关键帧中分数最多的了，所以这里不做判断直接使用原始的候选关键帧
        KeyFrame* pMostBoWMatchesKF = pKFi;

        // 2.4 匹配点数足够多的话准备计算sim3
        if(!bAbortByNearKF && numBoWMatches >= nBoWMatches) // TODO pick a good threshold
        {
            /*cout << "-------------------------------" << endl;
            cout << "Geometric validation with " << numBoWMatches << endl;
            cout << "KFc: " << mpCurrentKF->mnId << "; KFm: " << pMostBoWMatchesKF->mnId << endl;*/
            // Geometric validation
            // 除了纯单目剩下全部为true，另外单目imu时如果初始化第三阶段没完毕，也需要修正尺度
            bool bFixedScale = mbFixScale;
            if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                bFixedScale = false;

            Sim3Solver solver = Sim3Solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.SetRansacParameters(0.99, nBoWInliers, 300); // at least 15 inliers

            bool bNoMore = false;
            vector<bool> vbInliers;
            int nInliers;
            bool bConverge = false;  // 判断sim3成功与否
            cv::Mat mTcm;
            while(!bConverge && !bNoMore)
            {
                // 2.5 计算sim3
                mTcm = solver.iterate(20, bNoMore, vbInliers, nInliers, bConverge);
            }

            //cout << "Num inliers: " << nInliers << endl;
            // 2.6 如果当前候选关键帧通过了要求
            if(bConverge)
            {
                //cout <<"BoW: " << nInliers << " inliers in Sim3Solver" << endl;

                // Match by reprojection
                //int nNumCovisibles = 5;
                vpCovKFi.clear();

                // 2.6.1 取出这个候选关键帧对应的5个共视关键帧
                vpCovKFi = pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(nNumCovisibles);
                // int nInitialCov = vpCovKFi.size();  // 没用，注释掉了

                // 2.6.2 把候选关键帧也加进来并做成set
                vpCovKFi.push_back(pMostBoWMatchesKF);
                set<KeyFrame*> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end());

                set<MapPoint*> spMapPoints;     // 存放这些关键帧对应的所有的MP（用于判断是否添加过）
                vector<MapPoint*> vpMapPoints;  // 存放这些关键帧对应的所有的MP（具有唯一性）
                vector<KeyFrame*> vpKeyFrames;  // 存放关键帧，与vpMapPoints一一对应
                // vpMapPoints  {mp1, mp2, mp3, mp4}
                // vpKeyFrames  {kf1, kf1, kf2, kf3}
                for(KeyFrame* pCovKFi : vpCovKFi)
                {
                    for(MapPoint* pCovMPij : pCovKFi->GetMapPointMatches())
                    {
                        if(!pCovMPij || pCovMPij->isBad())
                            continue;

                        if(spMapPoints.find(pCovMPij) == spMapPoints.end())
                        {
                            spMapPoints.insert(pCovMPij);
                            vpMapPoints.push_back(pCovMPij);
                            vpKeyFrames.push_back(pCovKFi);
                        }
                    }
                }
                //cout << "Point cloud: " << vpMapPoints.size() << endl;

                // 候选关键帧到当前关键帧的sim3
                g2o::Sim3 gScm(Converter::toMatrix3d(solver.GetEstimatedRotation()),
                               Converter::toVector3d(solver.GetEstimatedTranslation()),
                               solver.GetEstimatedScale());
                // 世界坐标到候选关键帧的sim3
                g2o::Sim3 gSmw(Converter::toMatrix3d(pMostBoWMatchesKF->GetRotation()),
                               Converter::toVector3d(pMostBoWMatchesKF->GetTranslation()),
                               1.0);
                g2o::Sim3 gScw = gScm*gSmw; // Similarity matrix of current from the world position

                // 2.6.3 准备匹配工作，得到世界坐标到当前关键帧的相似变换，再声明存MP与关键帧的向量
                cv::Mat mScw = Converter::toCvMat(gScw);

                vector<MapPoint*> vpMatchedMP;
                vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                vector<KeyFrame*> vpMatchedKF;
                vpMatchedKF.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));

                // 2.6.4 将前面候选关键帧及其共视帧的所有mp点投影到当前关键帧，匹配得到匹配点
                int numProjMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5);
                // vpMapPoints与vpMatchedMP位置是一一对应的，匹配之后匹配上的点会在vpMatchedMP里面不为NULL，vpMatchedKF同理
                // cout <<"BoW: " << numProjMatches << " matches between " << vpMapPoints.size() << " points with coarse Sim3" << endl;
                // 匹配的点数起码要超过50
                if(numProjMatches >= nProjMatches)
                {
                    // Optimize Sim3 transformation with every matches
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    bool bFixedScale = mbFixScale;
                    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                        bFixedScale = false;

                    // 2.6.5 优化gScm
                    int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pKFi, vpMatchedMP, gScm, 10, mbFixScale, mHessian7x7, true);
                    //cout <<"BoW: " << numOptMatches << " inliers in the Sim3 optimization" << endl;
                    //cout << "Inliers in Sim3 optimization: " << numOptMatches << endl;

                    if(numOptMatches >= nSim3Inliers)  // 20
                    {
                        //cout <<"BoW: " << numOptMatches << " inliers in Sim3 optimization" << endl;
                        g2o::Sim3 gSmw(Converter::toMatrix3d(pMostBoWMatchesKF->GetRotation()), Converter::toVector3d(pMostBoWMatchesKF->GetTranslation()), 1.0);
                        g2o::Sim3 gScw = gScm*gSmw; // Similarity matrix of current from the world position
                        cv::Mat mScw = Converter::toCvMat(gScw);

                        vector<MapPoint*> vpMatchedMP;
                        vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));

                        // 2.6.6 根据优化后的结果重新计算匹配点，要求更苛刻一些，在vpMatchedMP上从 vpMapPoints（不变） 新拽出一些匹配点
                        int numProjOptMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpMatchedMP, 5, 1.0);
                        //cout <<"BoW: " << numProjOptMatches << " matches after of the Sim3 optimization" << endl;
                        if(numProjOptMatches >= nProjOptMatches)  // 80
                        {
                            /// 以下为调试信息
                            cout << "BoW: Current KF " << mpCurrentKF->mnId << "; candidate KF " << pKFi->mnId << endl;
                            cout << "BoW: There are " << numProjOptMatches << " matches between them with the optimized Sim3" << endl;
                            int max_x = -1, min_x = 1000000;
                            int max_y = -1, min_y = 1000000;
                            for(MapPoint* pMPi : vpMatchedMP)
                            {
                                if(!pMPi || pMPi->isBad())
                                {
                                    continue;
                                }

                                tuple<size_t, size_t> indexes = pMPi->GetIndexInKeyFrame(pKFi);
                                int index = get<0>(indexes);
                                if(index >= 0)
                                {
                                    int coord_x = pKFi->mvKeysUn[index].pt.x;
                                    if(coord_x < min_x)
                                    {
                                        min_x = coord_x;
                                    }
                                    if(coord_x > max_x)
                                    {
                                        max_x = coord_x;
                                    }
                                    int coord_y = pKFi->mvKeysUn[index].pt.y;
                                    if(coord_y < min_y)
                                    {
                                        min_y = coord_y;
                                    }
                                    if(coord_y > max_y)
                                    {
                                        max_y = coord_y;
                                    }
                                }
                            }
                            //cout << "BoW: Coord in X -> " << min_x << ", " << max_x << "; and Y -> " << min_y << ", " << max_y << endl;
                            //cout << "BoW: features area in X -> " << (max_x - min_x) << " and Y -> " << (max_y - min_y) << endl;
                            // 调试完毕
                            int nNumKFs = 0;
                            //vpMatchedMPs = vpMatchedMP;
                            //vpMPs = vpMapPoints;
                            // Check the Sim3 transformation with the current KeyFrame covisibles
                            vector<KeyFrame*> vpCurrentCovKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(nNumCovisibles);  // 5
                            //cout << "---" << endl;
                            //cout << "BoW: Geometrical validation" << endl;
                            int j = 0;
                            while(nNumKFs < 3 && j<vpCurrentCovKFs.size())
                            //for(int j=0; j<vpCurrentCovKFs.size(); ++j)
                            {
                                KeyFrame* pKFj = vpCurrentCovKFs[j];
                                cv::Mat mTjc = pKFj->GetPose() * mpCurrentKF->GetPoseInverse();
                                g2o::Sim3 gSjc(Converter::toMatrix3d(mTjc.rowRange(0, 3).colRange(0, 3)), Converter::toVector3d(mTjc.rowRange(0, 3).col(3)), 1.0);
                                g2o::Sim3 gSjw = gSjc * gScw;
                                int numProjMatches_j = 0;
                                vector<MapPoint*> vpMatchedMPs_j;
                                // 查看两个关键帧匹配关系，vpMapPoints重新清零，添加了更多的mp点
                                bool bValid = DetectCommonRegionsFromLastKF(pKFj, pMostBoWMatchesKF, gSjw, numProjMatches_j, vpMapPoints, vpMatchedMPs_j);

                                if(bValid)
                                {
                                    //cout << "BoW: KF " << pKFj->mnId << " has " << numProjMatches_j << " matches" << endl;
                                    cv::Mat Tc_w = mpCurrentKF->GetPose();
                                    cv::Mat Tw_cj = pKFj->GetPoseInverse();
                                    cv::Mat Tc_cj = Tc_w * Tw_cj;
                                    cv::Vec3d vector_dist =  Tc_cj.rowRange(0, 3).col(3);
                                    cv::Mat Rc_cj = Tc_cj.rowRange(0, 3).colRange(0, 3);
                                    double dist = cv::norm(vector_dist);
                                    cout << "BoW: KF " << pKFi->mnId << " to KF " << pKFj->mnId << " is separated by " << dist << " meters" << endl;
                                    cout << "BoW: Rotation between KF -> " << Rc_cj << endl;
                                    vector<float> v_euler = Converter::toEuler(Rc_cj);
                                    v_euler[0] *= 180 /3.1415;
                                    v_euler[1] *= 180 /3.1415;
                                    v_euler[2] *= 180 /3.1415;
                                    cout << "BoW: Rotation in angles (x, y, z) -> (" << v_euler[0] << ", " << v_euler[1] << ", " << v_euler[2] << ")" << endl;
                                    nNumKFs++;
                                    /*if(numProjMatches_j > numProjOptMatches)
                                    {
                                        pLastCurrentKF = pKFj;
                                        g2oScw = gSjw;
                                        vpMatchedMPs = vpMatchedMPs_j;
                                    }*/
                                }

                                j++;
                            }

                            if(nNumKFs < 3)
                            {
                                vnStage[index] = 8;
                                vnMatchesStage[index] = nNumKFs;
                            }
                            // 
                            if(nBestMatchesReproj < numProjOptMatches)
                            {
                                nBestMatchesReproj = numProjOptMatches;
                                nBestNumCoindicendes = nNumKFs;
                                pBestMatchedKF = pMostBoWMatchesKF;
                                g2oBestScw = gScw;
                                vpBestMapPoints = vpMapPoints;
                                vpBestMatchedMapPoints = vpMatchedMP;
                            }
                        }
                    }
                }
            }
        }
        // index++;
    }

    if(nBestMatchesReproj > 0)
    {
        pLastCurrentKF = mpCurrentKF;
        nNumCoincidences = nBestNumCoindicendes;
        pMatchedKF2 = pBestMatchedKF;
        pMatchedKF2->SetNotErase();
        g2oScw = g2oBestScw;
        vpMPs = vpBestMapPoints;
        vpMatchedMPs = vpBestMatchedMapPoints;

        return nNumCoincidences >= 3;
    }
    else
    {
        // 回环失败，跳过，下面代码用于输出调试信息
        int maxStage = -1;
        int maxMatched;
        for(int i=0; i<vnStage.size(); ++i)
        {
            if(vnStage[i] > maxStage)
            {
                maxStage = vnStage[i];
                maxMatched = vnMatchesStage[i];
            }
        }

//        f_succes_pr << mpCurrentKF->mNameFile << " " << std::to_string(maxStage) << endl;
//        f_succes_pr << "% NumCand: " << std::to_string(numCandidates) << "; matches: " << std::to_string(maxMatched) << endl;
    }
    return false;
}

/** 通过投影找匹配
 * @param pCurrentKF 当前关键帧或者与之共视的
 * @param pMatchedKF 候选关键帧
 * @param gScw 变换矩阵
 * @param nNumProjMatches 匹配的个数
 * @param vpMPs 候选关键帧及其共视帧及其共视帧的共视帧的所有mp
 * @param vpMatchedMPs 匹配上的点
 * @return 匹配成功或失败
 */
bool LoopClosing::DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches, 
                                                std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
    //cout << "Projection from last KF with " << nNumProjMatches << " matches" << endl;

    int nProjMatches = 30;
    if(nNumProjMatches >= nProjMatches)
    {
        /*cout << "PR_From_LastKF: KF " << pCurrentKF->mnId << " has " << nNumProjMatches << " with KF " << pMatchedKF->mnId << endl;

        int max_x = -1, min_x = 1000000;
        int max_y = -1, min_y = 1000000;
        for(MapPoint* pMPi : vpMatchedMPs)
        {
            if(!pMPi || pMPi->isBad())
            {
                continue;
            }

            tuple<size_t, size_t> indexes = pMPi->GetIndexInKeyFrame(pMatchedKF);
            int index = get<0>(indexes);
            if(index >= 0)
            {
                int coord_x = pCurrentKF->mvKeysUn[index].pt.x;
                if(coord_x < min_x)
                {
                    min_x = coord_x;
                }
                if(coord_x > max_x)
                {
                    max_x = coord_x;
                }
                int coord_y = pCurrentKF->mvKeysUn[index].pt.y;
                if(coord_y < min_y)
                {
                    min_y = coord_y;
                }
                if(coord_y > max_y)
                {
                    max_y = coord_y;
                }
            }
        }*/
        //cout << "PR_From_LastKF: Coord in X -> " << min_x << ", " << max_x << "; and Y -> " << min_y << ", " << max_y << endl;
        //cout << "PR_From_LastKF: features area in X -> " << (max_x - min_x) << " and Y -> " << (max_y - min_y) << endl;


        return true;
    }

    return false;
}

/** 通过投影找匹配
 * @param pCurrentKF 当前关键帧或者与之共视的
 * @param pMatchedKFw 候选关键帧
 * @param g2oScw 变换矩阵
 * @param spMatchedMPinOrigin 没用到
 * @param vpMapPoints 候选关键帧及其共视帧及其共视帧的共视帧的所有mp
 * @param vpMatchedMapPoints 匹配上的点
 * @return 匹配的个数
 */
int LoopClosing::FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw, 
                                         set<MapPoint*> &spMatchedMPinOrigin, vector<MapPoint*> &vpMapPoints, 
                                         vector<MapPoint*> &vpMatchedMapPoints)
{
    int nNumCovisibles = 5;
    vector<KeyFrame*> vpCovKFm = pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles);  // 获取候选关键帧共视最佳的5个
    int nInitialCov = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<KeyFrame*> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    set<KeyFrame*> spCurrentCovisbles = pCurrentKF->GetConnectedKeyFrames();  // 获得与pCurrentKF连接的关键帧（共同mp点数>15）
    // 1. 遍历候选关键帧共视最佳的5个关键帧
    for(int i=0; i<nInitialCov; ++i)
    {
        // 1.1 候选关键帧每个共视关键帧的共视关键帧。。。。。。有点绕
        vector<KeyFrame*> vpKFs = vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles);
        int nInserted = 0;
        int j = 0;
        while(j < vpKFs.size() && nInserted < nNumCovisibles)
        {
            // 如果上面得到帧不在候选关键帧共视最佳的5个（vpCovKFm）里面，且不在候选关键帧的所有共视关键帧中，添加至spCheckKFs组成候选关键帧集合，不超过5个
            // 一顿操作结果没b用，，，，这个while循环里面所有代码都没用。。
            if(spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && spCurrentCovisbles.find(vpKFs[j]) == spCurrentCovisbles.end())
            {
                spCheckKFs.insert(vpKFs[j]);
                ++nInserted;
            }
            ++j;
        }
        // 1.2 把候选关键帧的共视帧与共视帧的共视帧放到一起，目的为了提取候选关键帧那一片儿所有的mp
        vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
    }
    set<MapPoint*> spMapPoints;
    vpMapPoints.clear();
    vpMatchedMapPoints.clear();
    // 2. 提取一堆mp
    for(KeyFrame* pKFi : vpCovKFm)
    {
        for(MapPoint* pMPij : pKFi->GetMapPointMatches())
        {
            if(!pMPij || pMPij->isBad())
                continue;

            if(spMapPoints.find(pMPij) == spMapPoints.end())
            {
                spMapPoints.insert(pMPij);
                vpMapPoints.push_back(pMPij);
            }
        }
    }
    //cout << "Point cloud: " << vpMapPoints.size() << endl;

    cv::Mat mScw = Converter::toCvMat(g2oScw);

    ORBmatcher matcher(0.9, true);
    // 3. 匹配
    vpMatchedMapPoints.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
    // vpMatchedMapPoints存放的是pCurrentKF中特征点对应的候选帧中的mp，它本身可能自己也有mp，返回的是vpMatchedMapPoints里面不为NULL的数量
    int num_matches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

/**
 * @brief 相同地图检测到共同区域叫回环，不同地图叫融合，这个函数是在检测到回环后进行修正优化位姿
 */
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    // 1. 发出暂停localmapping线程指令，处理还没来得及处理的关键帧，并不重新生成mp
    mpLocalMapper->RequestStop();
    mpLocalMapper->EmptyQueue(); // Proccess keyframes in the queue

    // 2. 如果正在进行全局BA，丢弃它
    // If a Global Bundle Adjustment is running, abort it
    cout << "Request GBA abort" << endl;
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            cout << "GBA running... Abort!" << endl;
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    cout << "start updating connections" << endl;
    // 3. 更新当前关键帧连接关系，并检查本质图的完整性
    assert(mpCurrentKF->GetMap()->CheckEssentialGraph());
    mpCurrentKF->UpdateConnections();
    assert(mpCurrentKF->GetMap()->CheckEssentialGraph());

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    // 4. 通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的MapPoints
    // 当前帧与世界坐标系之间的Sim变换在ComputeSim3函数中已经确定并优化，
    // 通过相对位姿关系，可以确定这些相连的关键帧与世界坐标系之间的Sim3变换

    // 取出与当前帧相连的关键帧，包括当前关键帧
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    // 通过mg2oLoopScw计算的放CorrectedSim3， 通过pKFi->GetPose()计算的放NonCorrectedSim3也就是回环前的位姿
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    // 先将mpCurrentKF的Sim3变换存入，固定不动
    CorrectedSim3[mpCurrentKF] = mg2oLoopScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();

    Map* pLoopMap = mpCurrentKF->GetMap();

    {
        // Get Map Mutex
        unique_lock<mutex> lock(pLoopMap->mMutexMapUpdate);

        const bool bImuInit = pLoopMap->isImuInitialized();
        // 4.1 通过位姿传播，得到Sim3调整后其它与当前帧相连关键帧的位姿（只是得到，还没有修正）
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();
            // currentKF在前面已经添加
            if(pKFi != mpCurrentKF)
            {
                // 得到当前帧到pKFi帧的相对变换
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
                cv::Mat tic = Tic.rowRange(0, 3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
                // 当前帧的位姿固定不动，其它的关键帧根据相对关系得到Sim3调整的位姿
                g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oLoopScw;
                //Pose corrected with the Sim3 of the loop closure
                // 得到闭环g2o优化后各个关键帧的位姿
                CorrectedSim3[pKFi] = g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
            cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
            //Pose without correction
            // 当前帧相连关键帧，没有进行闭环g2o优化的位姿
            NonCorrectedSim3[pKFi] = g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        cout << "LC: start correcting KeyFrames" << endl;
        cout << "LC: there are " << CorrectedSim3.size() << " KFs in the local window" << endl;
        // 4.2 步骤4.1得到调整相连帧位姿后，修正这些关键帧的MapPoints
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF == mpCurrentKF->mnId)  // 防止重复修正，同时在优化的时候也可以找到对应的kf
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                // 将该未校正的eigP3Dw先从世界坐标系映射到未校正的pKFi相机坐标系，然后再反映射到校正后的世界坐标系下
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            // 4.3 将Sim3转换为SE3，根据更新的Sim3，更新关键帧的位姿
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();
            // cout << "scale for loop-closing: " << s << endl;

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

            pKFi->SetPose(correctedTiw);

            // Correct velocity according to orientation correction
            if(bImuInit)
            {
                Eigen::Matrix3d Rcor = eigR.transpose()*g2oSiw.rotation().toRotationMatrix();
                pKFi->SetVelocity(Converter::toCvMat(Rcor)*pKFi->GetVelocity());
            }

            // Make sure connections are updated
            // 4.4 根据共视关系更新当前帧与其它关键帧之间的连接
            pKFi->UpdateConnections();
        }
        // TODO Check this index increasement
        mpAtlas->GetCurrentMap()->IncreaseChangeIndex();
        cout << "LC: end correcting KeyFrames" << endl;


        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        cout << "LC: start replacing duplicated" << endl;
        // 4.5 提前先融合mpCurrentKF中的MP
        // mvpLoopMatchedMPs 大小与mpCurrentKF中特征点数量一样，mpCurrentKF本身每个特征点可能有对应的MP
        // 而mvpLoopMatchedMPs是从一堆候选帧对应的mp中算得的对应的MP，理论上讲同一个特征点有可能对应两个MP
        // 也有可能对应一个，也有可能都没有
        for(size_t i=0; i<mvpLoopMatchedMPs.size(); i++)
        {
            // 如果没有对应的候选点，保持原来的对应关系
            if(mvpLoopMatchedMPs[i])
            {
                MapPoint* pLoopMP = mvpLoopMatchedMPs[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                // 如果pCurMP存在表明他有原始点，直接替换
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else  // 如果没有表明没有原始点，则添加
                {
                    mpCurrentKF->AddMapPoint(pLoopMP, i);
                    pLoopMP->AddObservation(mpCurrentKF, i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }
        cout << "LC: end replacing duplicated" << endl;
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //cout << "LC: start SearchAndFuse" << endl;
    // 5. 再融合与mpCurrentKF相连关键帧对应的MP
    // mvpLoopMapPoints 表示一堆候选关键帧，每次新关键帧回环成功mvpLoopMapPoints都会重新计算
    // CorrectedSim3里面装的是当前关键帧的GetVectorCovisibleKeyFrames，不包括当前关键帧
    SearchAndFuse(CorrectedSim3, mvpLoopMapPoints);
    //cout << "LC: end SearchAndFuse" << endl;


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    //cout << "LC: start updating covisibility graph" << endl;
    // 6. 更新当前关键帧之间的共视相连关系，得到因闭环时MapPoints融合而新得到的连接关系
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;
    // 6.1 遍历当前帧相连关键帧（一级相连）
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        // 6.2 得到与当前帧相连关键帧的相连关键帧（二级相连）
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        // 6.3 更新一级相连关键帧的连接关系
        pKFi->UpdateConnections();
        // 6.4 取出该帧更新后的连接关系
        LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
        // 6.5 从连接关系中去除闭环之前的二级连接关系，剩下的连接就是由闭环得到的连接关系
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        // 6.6 从连接关系中去除闭环之前的一级连接关系，剩下的连接就是由闭环得到的连接关系
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }
    //cout << "LC: end updating covisibility graph" << endl;

    // Optimize graph
    //cout << "start opt essentialgraph" << endl;
    bool bFixedScale = mbFixScale;
    // TODO CHECK; Solo para el monocular inertial
    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
        bFixedScale = false;


    //cout << "Optimize essential graph" << endl;
    // 7. 进行EssentialGraph优化，LoopConnections是形成闭环后新生成的连接关系
    if(pLoopMap->IsInertial() && pLoopMap->isImuInitialized())
    {
        //cout << "With 4DoF" << endl;
        Optimizer::OptimizeEssentialGraph4DoF(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
    }
    else
    {
        //cout << "With 7DoF" << endl;
        Optimizer::OptimizeEssentialGraph(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixedScale);
    }


    //cout << "Optimize essential graph finished" << endl;
    //usleep(5*1000*1000);

    mpAtlas->InformNewBigChange();

    // Add loop edge
    // 8. 添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）
    // BUG 这两句话应该放在OptimizeEssentialGraph之前，因为OptimizeEssentialGraph的步骤4.2中有优化，（???）
    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment (Only if few keyframes, if not it would take too much time)
    // 回环地图没有imu初始化或者比较小时才执行全局BA，否则太慢
    if(!pLoopMap->isImuInitialized() || (pLoopMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1))
    {
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;

        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, pLoopMap, mpCurrentKF->mnId);
    }

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId; //TODO old varible, it is not use in the new algorithm
}

/** 
 * @brief 调试用的，先不看
 */
void LoopClosing::printReprojectionError(set<KeyFrame*> &spLocalWindowKFs, KeyFrame* mpCurrentKF, string &name)
{
    string path_imgs = "./test_Reproj/";
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        //cout << "KF " << pKFi->mnId << endl;
        cv::Mat img_i = cv::imread(pKFi->mNameFile, CV_LOAD_IMAGE_UNCHANGED);
        //cout << "Image -> " << img_i.cols << ", " << img_i.rows << endl;
        cv::cvtColor(img_i, img_i, CV_GRAY2BGR);
        //cout << "Change of color in the image " << endl;

        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        int num_points = 0;
        for(int j=0; j<vpMPs.size(); ++j)
        {
            MapPoint* pMPij = vpMPs[j];
            if(!pMPij || pMPij->isBad())
            {
                continue;
            }

            cv::KeyPoint point_img = pKFi->mvKeysUn[j];
            cv::Point2f reproj_p;
            float u, v;
            bool bIsInImage = pKFi->ProjectPointUnDistort(pMPij, reproj_p, u, v);
            if(bIsInImage){
                //cout << "Reproj in the image" << endl;
                cv::circle(img_i, point_img.pt, 1/*point_img.octave*/, cv::Scalar(0, 255, 0));
                cv::line(img_i, point_img.pt, reproj_p, cv::Scalar(0, 0, 255));
                num_points++;
            }
            else
            {
                //cout << "Reproj out of the image" << endl;
                cv::circle(img_i, point_img.pt, point_img.octave, cv::Scalar(0, 0, 255));
            }

        }
        //cout << "Image painted" << endl;
        string filename_img = path_imgs +  "KF" + to_string(mpCurrentKF->mnId) + "_" + to_string(pKFi->mnId) +  name + "points" + to_string(num_points) + ".png";
        cv::imwrite(filename_img, img_i);
    }

}

/**
 * @brief 带有imu时的地图融合，待融合的地图所有东西都融到当前地图里面，pMergeMap没有设置为BAD
 * 1. 关闭全局BA
 * 2. 暂停localmapping线程
 * 3. 将当前地图里的元素位姿转到待融合地图坐标系下
 * 4. 帮助当前地图快速完全初始化
 * 5. 将待融合地图里的元素全部挪到当前地图
 * 6. 更改树的关系
 * 7. 取两个地图匹配帧的附近帧及其MP做融合（融合部分有点小疑问）
 * 8. 更新连接关系，若融合前当前地图关键帧数量少，则返回
 * 9. 否则做跨地图的局部窗口优化
 */
void LoopClosing::MergeLocal2()
{
    Verbose::PrintMess("MERGE: Merge detected!!!!", Verbose::VERBOSITY_NORMAL);

    // int numTemporalKFs = 11; //TODO (set by parameter): Temporal KFs in the local window if the map is inertial.

    
    // NonCorrectedSim3[mpCurrentKF]=mg2oLoopScw;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    cout << "Check Full Bundle Adjustment" << endl;
    // If a Global Bundle Adjustment is running, abort it
    // 1. 如果正在进行全局BA，丢弃它
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        bRelaunchBA = true;
    }


    Verbose::PrintMess("MERGE-VISUAL: Request Stop Local Mapping", Verbose::VERBOSITY_DEBUG);
    // 2. 发出暂停localmapping线程指令
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    // 等待完全暂停
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }
    Verbose::PrintMess("MERGE-VISUAL: Local Map stopped", Verbose::VERBOSITY_DEBUG);

    // 3. 开始融合！此时地图不再更新，不再新添加帧与mp，更新当前地图关键帧及mp的坐标
    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();
    {
        // mSold_new = gSw2w1
        float s_on = mSold_new.scale();
        cv::Mat R_on = Converter::toCvMat(mSold_new.rotation().toRotationMatrix());
        cv::Mat t_on = Converter::toCvMat(mSold_new.translation());

        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

        // cout << "KFs before empty: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << endl;
        // 3.1 处理还没来得及处理的关键帧，并不重新生成mp
        mpLocalMapper->EmptyQueue();
        // cout << "KFs after empty: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << endl;

        // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        // cout << "updating active map to merge reference" << endl;
        // cout << "curr merge KF id: " << mpCurrentKF->mnId << endl;
        // cout << "curr tracking KF id: " << mpTracker->GetLastKeyFrame()->mnId << endl;
        bool bScaleVel = false;
        if(s_on!=1)
            bScaleVel = true;
        // 3.2 将当前map中的帧与mp全部更新到待融合的地图坐标系下，类似于imu初始化
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(R_on, s_on, bScaleVel, t_on);
        // 3.3 更新普通帧mLastFrame，与mCurrentFrame
        mpTracker->UpdateFrameIMU(s_on, mpCurrentKF->GetImuBias(), mpTracker->GetLastKeyFrame());

        // std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    }
    // 当前帧对应的地图的关键帧数量
    const int numKFnew = pCurrentMap->KeyFramesInMap();

    // 4. 如果当前地图没有初始化完全，帮助快速优化，作者可能认为反正都检测到融合了，这里就随随便便先优化下，回头直接全部放到融合地图里就好了
    if((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO)&& !pCurrentMap->GetIniertialBA2())
    {
        // Map is not completly initialized
        Eigen::Vector3d bg, ba;
        bg << 0., 0., 0.;
        ba << 0., 0., 0.;
        // 4.1 优化目标为速度与偏置
        Optimizer::InertialOptimization(pCurrentMap, bg, ba);
        IMU::Bias b (ba[0], ba[1], ba[2], bg[0], bg[1], bg[2]);
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        mpTracker->UpdateFrameIMU(1.0f, b, mpTracker->GetLastKeyFrame());

        // Set map initialized
        pCurrentMap->SetIniertialBA2();
        pCurrentMap->SetIniertialBA1();
        pCurrentMap->SetImuInitialized();

    }


    cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;
    // map<KeyFrame *, g2o::Sim3, std::less<KeyFrame *>, Eigen::aligned_allocator<std::pair<const KeyFrame *, g2o::Sim3>>>
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    // Load KFs and MPs from merge map
    cout << "updating current map" << endl;
    {
        // Get Merge Map Mutex (This section stops tracking!!)
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

        
        vector<KeyFrame*> vpMergeMapKFs = pMergeMap->GetAllKeyFrames();
        vector<MapPoint*> vpMergeMapMPs = pMergeMap->GetAllMapPoints();

        // 5.1 处理需要融合的地图的关键帧
        for(KeyFrame* pKFi : vpMergeMapKFs)
        {
            if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pMergeMap)
            {
                continue;
            }

            // Make sure connections are updated
            // 更新其归属地图，但是没有连接关系
            pKFi->UpdateMap(pCurrentMap);
            pCurrentMap->AddKeyFrame(pKFi);
            pMergeMap->EraseKeyFrame(pKFi);
        }
        // 5.2 处理需要融合的地图的mp
        for(MapPoint* pMPi : vpMergeMapMPs)
        {
            if(!pMPi || pMPi->isBad() || pMPi->GetMap() != pMergeMap)
                continue;

            pMPi->UpdateMap(pCurrentMap);
            pCurrentMap->AddMapPoint(pMPi);
            pMergeMap->EraseMapPoint(pMPi);
        }

        // Save non corrected poses (already merged maps)
        // 5.3 存放当前地图中位姿没有矫正的关键帧，下面这段没啥用注释了
        /*
        vector<KeyFrame*> vpKFs = pCurrentMap->GetAllKeyFrames();
        for(KeyFrame* pKFi : vpKFs)
        {
            cv::Mat Tiw = pKFi->GetPose();
            cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
            cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
            NonCorrectedSim3[pKFi] = g2oSiw;
        } */
    }

    cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    cout << "end updating current map" << endl;

    // Critical zone
    // 检查当前地图的EssentialGraph是否完整，主要是看融合后对地图是否有影响，调试可以用，下同
#ifdef DEBUG
    bool good = pCurrentMap->CheckEssentialGraph();
#endif
    /*if(!good)
        cout << "BAD ESSENTIAL GRAPH!!" << endl;*/

    cout << "Update essential graph" << endl;
    // mpCurrentKF->UpdateConnections(); // to put at false mbFirstConnection
    // 待融合地图的初始帧的mbFirstConnection设置为false，因为这里面所有关键帧将融到当前地图，一个地图里只能有一个为true
    pMergeMap->GetOriginKF()->SetFirstConnection(false);
    // 6. 下面这段代码主要是更新父子关系，mpMergeMatchedKF表示待融合地图中与当前关键帧对应上的帧
    // 因为整个待融合地图要融入到当前地图里，为了避免有两个父节点，mpMergeMatchedKF的原始父节点变成了它的子节点，而当前关键帧成了mpMergeMatchedKF的父节点
    // 同理，为了避免mpMergeMatchedKF原始父节点（现在已成它的子节点）有两个父节点，需要向上一直改到待融合地图最顶端的父节点
    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    pNewChild = mpMergeMatchedKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpMergeMatchedKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpMergeMatchedKF->ChangeParent(mpCurrentKF);
    // 一直向上搜索，调换父子节点关系
    while(pNewChild)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();
        pNewChild->ChangeParent(pNewParent);
        pNewParent = pNewChild;
        pNewChild = pOldParent;
    }

    cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    cout << "end update essential graph" << endl;
#ifdef DEBUG
    good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 1!!" << endl;
#endif
    cout << "Update relationship between KFs" << endl;
    
    // 7. 融合MP准备
    // 把mpMergeMatchedKF及其共视关键帧添加到mvpMergeConnectedKFs里面，且保证里面的帧数为6
    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    vector<KeyFrame*> aux = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.insert(mvpMergeConnectedKFs.end(), aux.begin(), aux.end());
    if (mvpMergeConnectedKFs.size()>6)
        mvpMergeConnectedKFs.erase(mvpMergeConnectedKFs.begin()+6, mvpMergeConnectedKFs.end());
    /*mvpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);*/
    // 更新连接关系
    mpCurrentKF->UpdateConnections();
    // 跟上面操作一样
    vector<KeyFrame*> vpCurrentConnectedKFs;
    vpCurrentConnectedKFs.push_back(mpCurrentKF);
    /*vpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);*/
    aux = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.insert(vpCurrentConnectedKFs.end(), aux.begin(), aux.end());
    if (vpCurrentConnectedKFs.size()>6)
        vpCurrentConnectedKFs.erase(vpCurrentConnectedKFs.begin()+6, vpCurrentConnectedKFs.end());

    // 找到mvpMergeConnectedKFs里面对应的mp，数量限制在1000
    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(), vpMPs.end());
        if(spMapPointMerge.size()>1000)
            break;
    }

    cout << "vpCurrentConnectedKFs.size() " << vpCurrentConnectedKFs.size() << endl;
    cout << "mvpMergeConnectedKFs.size() " << mvpMergeConnectedKFs.size() << endl;
    cout << "spMapPointMerge.size() " << spMapPointMerge.size() << endl;

    // spMapPointMerge所有mp添加到vpCheckFuseMapPoint
    vector<MapPoint*> vpCheckFuseMapPoint; // MapPoint vector from current map to allow to fuse duplicated points with the old map (merge)
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));
    cout << "Finished to update relationship between KFs" << endl;

    cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;
    // TODO pMergeMap没有设置为BAD
#ifdef DEBUG
    good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 2!!" << endl;
#endif
    cout << "start SearchAndFuse" << endl;
    // 当前地图的关键帧与待融合地图的mp
    // 当前关键帧中不重复的mp赋予待融合的mp
    // 如果冲突，替换成待融合地图的mp
    SearchAndFuse(vpCurrentConnectedKFs, vpCheckFuseMapPoint);
    cout << "end SearchAndFuse" << endl;

    cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

#ifdef DEBUG
    good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 3!!" << endl;
#endif
    cout << "Init to update connections" << endl;

    // 8. 更新连接关系
    for(KeyFrame* pKFi : vpCurrentConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    cout << "end update connections" << endl;

    cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

#ifdef DEBUG
    good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 4!!" << endl;
#endif
    // TODO Check: If new map is too small, we suppose that not informaiton can be propagated from new to old map
    // 如果更新前的当前地图包含的关键帧少于10个，直接结束返回，否则执行BA优化
    if (numKFnew < 10)
    {
        mpLocalMapper->Release();
        return;
    }

#ifdef DEBUG
    good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 5!!" << endl;
#endif
    // Perform BA
    bool bStopFlag = false;
    KeyFrame* pCurrKF = mpTracker->GetLastKeyFrame();
    cout << "start MergeInertialBA" << endl;
    Optimizer::MergeInertialBA(pCurrKF, mpMergeMatchedKF, &bStopFlag, pCurrentMap, CorrectedSim3);
    cout << "end MergeInertialBA" << endl;

#ifdef DEBUG
    good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 6!!" << endl;
#endif
    // Release Local Mapping.
    mpLocalMapper->Release();


    return;
}

/**
 * @brief 当前地图非imu时融合函数，待融合的地图所有东西都融到当前地图里面，以待融合地图的坐标系为准
 * 操作会多一些，后面涉及优化essentialgrap与全局BA
 * 1. 关闭全局BA
 * 2. 暂停localmapping线程
 * 3. 开始融合！此时地图不再更新，不再新添加帧与mp，提取相关的关键帧与MP
 * 4. 求解位姿关系，并将spLocalWindowKFs 与spLocalWindowMPs 转移到待融合地图坐标下
 * 5. 把上面做的关键帧与MP更换所属地图
 * 6. 更新父子关系，与另一个容和函数同理
 * 7. 融合MP
 * 8. 更新连接关系
 * 9. 局部优化
 * 10. 处理当前地图中剩下的元素，更改位姿，更换所属地图
 * 11. 全局优化，收尾
 */
void LoopClosing::MergeLocal()
{
    Verbose::PrintMess("MERGE-VISUAL: Merge Visual detected!!!!", Verbose::VERBOSITY_NORMAL);
    //mpTracker->SetStepByStep(true);

    int numTemporalKFs = 15; //TODO (set by parameter): Temporal KFs in the local window if the map is inertial.

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    Verbose::PrintMess("MERGE-VISUAL: Check Full Bundle Adjustment", Verbose::VERBOSITY_DEBUG);
    // If a Global Bundle Adjustment is running, abort it
    // 1. 如果正在进行全局BA，丢弃它
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        bRelaunchBA = true;
    }

    Verbose::PrintMess("MERGE-VISUAL: Request Stop Local Mapping", Verbose::VERBOSITY_DEBUG);
    // 2. 发出暂停localmapping线程指令
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }
    Verbose::PrintMess("MERGE-VISUAL: Local Map stopped", Verbose::VERBOSITY_DEBUG);
    // 3. 开始融合！此时地图不再更新，不再新添加帧与mp，提取相关的关键帧与MP
    // 3.1 处理还没来得及处理的关键帧，并不重新生成mp
    mpLocalMapper->EmptyQueue();

    // Merge map will become in the new active map with the local window of KFs and MPs from the current map.
    // Later, the elements of the current map will be transform to the new active map reference, in order to keep real time tracking
    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();

    Verbose::PrintMess("MERGE-VISUAL: Initially there are " + to_string(pCurrentMap->KeyFramesInMap()) + " KFs and " + to_string(pCurrentMap->MapPointsInMap()) + " MPs in the active map ", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("MERGE-VISUAL: Initially there are " + to_string(pMergeMap->KeyFramesInMap()) + " KFs and " + to_string(pMergeMap->MapPointsInMap()) + " MPs in the matched map ", Verbose::VERBOSITY_DEBUG);
    //vector<KeyFrame*> vpMergeKFs = pMergeMap->GetAllKeyFrames();

    ////

    // Ensure current keyframe is updated
    // 更新连接关系
    mpCurrentKF->UpdateConnections();

    //Get the current KF and its neighbors(visual->covisibles; inertial->temporal+covisibles)
    // 3.2 建立局部窗口关键帧与mp
    set<KeyFrame*> spLocalWindowKFs;
    //Get MPs in the welding area from the current map
    set<MapPoint*> spLocalWindowMPs;
    // 下面这段不会实现吧，，毕竟进来这个函数就表示当前地图非imu
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        // 把mpCurrentKF前后全加进来，同时伴随对应的所有mp，可以理解成所有帧与点
        KeyFrame* pKFi = mpCurrentKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spLocalWindowKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
        }

        pKFi = mpCurrentKF->mNextKF;
        while(pKFi)
        {
            spLocalWindowKFs.insert(pKFi);

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
        }
    }
    else
    {
        spLocalWindowKFs.insert(mpCurrentKF);
    }
    // 3.3 与mpCurrentKF共视帧也加进来（用set储存，不会出现重复）
    vector<KeyFrame*> vpCovisibleKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spLocalWindowKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    Verbose::PrintMess("MERGE-VISUAL: Initial number of KFs in local window from active map: " + to_string(spLocalWindowKFs.size()), Verbose::VERBOSITY_DEBUG);
    const int nMaxTries = 3;
    int nNumTries = 0;
    // 3.4 疯狂往里面加与vpCovisibleKFs共视的，遍历 nMaxTries 次或者数量达到 numTemporalKFs
    // spLocalWindowKFs 里面目前存放了跟mpCurrentKF 相关的一些帧，但并不一定是当前地图的所有帧
    while(spLocalWindowKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        vpNewCovKFs.empty();
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spLocalWindowKFs.find(pKFcov) == spLocalWindowKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spLocalWindowKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }
    Verbose::PrintMess("MERGE-VISUAL: Last number of KFs in local window from the active map: " + to_string(spLocalWindowKFs.size()), Verbose::VERBOSITY_DEBUG);

    //TODO TEST
    //vector<KeyFrame*> vpTestKFs = pCurrentMap->GetAllKeyFrames();
    //spLocalWindowKFs.insert(vpTestKFs.begin(), vpTestKFs.end());
    // 3.5 再加一次MP点，加入了spLocalWindowKFs帧对应的所有点
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        set<MapPoint*> spMPs = pKFi->GetMapPoints();
        spLocalWindowMPs.insert(spMPs.begin(), spMPs.end());
    }
    Verbose::PrintMess("MERGE-VISUAL: Number of MPs in local window from active map: " + to_string(spLocalWindowMPs.size()), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("MERGE-VISUAL: Number of MPs in the active map: " + to_string(pCurrentMap->GetAllMapPoints().size()), Verbose::VERBOSITY_DEBUG);

    Verbose::PrintMess("-------", Verbose::VERBOSITY_DEBUG);
    // 3.6 下面这段代码做了类似上面的事，只不过换成了mpMergeMatchedKF  存放了不同的变量里面
    //-----------------------------------------------------------------------------------------------
    set<KeyFrame*> spMergeConnectedKFs;
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        KeyFrame* pKFi = mpMergeMatchedKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;
        }

        pKFi = mpMergeMatchedKF->mNextKF;
        while(pKFi)
        {
            spMergeConnectedKFs.insert(pKFi);
        }
    }
    else
    {
        spMergeConnectedKFs.insert(mpMergeMatchedKF);
    }
    vpCovisibleKFs = mpMergeMatchedKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spMergeConnectedKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    Verbose::PrintMess("MERGE-VISUAL: Initial number of KFs in the local window from matched map: " + to_string(spMergeConnectedKFs.size()), Verbose::VERBOSITY_DEBUG);
    nNumTries = 0;
    while(spMergeConnectedKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        for(KeyFrame* pKFi : spMergeConnectedKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spMergeConnectedKFs.find(pKFcov) == spMergeConnectedKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spMergeConnectedKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }
    Verbose::PrintMess("MERGE-VISUAL: Last number of KFs in the localwindow from matched map: " + to_string(spMergeConnectedKFs.size()), Verbose::VERBOSITY_DEBUG);

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(), vpMPs.end());
    }

    vector<MapPoint*> vpCheckFuseMapPoint;
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));
    //-----------------------------------------------------------------------------------------------------
    // 4. 求解位姿关系，并将spLocalWindowKFs 与spLocalWindowMPs 转移到待融合地图坐标下
    // mg2oMergeScw 存放了校正后世界坐标到mpCurrentKF的sim3 
    // g2oNonCorrectedScw 存放的是校正前的
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();

    cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
    cv::Mat twc = Twc.rowRange(0, 3).col(3);
    g2o::Sim3 g2oNonCorrectedSwc(Converter::toMatrix3d(Rwc), Converter::toVector3d(twc), 1.0);
    g2o::Sim3 g2oNonCorrectedScw = g2oNonCorrectedSwc.inverse();
    g2o::Sim3 g2oCorrectedScw = mg2oMergeScw; //TODO Check the transformation

    KeyFrameAndPose vCorrectedSim3, vNonCorrectedSim3;
    vCorrectedSim3[mpCurrentKF] = g2oCorrectedScw;
    vNonCorrectedSim3[mpCurrentKF] = g2oNonCorrectedScw;


    //TODO Time test
#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point timeStartTransfMerge = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point timeStartTransfMerge = std::chrono::monotonic_clock::now();
#endif
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
        {
            Verbose::PrintMess("Bad KF in correction", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(pKFi->GetMap() != pCurrentMap)
            Verbose::PrintMess("Other map KF, this should't happen", Verbose::VERBOSITY_DEBUG);

        g2o::Sim3 g2oCorrectedSiw;

        // 如果不是mpCurrentKF需要计算与mpCurrentKF的位姿关系，然后在通过mg2oMergeScw计算得pKFi在融合地图坐标系下的位姿
        if(pKFi != mpCurrentKF)
        {
            cv::Mat Tiw = pKFi->GetPose();
            cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
            cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
            //Pose without correction
            vNonCorrectedSim3[pKFi] = g2oSiw;

            cv::Mat Tic = Tiw*Twc;
            cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
            cv::Mat tic = Tic.rowRange(0, 3).col(3);
            g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
            g2oCorrectedSiw = g2oSic*mg2oMergeScw;
            vCorrectedSim3[pKFi] = g2oCorrectedSiw;
        }
        else
        {
            g2oCorrectedSiw = g2oCorrectedScw;
        }
        pKFi->mTcwMerge  = pKFi->GetPose();

        // Update keyframe pose with corrected Sim3.
        // First transform Sim3 to SE3 (scale translation)
        Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
        double s = g2oCorrectedSiw.scale();

        pKFi->mfScale = s;
        eigt *=(1./s); 

        //cout << "R: " << mg2oMergeScw.rotation().toRotationMatrix() << endl;
        //cout << "angle: " << 180*LogSO3(mg2oMergeScw.rotation().toRotationMatrix())/3.14 << endl;
        //cout << "t: " << mg2oMergeScw.translation() << endl;
        //[R t/s;0 1] 
        // s*Riw * Pw + tiw = Pi  此时Pi在i坐标系下的坐标，尺度保留的是原来的
        // Riw * Pw + tiw/s = Pi/s 此时Pi/s在i坐标系下的坐标，尺度是最新的的，所以Rt要这么保留
        cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

        pKFi->mTcwMerge = correctedTiw;

        //pKFi->SetPose(correctedTiw);

        // Make sure connections are updated
        //pKFi->UpdateMap(pMergeMap);
        //pMergeMap->AddKeyFrame(pKFi);
        //pCurrentMap->EraseKeyFrame(pKFi);

        //cout << "After -> Map current: " << pCurrentMap << "; New map: " << pKFi->GetMap() << endl;

        // ??????????也执行不了吧这里
        if(pCurrentMap->isImuInitialized())
        {
            Eigen::Matrix3d Rcor = eigR.transpose()*vNonCorrectedSim3[pKFi].rotation().toRotationMatrix();
            pKFi->mVwbMerge = Converter::toCvMat(Rcor)*pKFi->GetVelocity();
            //pKFi->SetVelocity(Converter::toCvMat(Rcor)*pKFi->GetVelocity()); // TODO: should add here scale s
        }

        //TODO DEBUG to know which are the KFs that had been moved to the other map
        //pKFi->mnOriginMapId = 5;
    }

    for(MapPoint* pMPi : spLocalWindowMPs)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
        g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse(); // R.t()   -R.t()*t/s    1/s
        g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

        // Project with non-corrected pose and project back with corrected pose
        cv::Mat P3Dw = pMPi->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        // eigP3Dw 转到i帧坐标下（尺度变化为1） 又转到了融合地图坐标系下，尺度变化为 1/s
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(eigP3Dw));
        Eigen::Matrix3d eigR = g2oCorrectedSwi.rotation().toRotationMatrix();
        // Rw2i * Riw1 = Rw2w1
        Eigen::Matrix3d Rcor = eigR * g2oNonCorrectedSiw.rotation().toRotationMatrix();

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);

        pMPi->mPosMerge = cvCorrectedP3Dw;
        //cout << "Rcor: " << Rcor << endl;
        //cout << "Normal: " << pMPi->GetNormal() << endl;
        // 更新平均观测方向
        pMPi->mNormalVectorMerge = Converter::toCvMat(Rcor) * pMPi->GetNormal();
        //pMPi->SetWorldPos(cvCorrectedP3Dw);
        //pMPi->UpdateMap(pMergeMap);
        //pMergeMap->AddMapPoint(pMPi);
        //pCurrentMap->EraseMapPoint(pMPi);
        //pMPi->UpdateNormalAndDepth();
    }
#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point timeFinishTransfMerge = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point timeFinishTransfMerge = std::chrono::monotonic_clock::now();
#endif
    std::chrono::duration<double, std::milli> timeTransfMerge = timeFinishTransfMerge - timeStartTransfMerge; // Time in milliseconds
    Verbose::PrintMess("MERGE-VISUAL: TRANSF ms: " + to_string(timeTransfMerge.count()), Verbose::VERBOSITY_DEBUG);

    // 5. 把上面做的关键帧与MP更换所属地图
    //TODO Time test
#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point timeStartCritMerge = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point timeStartCritMerge = std::chrono::monotonic_clock::now();
#endif
    // 与另一个融合函数相反，这个是将当前地图元素转移到待融合地图里面，然后删除当前地图
    {
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            if(!pKFi || pKFi->isBad())
            {
                //cout << "Bad KF in correction" << endl;
                continue;
            }
            // 保存了校正前的位姿，在优化里面要用到
            pKFi->mTcwBefMerge = pKFi->GetPose();
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
            pKFi->SetPose(pKFi->mTcwMerge);

            // Make sure connections are updated
            pKFi->UpdateMap(pMergeMap);
            pKFi->mnMergeCorrectedForKF = mpCurrentKF->mnId;  // 调试用的，目的在于区分与其他帧
            pMergeMap->AddKeyFrame(pKFi);
            pCurrentMap->EraseKeyFrame(pKFi);

            if(pCurrentMap->isImuInitialized())
            {
                pKFi->SetVelocity(pKFi->mVwbMerge);
            }
        }

        for(MapPoint* pMPi : spLocalWindowMPs)
        {
            if(!pMPi || pMPi->isBad())
                continue;

            pMPi->SetWorldPos(pMPi->mPosMerge);
            pMPi->SetNormalVector(pMPi->mNormalVectorMerge);
            pMPi->UpdateMap(pMergeMap);
            pMergeMap->AddMapPoint(pMPi);
            pCurrentMap->EraseMapPoint(pMPi);
            //pMPi->UpdateNormalAndDepth();
        }

        mpAtlas->ChangeMap(pMergeMap);
        mpAtlas->SetMapBad(pCurrentMap);
        pMergeMap->IncreaseChangeIndex();
        //TODO for debug
        pMergeMap->ChangeId(pCurrentMap->GetId());
    }

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point timeFinishCritMerge = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point timeFinishCritMerge = std::chrono::monotonic_clock::now();
#endif
    std::chrono::duration<double, std::milli> timeCritMerge = timeFinishCritMerge - timeStartCritMerge; // Time in milliseconds
    Verbose::PrintMess("MERGE-VISUAL: New current map: " + to_string(pMergeMap->GetId()), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("MERGE-VISUAL: CRITICAL ms: " + to_string(timeCritMerge.count()), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("MERGE-VISUAL: LOCAL MAPPING number of KFs: " + to_string(mpLocalMapper->KeyframesInQueue()), Verbose::VERBOSITY_DEBUG);

    // Rebuild the essential graph in the local window
    pCurrentMap->GetOriginKF()->SetFirstConnection(false);
    // 6. 下面这段代码主要是更新父子关系，与另一个容和函数同理
    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    pNewChild = mpCurrentKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpCurrentKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpCurrentKF->ChangeParent(mpMergeMatchedKF);
    while(pNewChild /*&& spLocalWindowKFs.find(pNewChild) != spLocalWindowKFs.end()*/)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();

        pNewChild->ChangeParent(pNewParent);
        //cout << "The new parent of KF " << pNewChild->mnId << " was " << pNewChild->GetParent()->mnId << endl;

        pNewParent = pNewChild;
        pNewChild = pOldParent;

    }


    // 7. 融合MP
    //Update the connections between the local window
    mpMergeMatchedKF->UpdateConnections();
    //cout << "MERGE-VISUAL: Essential graph rebuilded" << endl;

    //std::copy(spMapPointCurrent.begin(), spMapPointCurrent.end(), std::back_inserter(vpCheckFuseMapPoint));
    vector<KeyFrame*> vpMergeConnectedKFs;
    vpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    vpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));


    //TODO Time test
#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point timeStartFuseMerge = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point timeStartFuseMerge = std::chrono::monotonic_clock::now();
#endif

    // Project MapPoints observed in the neighborhood of the merge keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.

    // 当前地图的关键帧与待融合地图的mp
    // 当前关键帧中不重复的mp赋予待融合的mp
    // 如果冲突，替换成待融合地图的mp
    // vCorrectedSim3里面存放的帧与spLocalWindowKFs一样，另外还有校正后的sim3（有尺度）
    SearchAndFuse(vCorrectedSim3, vpCheckFuseMapPoint);

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point timeFinishFuseMerge = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point timeFinishFuseMerge = std::chrono::monotonic_clock::now();
#endif
    std::chrono::duration<double, std::milli> timeFuseMerge = timeFinishFuseMerge - timeStartFuseMerge; // Time in milliseconds
    Verbose::PrintMess("MERGE-VISUAL: FUSE DUPLICATED ms: " + to_string(timeFuseMerge.count()), Verbose::VERBOSITY_DEBUG);

    // 8. 更新连接关系
    Verbose::PrintMess("MERGE-VISUAL: Init to update connections in the welding area", Verbose::VERBOSITY_DEBUG);
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }

    //CheckObservations(spLocalWindowKFs, spMergeConnectedKFs);
    // 9. 局部优化
    Verbose::PrintMess("MERGE-VISUAL: Finish to update connections in the welding area", Verbose::VERBOSITY_DEBUG);

    bool bStop = false;
    Verbose::PrintMess("MERGE-VISUAL: Start local BA ", Verbose::VERBOSITY_DEBUG);
    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vpLocalCurrentWindowKFs.clear();
    vpMergeConnectedKFs.clear();
    std::copy(spLocalWindowKFs.begin(), spLocalWindowKFs.end(), std::back_inserter(vpLocalCurrentWindowKFs));
    std::copy(spMergeConnectedKFs.begin(), spMergeConnectedKFs.end(), std::back_inserter(vpMergeConnectedKFs));
    if (mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO)
    {
        Verbose::PrintMess("MERGE-VISUAL: Visual-Inertial", Verbose::VERBOSITY_DEBUG);
        Optimizer::MergeInertialBA(mpLocalMapper->GetCurrKF(), mpMergeMatchedKF, &bStop, mpCurrentKF->GetMap(), vCorrectedSim3);
    }
    else
    {
        Verbose::PrintMess("MERGE-VISUAL: Visual", Verbose::VERBOSITY_DEBUG);
        Verbose::PrintMess("MERGE-VISUAL: Local current window->" + to_string(vpLocalCurrentWindowKFs.size()) + "; Local merge window->" + to_string(vpMergeConnectedKFs.size()), Verbose::VERBOSITY_DEBUG);
        Optimizer::LocalBundleAdjustment(mpCurrentKF, vpLocalCurrentWindowKFs, vpMergeConnectedKFs, &bStop);
    }

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();


    //return;
    // 下面比另一个融合函数多
    Verbose::PrintMess("MERGE-VISUAL: Finish the LBA", Verbose::VERBOSITY_DEBUG);

    // 10. 处理当前地图中剩下的元素，更改位姿，更换所属地图
    ////
    //Update the non critical area from the current map to the merged map
    vector<KeyFrame*> vpCurrentMapKFs = pCurrentMap->GetAllKeyFrames();
    vector<MapPoint*> vpCurrentMapMPs = pCurrentMap->GetAllMapPoints();

    if(vpCurrentMapKFs.size() == 0)
    {
        Verbose::PrintMess("MERGE-VISUAL: There are not KFs outside of the welding area", Verbose::VERBOSITY_DEBUG);
    }
    else
    {
        Verbose::PrintMess("MERGE-VISUAL: Calculate the new position of the elements outside of the window", Verbose::VERBOSITY_DEBUG);
        //Apply the transformation
        // 处理pCurrentMap剩下的KFs与MP，更改了位姿，但是没有更换所属地图
        {
            if(mpTracker->mSensor == System::MONOCULAR)
            {
                unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information

                for(KeyFrame* pKFi : vpCurrentMapKFs)
                {
                    if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                    {
                        continue;
                    }

                    g2o::Sim3 g2oCorrectedSiw;

                    cv::Mat Tiw = pKFi->GetPose();
                    cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
                    cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
                    g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
                    //Pose without correction
                    vNonCorrectedSim3[pKFi]=g2oSiw;

                    cv::Mat Tic = Tiw*Twc;
                    cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
                    cv::Mat tic = Tic.rowRange(0, 3).col(3);
                    g2o::Sim3 g2oSim(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
                    g2oCorrectedSiw = g2oSim*mg2oMergeScw;
                    vCorrectedSim3[pKFi]=g2oCorrectedSiw;

                    // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                    Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
                    Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
                    double s = g2oCorrectedSiw.scale();

                    pKFi->mfScale = s;
                    eigt *=(1./s); //[R t/s;0 1]

                    cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

                    pKFi->mTcwBefMerge = pKFi->GetPose();
                    pKFi->mTwcBefMerge = pKFi->GetPoseInverse();

                    pKFi->SetPose(correctedTiw);

                    if(pCurrentMap->isImuInitialized())
                    {
                        Eigen::Matrix3d Rcor = eigR.transpose()*vNonCorrectedSim3[pKFi].rotation().toRotationMatrix();
                        pKFi->SetVelocity(Converter::toCvMat(Rcor)*pKFi->GetVelocity()); // TODO: should add here scale s
                    }

                }
                for(MapPoint* pMPi : vpCurrentMapMPs)
                {
                    if(!pMPi || pMPi->isBad()|| pMPi->GetMap() != pCurrentMap)
                        continue;

                    KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
                    g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
                    g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

                    // Project with non-corrected pose and project back with corrected pose
                    cv::Mat P3Dw = pMPi->GetWorldPos();
                    Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
                    Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(eigP3Dw));

                    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                    pMPi->SetWorldPos(cvCorrectedP3Dw);

                    pMPi->UpdateNormalAndDepth();
                }
            }
        }
        Verbose::PrintMess("MERGE-VISUAL: Apply transformation to all elements of the old map", Verbose::VERBOSITY_DEBUG);

        mpLocalMapper->RequestStop();
        // Wait until Local Mapping has effectively stopped
        while(!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }
        Verbose::PrintMess("MERGE-VISUAL: Local Map stopped", Verbose::VERBOSITY_DEBUG);

        // Optimize graph (and update the loop position for each element form the begining to the end)
        if(mpTracker->mSensor != System::MONOCULAR)
        {
            Optimizer::OptimizeEssentialGraph(mpCurrentKF, vpMergeConnectedKFs, vpLocalCurrentWindowKFs, vpCurrentMapKFs, vpCurrentMapMPs);
        }

        // 更换所属地图
        {
            // Get Merge Map Mutex
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
            unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

            Verbose::PrintMess("MERGE-VISUAL: There are " + to_string(pMergeMap->KeyFramesInMap()) + " KFs in the map", Verbose::VERBOSITY_DEBUG);
            Verbose::PrintMess("MERGE-VISUAL: It will be inserted " + to_string(vpCurrentMapKFs.size()) + " KFs in the map", Verbose::VERBOSITY_DEBUG);

            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }

                // Make sure connections are updated
                pKFi->UpdateMap(pMergeMap);
                pMergeMap->AddKeyFrame(pKFi);
                pCurrentMap->EraseKeyFrame(pKFi);
            }
            Verbose::PrintMess("MERGE-VISUAL: There are " + to_string(pMergeMap->MapPointsInMap()) + " MPs in the map", Verbose::VERBOSITY_DEBUG);
            Verbose::PrintMess("MERGE-VISUAL: It will be inserted " + to_string(vpCurrentMapMPs.size()) + " MPs in the map", Verbose::VERBOSITY_DEBUG);

            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad())
                    continue;

                pMPi->UpdateMap(pMergeMap);
                pMergeMap->AddMapPoint(pMPi);
                pCurrentMap->EraseMapPoint(pMPi);
            }
            Verbose::PrintMess("MERGE-VISUAL: There are " + to_string(pMergeMap->MapPointsInMap()) + " MPs in the map", Verbose::VERBOSITY_DEBUG);
        }

        Verbose::PrintMess("MERGE-VISUAL: Optimaze the essential graph", Verbose::VERBOSITY_DEBUG);
    }



    mpLocalMapper->Release();

    // 11. 全局优化，收尾
    Verbose::PrintMess("MERGE-VISUAL: Finally there are " + to_string(pMergeMap->KeyFramesInMap()) + "KFs and " + to_string(pMergeMap->MapPointsInMap()) + " MPs in the complete map ", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("MERGE-VISUAL:Completed!!!!!", Verbose::VERBOSITY_DEBUG);
    // 当前地图没有经过一阶段初始化或（当前地图关键帧数量小于200且地图数量只有一个）
    // TODO ps:地图都融合了咋能只有一个呢？后面需要找一个拼接地图的数据试试
    if(bRelaunchBA &&
       (!pCurrentMap->isImuInitialized() || (pCurrentMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1)))
    {
        // Launch a new thread to perform Global Bundle Adjustment
        Verbose::PrintMess("Relaunch Global BA", Verbose::VERBOSITY_DEBUG);
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, pMergeMap, mpCurrentKF->mnId);
    }

    mpMergeMatchedKF->AddMergeEdge(mpCurrentKF);
    mpCurrentKF->AddMergeEdge(mpMergeMatchedKF);  // 多余否？地图都要删了

    pCurrentMap->IncreaseChangeIndex();
    pMergeMap->IncreaseChangeIndex();

    mpAtlas->RemoveBadMaps();

}

/** 
 * @brief 调试用的，先不看
 */
void LoopClosing::CheckObservations(set<KeyFrame*> &spKFsMap1, set<KeyFrame*> &spKFsMap2)
{
    cout << "----------------------" << endl;
    for(KeyFrame* pKFi1 : spKFsMap1)
    {
        map<KeyFrame*, int> mMatchedMP;
        set<MapPoint*> spMPs = pKFi1->GetMapPoints();

        for(MapPoint* pMPij : spMPs)
        {
            if(!pMPij || pMPij->isBad())
            {
                continue;
            }

            map<KeyFrame*, tuple<int, int>> mMPijObs = pMPij->GetObservations();
            for(KeyFrame* pKFi2 : spKFsMap2)
            {
                if(mMPijObs.find(pKFi2) != mMPijObs.end())
                {
                    if(mMatchedMP.find(pKFi2) != mMatchedMP.end())
                    {
                        mMatchedMP[pKFi2] = mMatchedMP[pKFi2] + 1;
                    }
                    else
                    {
                        mMatchedMP[pKFi2] = 1;
                    }
                }
            }

        }

        if(mMatchedMP.size() == 0)
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has not any matched MP with the other map" << endl;
        }
        else
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has matched MP with " << mMatchedMP.size() << " KF from the other map" << endl;
            for(pair<KeyFrame*, int> matchedKF : mMatchedMP)
            {
                cout << "   -KF: " << matchedKF.first->mnId << ", Number of matches: " << matchedKF.second << endl;
            }
        }
    }
    cout << "----------------------" << endl;
}

/** 
 * @brief 功能与同名函数一样，只是输入不同，回环矫正里面用到
 */
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;  // 计数用，看看一共融合了多少点

    cout << "FUSE: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    cout << "FUSE: Intially there are " << CorrectedPosesMap.size() << " KFs" << endl;
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKFi = mit->first;
        Map* pMap = pKFi->GetMap();

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(), static_cast<MapPoint*>(NULL));
        // 新点表示pKFi对应的点，老点表示pKFi对应的回环点
        // 将vpMapPoints投到pKF里面看看有没有匹配的MP，如果没有直接添加，如果有，暂时将老点放入至vpReplacePoints
        // vpReplacePoints下标表示第n个vpMapPoints，存放着新点，可以直接找到对应信息
        int numFused = matcher.Fuse(pKFi, cvScw, vpMapPoints, 4, vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        // 更新点
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            // vpReplacePoints如果存在新点，则替换成老点，这里注意如果老点已经在新点对应的kf中
            // 也就是之前某次matcher.Fuse 把老点放入到新的关键帧中，下次遍历时，如果老点已经在被代替点的对应的某一个关键帧内
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                num_replaces += 1;
                // 替换掉较新的
                pRep->Replace(vpMapPoints[i]);
            }
        }
        total_replaces += num_replaces;
    }
    cout << "FUSE: " << total_replaces << " MPs had been fused" << endl;
}

/** 
 * @brief 查找对应MP与融合
 * @param vConectedKFs 当前地图的当前关键帧及5个共视关键帧
 * @param vpMapPoints 待融合地图的融合帧及其5个共视关键帧对应的mp（1000个以内）（注意此时所有kf与mp全部移至当前地图，这里的待融合地图的说法只为区分，因为还没有融合）
 */
void LoopClosing::SearchAndFuse(const vector<KeyFrame*> &vConectedKFs, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;  // 计数用，看看一共融合了多少点

    cout << "FUSE-POSE: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    cout << "FUSE-POSE: Intially there are " << vConectedKFs.size() << " KFs" << endl;
    for(auto mit=vConectedKFs.begin(), mend=vConectedKFs.end(); mit!=mend; mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKF = (*mit);
        Map* pMap = pKF->GetMap();
        cv::Mat cvScw = pKF->GetPose();

        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(), static_cast<MapPoint*>(NULL));
        // 将vpMapPoints投到pKF里面看看有没有匹配的MP，如果没有直接添加，如果有，暂时将老点放入至vpReplacePoints
        // vpReplacePoints下标表示第n个vpMapPoints，存放着新点，可以直接找到对应信息
        matcher.Fuse(pKF, cvScw, vpMapPoints, 4, vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        // 更新点
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            // vpReplacePoints如果存在新点，则替换成老点，这里注意如果老点已经在新点对应的kf中
            // 也就是之前某次matcher.Fuse 把老点放入到新的关键帧中，下次遍历时，如果老点已经在被代替点的对应的某一个关键帧内
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                num_replaces += 1;
                // 替换掉较新的
                pRep->Replace(vpMapPoints[i]);
            }
        }
        cout << "FUSE-POSE: KF " << pKF->mnId << " ->" << num_replaces << " MPs fused" << endl;
        total_replaces += num_replaces;
    }
    cout << "FUSE-POSE: " << total_replaces << " MPs had been fused" << endl;
}



void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(5000);
    }
}

void LoopClosing::RequestResetActiveMap(Map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset = pMap;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetActiveMapRequested)
                break;
        }
        usleep(3000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        cout << "Loop closer reset requested..." << endl;
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;  //TODO old variable, it is not use in the new algorithm
        mbResetRequested=false;
        mbResetActiveMapRequested = false;
    }
    else if(mbResetActiveMapRequested)
    {

        for (list<KeyFrame*>::const_iterator it=mlpLoopKeyFrameQueue.begin(); it != mlpLoopKeyFrameQueue.end();)
        {
            KeyFrame* pKFi = *it;
            if(pKFi->GetMap() == mpMapToReset)
            {
                it = mlpLoopKeyFrameQueue.erase(it);
            }
            else
                ++it;
        }

        mLastLoopKFid=mpAtlas->GetLastInitKFid(); //TODO old variable, it is not use in the new algorithm
        mbResetActiveMapRequested=false;

    }
}

/** 
 * @brief MergeLocal CorrectLoop 中调用
 * @param pActiveMap 当前地图
 * @param nLoopKF 检测到回环成功的关键帧，不是与之匹配的老关键帧
 */
void LoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF)
{
    Verbose::PrintMess("Starting Global Bundle Adjustment", Verbose::VERBOSITY_NORMAL);
    // imu 初始化成功才返回true，只要一阶段成功就为true
    const bool bImuInit = pActiveMap->isImuInitialized();

    if(!bImuInit)
        Optimizer::GlobalBundleAdjustemnt(pActiveMap, 10, &mbStopGBA, nLoopKF, false);
    else
        Optimizer::FullInertialBA(pActiveMap, 7, false, nLoopKF, &mbStopGBA);


    int idx =  mnFullBAIdx;
    // Optimizer::GlobalBundleAdjustemnt(mpMap, 10, &mbStopGBA, nLoopKF, false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx != mnFullBAIdx)
            return;

        if(!bImuInit && pActiveMap->isImuInitialized())
            return;

        if(!mbStopGBA)
        {
            Verbose::PrintMess("Global Bundle Adjustment finished", Verbose::VERBOSITY_NORMAL);
            Verbose::PrintMess("Updating map ...", Verbose::VERBOSITY_NORMAL);

            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(pActiveMap->mMutexMapUpdate);
            // cout << "LC: Update Map Mutex adquired" << endl;

            //pActiveMap->PrintEssentialGraph();
            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(pActiveMap->mvpKeyFrameOrigins.begin(), pActiveMap->mvpKeyFrameOrigins.end());
            // 通过树的方式更新未参与全局优化的关键帧，一个关键帧与其父节点的共视点数最多，所以选其作为参考帧
            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                //cout << "---Updating KF " << pKF->mnId << " with " << sChilds.size() << " childs" << endl;
                //cout << " KF mnBAGlobalForKF: " << pKF->mnBAGlobalForKF << endl;
                cv::Mat Twc = pKF->GetPoseInverse();
                //cout << "Twc: " << Twc << endl;
                //cout << "GBA: Correct KeyFrames" << endl;
                // 广度优先搜索
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(!pChild || pChild->isBad())
                        continue;
                    // 专门处理没有参与优化的新关键帧
                    if(pChild->mnBAGlobalForKF != nLoopKF)
                    {
                        //cout << "++++New child with flag " << pChild->mnBAGlobalForKF << "; LoopKF: " << nLoopKF << endl;
                        //cout << " child id: " << pChild->mnId << endl;
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        //cout << "Child pose: " << Tchildc << endl;
                        //cout << "pKF->mTcwGBA: " << pKF->mTcwGBA << endl;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;

                        cv::Mat Rcor = pChild->mTcwGBA.rowRange(0, 3).colRange(0, 3).t()*pChild->GetRotation();
                        if(!pChild->GetVelocity().empty())
                        {
                            //cout << "Child velocity: " << pChild->GetVelocity() << endl;
                            pChild->mVwbGBA = Rcor*pChild->GetVelocity();
                        }
                        else
                            Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);


                        //cout << "Child bias: " << pChild->GetImuBias() << endl;
                        pChild->mBiasGBA = pChild->GetImuBias();


                        pChild->mnBAGlobalForKF = nLoopKF;  // 标记成更新过的

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                //cout << "-------Update pose" << endl;
                pKF->mTcwBefGBA = pKF->GetPose();
                //cout << "pKF->mTcwBefGBA: " << pKF->mTcwBefGBA << endl;
                pKF->SetPose(pKF->mTcwGBA);
                /*cv::Mat Tco_cn = pKF->mTcwBefGBA * pKF->mTcwGBA.inv();
                cv::Vec3d trasl = Tco_cn.rowRange(0, 3).col(3);
                double dist = cv::norm(trasl);
                cout << "GBA: KF " << pKF->mnId << " had been moved " << dist << " meters" << endl;
                double desvX = 0;
                double desvY = 0;
                double desvZ = 0;
                if(pKF->mbHasHessian)
                {
                    cv::Mat hessianInv = pKF->mHessianPose.inv();

                    double covX = hessianInv.at<double>(3, 3);
                    desvX = std::sqrt(covX);
                    double covY = hessianInv.at<double>(4, 4);
                    desvY = std::sqrt(covY);
                    double covZ = hessianInv.at<double>(5, 5);
                    desvZ = std::sqrt(covZ);
                    pKF->mbHasHessian = false;
                }
                if(dist > 1)
                {
                    cout << "--To much distance correction: It has " << pKF->GetConnectedKeyFrames().size() << " connected KFs" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(80).size() << " connected KF with 80 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(50).size() << " connected KF with 50 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(20).size() << " connected KF with 20 common matches or more" << endl;

                    cout << "--STD in meters(x, y, z): " << desvX << ", " << desvY << ", " << desvZ << endl;


                    string strNameFile = pKF->mNameFile;
                    cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

                    cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

                    vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
                    int num_MPs = 0;
                    for(int i=0; i<vpMapPointsKF.size(); ++i)
                    {
                        if(!vpMapPointsKF[i] || vpMapPointsKF[i]->isBad())
                        {
                            continue;
                        }
                        num_MPs += 1;
                        string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                        cv::circle(imLeft, pKF->mvKeys[i].pt, 2, cv::Scalar(0, 255, 0));
                        cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    }
                    cout << "--It has " << num_MPs << " MPs matched in the map" << endl;

                    string namefile = "./test_GBA/GBA_" + to_string(nLoopKF) + "_KF" + to_string(pKF->mnId) +"_D" + to_string(dist) +".png";
                    cv::imwrite(namefile, imLeft);
                }*/


                if(pKF->bImu)
                {
                    //cout << "-------Update inertial values" << endl;
                    pKF->mVwbBefGBA = pKF->GetVelocity();
                    if (pKF->mVwbGBA.empty())
                        Verbose::PrintMess("pKF->mVwbGBA is empty", Verbose::VERBOSITY_NORMAL);

                    assert(!pKF->mVwbGBA.empty());
                    pKF->SetVelocity(pKF->mVwbGBA);
                    pKF->SetNewBias(pKF->mBiasGBA);                    
                }

                lpKFtoCheck.pop_front();
            }

            //cout << "GBA: Correct MapPoints" << endl;
            // Correct MapPoints
            // 更新mp点
            const vector<MapPoint*> vpMPs = pActiveMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;
                // 参与全局BA的点
                if(pMP->mnBAGlobalForKF == nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // 没参与全局优化，通过更新过位姿的关键帧来更新坐标
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF != nLoopKF)
                        continue;

                    if(pRefKF->mTcwBefGBA.empty())
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
                    cv::Mat twc = Twc.rowRange(0, 3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }

            pActiveMap->InformNewBigChange();
            pActiveMap->IncreaseChangeIndex();

            // TODO Check this update
            // mpTracker->UpdateFrameIMU(1.0f, mpTracker->GetLastKeyFrame()->GetImuBias(), mpTracker->GetLastKeyFrame());

            mpLocalMapper->Release();

            Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    // cout << "LC: Finish requested" << endl;
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
