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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include <mutex>

namespace ORB_SLAM3
{

long unsigned int MapPoint::nNextId = 0;
mutex MapPoint::mGlobalMutex;

/** 
 * @brief 构造函数
 */
MapPoint::MapPoint() : mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
                        mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
                        mnCorrectedReference(0), mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
                        mpReplaced(static_cast<MapPoint *>(NULL))
{
    mpReplaced = static_cast<MapPoint *>(NULL);
}

/** 
 * @brief 构造函数
 */
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap) : 
                    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
                    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
                    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
                    mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
                    mnOriginMapId(pMap->GetId())
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

    mbTrackInViewR = false;
    mbTrackInView = false;

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

/** 
 * @brief 构造函数
 */
MapPoint::MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame *pRefKF, KeyFrame *pHostKF, Map *pMap) : 
                    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
                    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
                    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
                    mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
                    mnOriginMapId(pMap->GetId())
{
    mInvDepth = invDepth;
    mInitU = (double)uv_init.x;
    mInitV = (double)uv_init.y;
    mpHostKF = pHostKF;

    mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

    // Worldpos is not set
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

/** 
 * @brief 构造函数
 */
MapPoint::MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF) : 
                mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
                mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
                mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame *>(NULL)), mnVisible(1),
                mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap), mnOriginMapId(pMap->GetId())
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow;
    if (pFrame->Nleft == -1 || idxF < pFrame->Nleft)
    {
        Ow = pFrame->GetCameraCenter();
    }
    else
    {
        cv::Mat Rwl = pFrame->mRwc;
        cv::Mat tlr = pFrame->mTlr.col(3);
        cv::Mat twl = pFrame->mOw;

        Ow = Rwl * tlr + twl;
    }
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = (pFrame->Nleft == -1) ? pFrame->mvKeysUn[idxF].octave
                                            : (idxF < pFrame->Nleft) ? pFrame->mvKeys[idxF].octave
                                                                        : pFrame->mvKeysRight[idxF].octave;
    const float levelScaleFactor = pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

/**
 * @brief 设定该mp的世界坐标
 * @param Pos         坐标值
 */
void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

/**
 * @brief 返回该mp的世界坐标
 */
cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

/**
 * @brief 获取平均观测方向
 */
cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

/**
 * @brief 获取它的参考帧
 */
KeyFrame *MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

/**
 * @brief 添加观测，向该点添加可以观测到其的关键帧及对应的特征点
 * @param pKF         关键帧
 * @param idx         特征点在关键帧的id
 */
void MapPoint::AddObservation(KeyFrame *pKF, int idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    tuple<int, int> indexes;

    if (mObservations.count(pKF))
    {
        indexes = mObservations[pKF];
    }
    else
    {
        indexes = tuple<int, int>(-1, -1);
    }

    if (pKF->NLeft != -1 && idx >= pKF->NLeft)
    {
        get<1>(indexes) = idx;
    }
    else
    {
        get<0>(indexes) = idx;
    }

    mObservations[pKF] = indexes;

    if (!pKF->mpCamera2 && pKF->mvuRight[idx] >= 0)
        nObs += 2;
    else
        nObs++;
}

/**
 * @brief 删除观测
 * @param pKF         关键帧
 */
void MapPoint::EraseObservation(KeyFrame *pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
        {
            //int idx = mObservations[pKF];
            tuple<int, int> indexes = mObservations[pKF];
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if (leftIndex != -1)
            {
                if (!pKF->mpCamera2 && pKF->mvuRight[leftIndex] >= 0)
                    nObs -= 2;
                else
                    nObs--;
            }
            if (rightIndex != -1)
            {
                nObs--;
            }

            mObservations.erase(pKF);

            if (mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if (nObs <= 2)
                bBad = true;
        }
    }

    if (bBad)
        SetBadFlag();
}

/**
 * @brief 返回观测数据结构
 * @return mObservations
 */
std::map<KeyFrame *, std::tuple<int, int>> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

/**
 * @brief 返回被观测次数，双目一帧算两次，左右目各算各的
 * @return nObs
 */
int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

/**
 * @brief 设定该MP坏了，后面优化什么的就不要了
 */
void MapPoint::SetBadFlag()
{
    map<KeyFrame *, tuple<int, int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
    for (map<KeyFrame *, tuple<int, int>>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;
        int leftIndex = get<0>(mit->second), rightIndex = get<1>(mit->second);
        if (leftIndex != -1)
        {
            pKF->EraseMapPointMatch(leftIndex);
        }
        if (rightIndex != -1)
        {
            pKF->EraseMapPointMatch(rightIndex);
        }
    }

    mpMap->EraseMapPoint(this);
}

/**
 * @brief 判断该点是否已经被替换，因为替换并没有考虑普通帧的替换，不利于下一帧的跟踪，所以要坐下标记
 * @return 替换的新的点
 */
MapPoint *MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

/** 
 * @brief 将该点替换成pMP，并不是直接把该点赋值成新的点，而是干掉this点，将里面的信息保存至pMP
 * @param pMP 替换成的mp
 */
void MapPoint::Replace(MapPoint *pMP)
{
    if (pMP->mnId == this->mnId)
        return;

    int nvisible, nfound;
    // 1. 保存原始点的信息
    map<KeyFrame *, tuple<int, int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs = mObservations;
        mObservations.clear();
        mbBad = true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }
    // 2. 遍历旧点的观测信息
    for (map<KeyFrame *, tuple<int, int>>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame *pKF = mit->first;

        tuple<int, int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
        // 2.1 判断新点是否已经在pKF里面
        if (!pMP->IsInKeyFrame(pKF))
        {
            // 如果不在，替换特征点与mp的匹配关系
            if (leftIndex != -1)
            {
                pKF->ReplaceMapPointMatch(leftIndex, pMP);
                pMP->AddObservation(pKF, leftIndex);
            }
            if (rightIndex != -1)
            {
                pKF->ReplaceMapPointMatch(rightIndex, pMP);
                pMP->AddObservation(pKF, rightIndex);
            }
        }
        // 如果新的MP在之前MP对应的关键帧里面，就撞车了。
        // 本来目的想新旧MP融为一个，这样以来一个点有可能对应两个特征点，这样是决不允许的，所以删除旧的，不动新的
        else
        {
            
            if (leftIndex != -1)
            {
                pKF->EraseMapPointMatch(leftIndex);
            }
            if (rightIndex != -1)
            {
                pKF->EraseMapPointMatch(rightIndex);
            }
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

/**
 * @brief 判断这个点是否设置为bad了
 */
bool MapPoint::isBad()
{
    unique_lock<mutex> lock1(mMutexFeatures, std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos, std::defer_lock);
    lock(lock1, lock2);

    return mbBad;
}

/**
 * @brief Increase Visible
 *
 * Visible表示：
 * 1. 该MapPoint在某些帧的视野范围内，通过Frame::isInFrustum()函数判断
 * 2. 该MapPoint被这些帧观测到，但并不一定能和这些帧的特征点匹配上
 *    例如：有一个MapPoint（记为M），在某一帧F的视野范围内，
 *    但并不表明该点M可以和F这一帧的某个特征点能匹配上
 */
void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}

/**
 * @brief Increase Found
 *
 * 能找到该点的帧数+n，n默认为1
 * @see Tracking::TrackLocalMap()
 */
void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}

/**
 * @brief 返回被找到/被看到
 * @return 被找到/被看到
 */
float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / mnVisible;
}

/**
 * @brief 计算具有代表的描述子
 *
 * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前点的最适合的描述子 \n
 * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
 * @see III - C3.3
 */
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame *, tuple<int, int>> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if (mbBad)
            return;
        observations = mObservations;
    }

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());
    // 遍历观测到3d点的所有关键帧，获得orb描述子，并插入到vDescriptors中
    for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;

        if (!pKF->isBad())
        {
            tuple<int, int> indexes = mit->second;
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if (leftIndex != -1)
            {
                vDescriptors.push_back(pKF->mDescriptors.row(leftIndex));
            }
            if (rightIndex != -1)
            {
                vDescriptors.push_back(pKF->mDescriptors.row(rightIndex));
            }
        }
    }

    if (vDescriptors.empty())
        return;
    // 获得这些描述子两两之间的距离
    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for (size_t i = 0; i < N; i++)
    {

        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for (size_t i = 0; i < N; i++)
    {
        // 第i个描述子到其它所有所有描述子之间的距离
        vector<int> vDists(Distances[i], Distances[i] + N);
        sort(vDists.begin(), vDists.end());
        // 获得中值
        int median = vDists[0.5 * (N - 1)];

        if (median < BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        // 最好的描述子，该描述子相对于其他描述子有最小的距离中值
        // 简化来讲，中值代表了这个描述子到其它描述子的平均距离
        // 最好的描述子就是和其它描述子的平均距离最小
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

/**
 * @brief 返回描述子
 */
cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

/**
 * @brief 返回这个点在关键帧中对应的特征点id
 */
tuple<int, int> MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return tuple<int, int>(-1, -1);
}

/**
 * @brief return (mObservations.count(pKF));
 */
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

/**
 * @brief 更新平均观测方向以及观测距离范围
 *
 * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要更新相应变量
 * @see III - C2.2 c2.4
 * 全局优化后使用
 */
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame *, tuple<int, int>> observations;
    KeyFrame *pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if (mbBad)
            return;
        observations = mObservations; // 获得观测到该3d点的所有关键帧
        pRefKF = mpRefKF;             // 观测到该点的参考关键帧
        Pos = mWorldPos.clone();      // 3d点在世界坐标系中的位置
    }

    if (observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
    int n = 0;
    for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;

        tuple<int, int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if (leftIndex != -1)
        {
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali / cv::norm(normali); // 对所有关键帧对该点的观测方向归一化为单位向量进行求和
            n++;
        }
        if (rightIndex != -1)
        {
            cv::Mat Owi = pKF->GetRightCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali / cv::norm(normali);
            n++;
        }
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter(); // 参考关键帧相机指向3D点的向量（在世界坐标系下的表示）
    const float dist = cv::norm(PC);              // 该点到参考关键帧相机的距离

    tuple<int, int> indexes = observations[pRefKF];
    int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    int level;
    if (pRefKF->NLeft == -1)
    {
        level = pRefKF->mvKeysUn[leftIndex].octave;
    }
    else if (leftIndex != -1)
    {
        level = pRefKF->mvKeys[leftIndex].octave;
    }
    else
    {
        level = pRefKF->mvKeysRight[rightIndex - pRefKF->NLeft].octave;
    }

    //const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels; // 金字塔层数

    {
        unique_lock<mutex> lock3(mMutexPos);
        // 另见PredictScale函数前的注释
        mfMaxDistance = dist * levelScaleFactor;                             // 观测到该点的距离下限
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1]; // 观测到该点的距离上限
        mNormalVector = normal / n;                                          // 获得平均的观测方向
    }
}

/**
 * @brief 设置平均观测方向
 * @param normal         观测方向
 */
void MapPoint::SetNormalVector(cv::Mat &normal)
{
    unique_lock<mutex> lock3(mMutexPos);
    mNormalVector = normal;
}

/**
 * @brief 返回最近距离
 */
float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}

/**
 * @brief 返回最远距离
 */
float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}

//                        ____
// Nearer               /_____\     level:n-1 --> dmin
//                    /________\                       d/dmin = 1.2^(n-1-m)
//                  /___________\   level:m   --> d
//                /______________\                     dmax/d = 1.2^m
// Farther      /_________________\ level:0   --> dmax
//
//           log(dmax/d)
// m = ceil(------------)
//            log(1.2)
// dmax/d = 1.2^m
// 注意，同一个点在上面金字塔层数越高时表示越近，越低表示越远，计算最远与当前距离的比例，用这个比例来计算中间跨了多少层
int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels - 1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame *pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pF->mnScaleLevels)
        nScale = pF->mnScaleLevels - 1;

    return nScale;
}

/**
 * @brief 打印数值
 */
void MapPoint::PrintObservations()
{
    cout << "MP_OBS: MP " << mnId << endl;
    for (map<KeyFrame *, tuple<int, int>>::iterator mit = mObservations.begin(), mend = mObservations.end(); mit != mend; mit++)
    {
        KeyFrame *pKFi = mit->first;
        tuple<int, int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
        cout << "--OBS in KF " << pKFi->mnId << " in map " << pKFi->GetMap()->GetId() << endl;
    }
}

/**
 * @brief 获取这个点对应的地图
 */
Map *MapPoint::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

/**
 * @brief 更换对应的地图
 */
void MapPoint::UpdateMap(Map *pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void MapPoint::PreSave(set<KeyFrame *> &spKF, set<MapPoint *> &spMP)
{
    mBackupReplacedId = -1;
    if (mpReplaced && spMP.find(mpReplaced) != spMP.end())
        mBackupReplacedId = mpReplaced->mnId;

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
    // Save the id and position in each KF who view it
    for (std::map<KeyFrame *, std::tuple<int, int>>::const_iterator it = mObservations.begin(), end = mObservations.end(); it != end; ++it)
    {
        KeyFrame *pKFi = it->first;
        if (spKF.find(pKFi) != spKF.end())
        {
            mBackupObservationsId1[it->first->mnId] = get<0>(it->second);
            mBackupObservationsId2[it->first->mnId] = get<1>(it->second);
        }
        else
        {
            EraseObservation(pKFi);
        }
    }

    // Save the id of the reference KF
    if (spKF.find(mpRefKF) != spKF.end())
    {
        mBackupRefKFId = mpRefKF->mnId;
    }
}

void MapPoint::PostLoad(map<long unsigned int, KeyFrame *> &mpKFid, map<long unsigned int, MapPoint *> &mpMPid)
{
    mpRefKF = mpKFid[mBackupRefKFId];
    if (!mpRefKF)
    {
        cout << "MP without KF reference " << mBackupRefKFId << "; Num obs: " << nObs << endl;
    }
    mpReplaced = static_cast<MapPoint *>(NULL);
    if (mBackupReplacedId >= 0)
    {
        map<long unsigned int, MapPoint *>::iterator it = mpMPid.find(mBackupReplacedId);
        if (it != mpMPid.end())
            mpReplaced = it->second;
    }

    mObservations.clear();

    for (map<long unsigned int, int>::const_iterator it = mBackupObservationsId1.begin(), end = mBackupObservationsId1.end(); it != end; ++it)
    {
        KeyFrame *pKFi = mpKFid[it->first];
        map<long unsigned int, int>::const_iterator it2 = mBackupObservationsId2.find(it->first);
        std::tuple<int, int> indexes = tuple<int, int>(it->second, it2->second);
        if (pKFi)
        {
            mObservations[pKFi] = indexes;
        }
    }

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
}

} // namespace ORB_SLAM3
