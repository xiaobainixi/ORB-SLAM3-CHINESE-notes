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

#include "ORBmatcher.h"

#include <limits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include <stdint-gcc.h>

using namespace std;

namespace ORB_SLAM3
{

const int ORBmatcher::TH_HIGH = 100;     // 100
const int ORBmatcher::TH_LOW = 50;       // 50
const int ORBmatcher::HISTO_LENGTH = 30; // 30

ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

/** 
 * @brief Tracking::SearchLocalPoints() 调用，通过投影的方式找与当前帧的匹配点
 * @param F 当前帧
 * @param vpMapPoints 局部地图的MP
 * @param th 搜索半径
 * @param bFarPoints 是否做远点区分
 * @param thFarPoints 判断远点的阈值
 */
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint *> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
{
    int nmatches = 0, left = 0, right = 0;

    const bool bFactor = th != 1.0;

    for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
    {
        MapPoint *pMP = vpMapPoints[iMP];
        // 如果已经匹配就没必要在寻找匹配点了
        if (!pMP->mbTrackInView && !pMP->mbTrackInViewR)
            continue;

        // 深度检测
        if (bFarPoints && pMP->mTrackDepth > thFarPoints)
            continue;

        if (pMP->isBad())
            continue;
        // mbTrackInView为true时表示这个MP在这个关键帧的视野内，且不为已经匹配好的点
        if (pMP->mbTrackInView)
        {
            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            // 确定搜索范围的倍率，夹角小于一定值搜索范围就很小
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            if (bFactor)
                r *= th;

            const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

            if (!vIndices.empty())
            {
                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist = 256;
                int bestLevel = -1;
                int bestDist2 = 256;
                int bestLevel2 = -1;
                int bestIdx = -1;

                // Get best and second matches with near keypoints
                for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
                {
                    const size_t idx = *vit;

                    if (F.mvpMapPoints[idx])
                        if (F.mvpMapPoints[idx]->Observations() > 0)
                            continue;

                    if (F.Nleft == -1 && F.mvuRight[idx] > 0)
                    {
                        const float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
                        if (er > r * F.mvScaleFactors[nPredictedLevel])
                            continue;
                    }

                    const cv::Mat &d = F.mDescriptors.row(idx);

                    const int dist = DescriptorDistance(MPdescriptor, d);

                    if (dist < bestDist)
                    {
                        bestDist2 = bestDist;
                        bestDist = dist;
                        bestLevel2 = bestLevel;
                        bestLevel = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                    : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                        : F.mvKeysRight[idx - F.Nleft].octave;
                        bestIdx = idx;
                    }
                    else if (dist < bestDist2)
                    {
                        bestLevel2 = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                        : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                        : F.mvKeysRight[idx - F.Nleft].octave;
                        bestDist2 = dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if (bestDist <= TH_HIGH)
                {
                    if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                        continue;

                    if (bestLevel != bestLevel2 || bestDist <= mfNNratio * bestDist2)
                    {
                        F.mvpMapPoints[bestIdx] = pMP;

                        if (F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1)
                        { //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                            nmatches++;
                            right++;
                        }

                        nmatches++;
                        left++;
                    }
                }
            }
        }

        if (F.Nleft != -1 && pMP->mbTrackInViewR)
        {
            const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
            if (nPredictedLevel != -1)
            {
                float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                const vector<size_t> vIndices =
                    F.GetFeaturesInArea(pMP->mTrackProjXR, pMP->mTrackProjYR, r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel, true);

                if (vIndices.empty())
                    continue;

                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist = 256;
                int bestLevel = -1;
                int bestDist2 = 256;
                int bestLevel2 = -1;
                int bestIdx = -1;

                // Get best and second matches with near keypoints
                for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
                {
                    const size_t idx = *vit;

                    if (F.mvpMapPoints[idx + F.Nleft])
                        if (F.mvpMapPoints[idx + F.Nleft]->Observations() > 0)
                            continue;

                    const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                    const int dist = DescriptorDistance(MPdescriptor, d);

                    if (dist < bestDist)
                    {
                        bestDist2 = bestDist;
                        bestDist = dist;
                        bestLevel2 = bestLevel;
                        bestLevel = F.mvKeysRight[idx].octave;
                        bestIdx = idx;
                    }
                    else if (dist < bestDist2)
                    {
                        bestLevel2 = F.mvKeysRight[idx].octave;
                        bestDist2 = dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if (bestDist <= TH_HIGH)
                {
                    if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                        continue;

                    if (F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1)
                    { //Also match with the stereo observation at right camera
                        F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                        nmatches++;
                        left++;
                    }

                    F.mvpMapPoints[bestIdx + F.Nleft] = pMP;
                    nmatches++;
                    right++;
                }
            }
        }
    }
    return nmatches;
}

/** 
 * @brief 确定搜索范围的倍率，夹角小于一定值搜索范围就很小
 * @param viewCos 这个点到相机光心与平均观测方向的夹角
 */
float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if (viewCos > 0.998)
        return 2.5;
    else
        return 4.0;
}

/** 
 * @brief 没用到，暂时不看
 */
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2, const bool b1)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
    const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
    const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

    const float num = a * kp2.pt.x + b * kp2.pt.y + c;

    const float den = a * a + b * b;

    if (den == 0)
        return false;

    const float dsqr = num * num / den;

    if (!b1)
        return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];
    else
        return dsqr < 6.63 * pKF2->mvLevelSigma2[kp2.octave];
}

/** 
 * @brief 没用到，暂时不看
 */
bool ORBmatcher::CheckDistEpipolarLine2(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2, const float unc)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
    const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
    const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

    const float num = a * kp2.pt.x + b * kp2.pt.y + c;

    const float den = a * a + b * b;

    if (den == 0)
        return false;

    const float dsqr = num * num / den;

    if (unc == 1.f)
        return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];
    else
        return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave] * unc;
}

/** 
 * @brief 通过BoW查找两个关键帧共有的MP，Tracking::TrackReferenceKeyFrame()与Tracking::Relocalization()中有使用
 * @param pKF 参考关键帧
 * @param F 当前帧
 * @param vpMapPointMatches 当前帧匹配上的MP
 */
int ORBmatcher::SearchByBoW(KeyFrame *pKF, Frame &F, vector<MapPoint *> &vpMapPointMatches)
{
    // 取出KF中每个特征点所对应的MP其中有的可能为空
    const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches = 0;

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while (KFit != KFend && Fit != Fend)
    {
        // 找到同一词汇所对应的集合
        if (KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;
            // 遍历关键帧在这个集合中所有MP
            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint *pMP = vpMapPointsKF[realIdxKF];

                if (!pMP)
                    continue;

                if (pMP->isBad())
                    continue;
                // 取出对应的描述子
                const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

                int bestDist1 = 256;
                int bestIdxF = -1;
                int bestDist2 = 256;

                int bestDist1R = 256;
                int bestIdxFR = -1;
                int bestDist2R = 256;
                // 遍历基础帧中每个特征点，找到与MP匹配的最优点与次优点
                for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                {
                    // 单目使用的是F.N 如果F.Nleft这个值是-1表示此时是单目
                    if (F.Nleft == -1)
                    {
                        const unsigned int realIdxF = vIndicesF[iF];
                        // 如果已经匹配过了，跳过
                        if (vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        const int dist = DescriptorDistance(dKF, dF);

                        if (dist < bestDist1)
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdxF = realIdxF;
                        }
                        else if (dist < bestDist2)
                        {
                            bestDist2 = dist;
                        }
                    }
                    else // 左右目时
                    {
                        const unsigned int realIdxF = vIndicesF[iF];
                        // 如果已经匹配过了，跳过
                        if (vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        const int dist = DescriptorDistance(dKF, dF);

                        if (realIdxF < F.Nleft && dist < bestDist1)
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdxF = realIdxF;
                        }
                        else if (realIdxF < F.Nleft && dist < bestDist2)
                        {
                            bestDist2 = dist;
                        }

                        if (realIdxF >= F.Nleft && dist < bestDist1R)
                        {
                            bestDist2R = bestDist1R;
                            bestDist1R = dist;
                            bestIdxFR = realIdxF;
                        }
                        else if (realIdxF >= F.Nleft && dist < bestDist2R)
                        {
                            bestDist2R = dist;
                        }
                    }
                }
                // TODO 左右目模式时，假设右目的特征点匹配上了，但左目没有匹配上，跳过，以左目为准
                // 这里可以改成同优先级的，不过也可能有加速考虑，因为左右目离的很近，左目都没看见看右目匹配也费劲
                // 想提高跟踪成功率可以改一下
                if (bestDist1 <= TH_LOW)
                {
                    // 与初始化时一样，最小距离要比次小距离的倍数还要少，这个倍数小于1,到这里时有可能是双目，有可能是单目，也有可能KF左目某点与F右目某点找到对应关系
                    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                    {
                        // 确定匹配关系
                        vpMapPointMatches[bestIdxF] = pMP;
                        // 单双目特征点存放的变量不同，需要判断，从KF中取出对应的KP
                        const cv::KeyPoint &kp =
                            (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :                           // 相机2不存在时
                                (realIdxKF >= pKF->NLeft) ? pKF->mvKeysRight[realIdxKF - pKF->NLeft] // 相机2存在且realIdxKF属于右目
                                                            : pKF->mvKeys[realIdxKF];                  // 相机2存在且realIdxKF属于左目
                        // 取出在帧F中对应的特征点
                        if (mbCheckOrientation)
                        {
                            cv::KeyPoint &Fkp =
                                (!pKF->mpCamera2 || F.Nleft == -1) ? F.mvKeys[bestIdxF] : (bestIdxF >= F.Nleft) ? F.mvKeysRight[bestIdxF - F.Nleft] : F.mvKeys[bestIdxF];

                            float rot = kp.angle - Fkp.angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                    // 只有在左目通过时才考虑右目
                    if (bestDist1R <= TH_LOW)
                    {
                        if (static_cast<float>(bestDist1R) < mfNNratio * static_cast<float>(bestDist2R) || true)
                        {
                            vpMapPointMatches[bestIdxFR] = pMP;

                            const cv::KeyPoint &kp =
                                (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] : (realIdxKF >= pKF->NLeft) ? pKF->mvKeysRight[realIdxKF - pKF->NLeft] : pKF->mvKeys[realIdxKF];

                            if (mbCheckOrientation)
                            {
                                cv::KeyPoint &Fkp =
                                    (!F.mpCamera2) ? F.mvKeys[bestIdxFR] : (bestIdxFR >= F.Nleft) ? F.mvKeysRight[bestIdxFR - F.Nleft] : F.mvKeys[bestIdxFR];

                                float rot = kp.angle - Fkp.angle;
                                if (rot < 0.0)
                                    rot += 360.0f;
                                int bin = round(rot * factor);
                                if (bin == HISTO_LENGTH)
                                    bin = 0;
                                assert(bin >= 0 && bin < HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxFR);
                            }
                            nmatches++;
                        }
                    }
                }
            }
            // 下一组单词集合
            KFit++;
            Fit++;
        }
        else if (KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first); // 返回map中第一个大于或等于key的迭代器指针
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }
    // 与初始化一样，保证旋转不变性
    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/** 
 * @brief LoopClosing::DetectCommonRegionsFromBoW中使用，针对优化后的Scw， 先做投影，设定范围，进行筛选，选择距离最小的匹配点
 * @param pKF 当前关键帧
 * @param Scw 世界坐标到相机的相似变换矩阵
 * @param vpPoints 所有候选MP点
 * @param vpMatched 每次进来都是空的，大小为pKF当前关键帧的MP数量（也就是特征点数量，中间有NULL）
 * @param th 搜索半径
 * @param ratioHamming 汉明距离的比率
 */
int ORBmatcher::SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, vector<MapPoint *> &vpMatched, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw / scw;
    cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
    cv::Mat Ow = -Rcw.t() * tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint *>(NULL));

    int nmatches = 0;

    // For each Candidate MapPoint Project and Match
    for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++)
    {
        MapPoint *pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw * p3Dw + tcw;

        // Depth must be positive
        if (p3Dc.at<float>(2) < 0.0)
            continue;

        // Project into Image
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x, y, z));

        // Point must be inside the image
        if (!pKF->IsInImage(uv.x, uv.y))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw - Ow;
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        // 夹角大于60度的不要
        if (PO.dot(Pn) < 0.5 * dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist, pKF);

        // Search in a radius
        const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
        // 找到在这个范围内所有的候选点
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x, uv.y, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;
            // 如果候选点已经在vpMatched里面了，不考虑
            if (vpMatched[idx])
                continue;

            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_LOW * ratioHamming)
        {
            vpMatched[bestIdx] = pMP;
            nmatches++;
        }
    }

    return nmatches;
}

/** 
 * @brief LoopClosing::DetectCommonRegionsFromBoW中使用，通过对所有候选MP投影到当前关键帧查找匹配的特征点
 * @param pKF 当前关键帧
 * @param Scw 世界坐标到相机的相似变换矩阵
 * @param vpPoints 候选MP点
 * @param vpPointsKFs 候选MP点对应的关键帧
 * @param vpMatched 匹配上的点
 * @param vpMatchedKF 匹配上的点对应的候选帧
 * @param th 搜索半径
 * @param ratioHamming 汉明距离的比率
 */
int ORBmatcher::SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints, const std::vector<KeyFrame *> &vpPointsKFs,
                                    std::vector<MapPoint *> &vpMatched, std::vector<KeyFrame *> &vpMatchedKF, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    // 内参
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    // 把尺度提取出来，还有旋转平移以及当前关键帧在世界坐标系下的三维坐标
    cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw / scw;
    cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
    cv::Mat Ow = -Rcw.t() * tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint *>(NULL));

    int nmatches = 0;

    // For each Candidate MapPoint Project and Match
    for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++)
    {
        // 1. 取出候选点与对应的关键帧
        MapPoint *pMP = vpPoints[iMP];
        KeyFrame *pKFi = vpPointsKFs[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        // 2. 计算候选点在当前关键帧下的坐标
        cv::Mat p3Dc = Rcw * p3Dw + tcw;

        // Depth must be positive
        // 过滤深度不合格的
        if (p3Dc.at<float>(2) < 0.0)
            continue;

        // Project into Image
        // 3. 投影到像素坐标系
        const float invz = 1 / p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0) * invz;
        const float y = p3Dc.at<float>(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        // 过滤掉像素坐标不合格的
        if (!pKF->IsInImage(u, v))
            continue;

        // Depth must be inside the scale invariance region of the point
        // TODO 4. 计算距离查看是否符合范围内（这个是回环用的，用这个判定真的合适吗？）
        const float maxDistance = pMP->GetMaxDistanceInvariance(); // 1.2f*mfMaxDistance
        const float minDistance = pMP->GetMinDistanceInvariance(); // 0.8f*mfMinDistance
        cv::Mat PO = p3Dw - Ow;
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        // 5. 角度判定
        cv::Mat Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5 * dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist, pKF);

        // Search in a radius
        // 6. 根据尺度计算搜索范围
        const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
        // 7. 根据范围选择候选特征点
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();
        // 8.找到最佳的匹配点，但并不一定每次都找到
        int bestDist = 256;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;
            if (vpMatched[idx])
                continue;

            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_LOW * ratioHamming)
        {
            vpMatched[bestIdx] = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            nmatches++;
        }
    }

    return nmatches;
}

/** 
 * @brief 查找两个帧匹配关系，单目初始化中有使用
 * @param F1 第一帧，在重置初始化前这个一直是第一帧
 * @param F2 候选帧
 * @param vbPrevMatched 长度与F1的特征点数量一样，里面存放的是对应位置的在上一次初始化的匹配点的位置，作用详情请看调用时的解释
 * @param vnMatches12 长度与F1的特征点数量一样，在相同位置存放与之匹配的F2特征点的id，也就是特征点在F2的id
 * @param windowSize 创建搜寻窗口
 */
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches = 0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

    vector<int> rotHist[HISTO_LENGTH]; // 30
    // 直方图30格。每格500个位置
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(), -1); // 它的顺序对应着第二帧特征点的顺序，内容为与第一帧匹配特征点的id，反之毅然

    // 遍历上一帧所有特征点
    for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave; // 返回该特征点所在的金字塔层数
        if (level1 > 0)
            continue;

        // 根据范围及金字塔层数提取候选点，F1中一个特征点对应了多个候选点，候选点满足在范围内，且金字塔层数一样
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1, level1); // TODO 这里windowSize阈值能否改成随着图像大小变化而变化

        if (vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1); // 获取描述子

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        // 遍历候选点，得到描述子距离最小的那个
        for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1, d2);

            if (vMatchedDistance[i2] <= dist)
                continue;

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if (bestDist <= TH_LOW) // 如果小于50
        {
            // 最小距离要比次小距离的mfNNratio倍数还要小，单目初始化时mfNNratio 0.9 相对于正常跟踪要求要低一些
            if (bestDist < (float)bestDist2 * mfNNratio)
            {
                // 如果帧2中的点在帧1中已经有了匹配对象，踢出
                if (vnMatches21[bestIdx2] >= 0)
                {
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                }
                // 互相赋值
                vnMatches12[i1] = bestIdx2;
                vnMatches21[bestIdx2] = i1;
                vMatchedDistance[bestIdx2] = bestDist;
                nmatches++;

                // 向直方图里面存放每个点的转动角度
                if (mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
                    if (rot < 0.0)
                        rot += 360.0f;
                    int bin = round(rot * factor);
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    assert(bin >= 0 && bin < HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        // 找到直方图中数量最多的前三个，如果数量差的过多，则有可能前两个或一个
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        // 过滤掉未在选出的直方图的匹配
        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                int idx1 = rotHist[i][j];
                if (vnMatches12[idx1] >= 0)
                {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    //Update prev matched
    // 更新vbPrevMatched供下一帧使用，顺序为上一帧的id，内容为当前帧的特征点坐标
    for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
        if (vnMatches12[i1] >= 0)
            vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

/** 
 * @brief 通过BoW查找两个关键帧共有的MP，LoopClosing::DetectCommonRegionsFromBoW中有使用
 * @param pKF1 当前帧
 * @param pKF2 候选帧
 * @param vpMatches12 对应点，顺序和大小与当前帧特征点顺序一样
 */
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    // 1. 准备工作，需要注意的是vpMapPoints1与vpMapPoints2中可能有一些点表示一个，但回环前在地图上显示的是两个MP
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint *>(vpMapPoints1.size(), static_cast<MapPoint *>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(), false);
    // 准备直方图滤波
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    int nmatches = 0;
    // 2. 遍历两个关键帧的单词，找到一样的
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                // 两个单目情况下，点为第二个相机上的则跳过
                if (pKF1->NLeft != -1 && idx1 >= pKF1->mvKeysUn.size())
                {
                    continue;
                }
                // 2.1 取点
                MapPoint *pMP1 = vpMapPoints1[idx1];
                if (!pMP1)
                    continue;
                if (pMP1->isBad())
                    continue;
                // 2.2 取描述子
                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1 = 256;
                int bestIdx2 = -1;
                int bestDist2 = 256;
                // 2.3 从候选关键帧中对应单词取出每一个MP点
                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    // 跟上面同样的操作
                    const size_t idx2 = f2it->second[i2];

                    if (pKF2->NLeft != -1 && idx2 >= pKF2->mvKeysUn.size())
                    {
                        continue;
                    }

                    MapPoint *pMP2 = vpMapPoints2[idx2];

                    if (vbMatched2[idx2] || !pMP2)
                        continue;

                    if (pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);
                    // 2.4 计算描述子距离
                    int dist = DescriptorDistance(d1, d2);

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2 = idx2;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }
                //
                if (bestDist1 < TH_LOW)
                {
                    // 2.5 确保最小距离要比次小距离的mfNNratio倍还要少才可以，一般mfNNratio为0~1的小数
                    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                    {
                        // 2.6 赋值了，表示当前关键帧pKF1中的MP对应pKF2中的MP
                        vpMatches12[idx1] = vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2] = true;
                        // 2,7 旋转直方图滤波
                        if (mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }
    // 3. 直方图滤波去掉转角差别大的个别点
    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vpMatches12[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/** 
 * @brief 通过三角化、基线方式匹配
 * LocalMapping::CreateNewMapPoints() 中使用，特别注意的是对于非单目一些特征点可能已经通过左右目生成了MP，这里直接跳过
 * @param pKF1 当前关键帧
 * @param pKF2 与当前关键帧共视度比较高的帧
 * @param F12  基本矩阵，没用到
 * @param vMatchedPairs 存放匹配结果
 * @param bOnlyStereo 在双目和rgbd情况下，要求特征点在右图存在匹配
 * @param bCoarse 是否不使用极线约束
 */
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                        vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
{
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    // Compute epipole in second image
    // 将KF1在世界坐标系的坐标转换到KF2的相机坐标系下，也就是KF1在以KF2为原点的坐标系下的坐标C2
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w * Cw + t2w;

    cv::Point2f ep = pKF2->mpCamera->project(C2); // 投影到像素坐标

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    cv::Mat R12;
    cv::Mat t12;

    // 第一个是KF1的，第二个是KF2的，例Rll表示的是KF2左目到KF1左目的旋转变换
    cv::Mat Rll, Rlr, Rrl, Rrr;
    cv::Mat tll, tlr, trl, trr;

    GeometricCamera *pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

    //
    if (!pKF1->mpCamera2 && !pKF2->mpCamera2)
    {
        R12 = R1w * R2w.t();
        t12 = -R1w * R2w.t() * t2w + t1w;
    }
    else
    {
        Rll = pKF1->GetRotation() * pKF2->GetRotation().t();
        Rlr = pKF1->GetRotation() * pKF2->GetRightRotation().t();
        Rrl = pKF1->GetRightRotation() * pKF2->GetRotation().t();
        Rrr = pKF1->GetRightRotation() * pKF2->GetRightRotation().t();

        tll = pKF1->GetRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetTranslation();
        tlr = pKF1->GetRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetTranslation();
        trl = pKF1->GetRightRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetRightTranslation();
        trr = pKF1->GetRightRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetRightTranslation();
    }

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches = 0;
    vector<bool> vbMatched2(pKF2->N, false);
    vector<int> vMatches12(pKF1->N, -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint *pMP1 = pKF1->GetMapPoint(idx1);

                // If there is already a MapPoint skip
                // 这里跟别的不一样，如果已经是MP，则跳过，因为这一步匹配是为了要造一些新MP出来
                if (pMP1)
                {
                    continue;
                }

                const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1] >= 0);
                // LocalMapping里面这里是false，不管什么模式，暂时走不到这里
                if (bOnlyStereo)
                    if (!bStereo1)
                        continue;

                // 取对应的像素坐标点
                const cv::KeyPoint &kp1 = (pKF1->NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                : (idx1 < pKF1->NLeft) ? pKF1->mvKeys[idx1]
                                                                                        : pKF1->mvKeysRight[idx1 - pKF1->NLeft];
                // 两个相机是否分开讨论
                const bool bRight1 = (pKF1->NLeft == -1 || idx1 < pKF1->NLeft) ? false
                                                                                : true;
                //if(bRight1) continue;
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                // 遍历KF2的词袋数据
                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

                    // If we have already matched or there is a MapPoint skip
                    // 如果pKF2当前特征点索引idx2已经被匹配过或者对应的3d点非空
                    // 那么这个索引idx2就不能被考虑
                    if (vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = (!pKF2->mpCamera2 && pKF2->mvuRight[idx2] >= 0);

                    if (bOnlyStereo)
                        if (!bStereo2)
                            continue;

                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                    const int dist = DescriptorDistance(d1, d2);

                    if (dist > TH_LOW || dist > bestDist)
                        continue;
                    // 通过特征点索引idx2在pKF2中取出对应的特征点
                    const cv::KeyPoint &kp2 = (pKF2->NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                    : (idx2 < pKF2->NLeft) ? pKF2->mvKeys[idx2]
                                                                                            : pKF2->mvKeysRight[idx2 - pKF2->NLeft];
                    const bool bRight2 = (pKF2->NLeft == -1 || idx2 < pKF2->NLeft) ? false
                                                                                    : true;

                    // 通过像素平面来判断MP点与KF1之间的距离是否在范围内
                    if (!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                    {
                        const float distex = ep.x - kp2.pt.x;
                        const float distey = ep.y - kp2.pt.y;
                        // 该特征点距离极点太近，表明kp2对应的MapPoint距离pKF1相机太近
                        if (distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave])
                        {
                            continue;
                        }
                    }

                    if (pKF1->mpCamera2 && pKF2->mpCamera2)
                    {
                        if (bRight1 && bRight2)
                        {
                            R12 = Rrr;
                            t12 = trr;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else if (bRight1 && !bRight2)
                        {
                            R12 = Rrl;
                            t12 = trl;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera;
                        }
                        else if (!bRight1 && bRight2)
                        {
                            R12 = Rlr;
                            t12 = tlr;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else
                        {
                            R12 = Rll;
                            t12 = tll;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera;
                        }
                    }

                    // 步骤4：计算特征点kp2到kp1极线（kp1对应pKF2的一条极线）的距离是否小于阈值
                    if (pCamera1->epipolarConstrain(pCamera2, kp1, kp2, R12, t12, pKF1->mvLevelSigma2[kp1.octave], pKF2->mvLevelSigma2[kp2.octave]) || bCoarse) // MODIFICATION_2
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if (bestIdx2 >= 0)
                {
                    const cv::KeyPoint &kp2 = (pKF2->NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                    : (bestIdx2 < pKF2->NLeft) ? pKF2->mvKeys[bestIdx2]
                                                                                                : pKF2->mvKeysRight[bestIdx2 - pKF2->NLeft];
                    vMatches12[idx1] = bestIdx2;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        float rot = kp1.angle - kp2.angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vMatches12[rotHist[i][j]] = -1;
                nmatches--;
            }
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (vMatches12[i] < 0)
            continue;
        vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
    }

    return nmatches;
}

/** 
 * @brief 纵观全局，并没有用到，暂时不看了，懒。。。
 */
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                        vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo, vector<cv::Mat> &vMatchedPoints)
{
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w * Cw + t2w;

    cv::Point2f ep = pKF2->mpCamera->project(C2);

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    GeometricCamera *pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;
    cv::Mat Tcw1, Tcw2;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches = 0;
    vector<bool> vbMatched2(pKF2->N, false);
    vector<int> vMatches12(pKF1->N, -1);

    vector<cv::Mat> vMatchesPoints12(pKF1->N);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
    int right = 0;
    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint *pMP1 = pKF1->GetMapPoint(idx1);

                // If there is already a MapPoint skip
                if (pMP1)
                    continue;

                const cv::KeyPoint &kp1 = (pKF1->NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                : (idx1 < pKF1->NLeft) ? pKF1->mvKeys[idx1]
                                                                                        : pKF1->mvKeysRight[idx1 - pKF1->NLeft];

                const bool bRight1 = (pKF1->NLeft == -1 || idx1 < pKF1->NLeft) ? false
                                                                                : true;

                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = TH_LOW;
                int bestIdx2 = -1;

                cv::Mat bestPoint;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

                    // If we have already matched or there is a MapPoint skip
                    if (vbMatched2[idx2] || pMP2)
                        continue;

                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                    const int dist = DescriptorDistance(d1, d2);

                    if (dist > TH_LOW || dist > bestDist)
                    {
                        continue;
                    }

                    const cv::KeyPoint &kp2 = (pKF2->NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                    : (idx2 < pKF2->NLeft) ? pKF2->mvKeys[idx2]
                                                                                            : pKF2->mvKeysRight[idx2 - pKF2->NLeft];
                    const bool bRight2 = (pKF2->NLeft == -1 || idx2 < pKF2->NLeft) ? false
                                                                                    : true;

                    if (bRight1)
                    {
                        Tcw1 = pKF1->GetRightPose();
                        pCamera1 = pKF1->mpCamera2;
                    }
                    else
                    {
                        Tcw1 = pKF1->GetPose();
                        pCamera1 = pKF1->mpCamera;
                    }

                    if (bRight2)
                    {
                        Tcw2 = pKF2->GetRightPose();
                        pCamera2 = pKF2->mpCamera2;
                    }
                    else
                    {
                        Tcw2 = pKF2->GetPose();
                        pCamera2 = pKF2->mpCamera;
                    }

                    cv::Mat x3D;
                    if (pCamera1->matchAndtriangulate(kp1, kp2, pCamera2, Tcw1, Tcw2, pKF1->mvLevelSigma2[kp1.octave], pKF2->mvLevelSigma2[kp2.octave], x3D))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                        bestPoint = x3D;
                    }
                }

                if (bestIdx2 >= 0)
                {
                    const cv::KeyPoint &kp2 = (pKF2->NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                    : (bestIdx2 < pKF2->NLeft) ? pKF2->mvKeys[bestIdx2]
                                                                                                : pKF2->mvKeysRight[bestIdx2 - pKF2->NLeft];
                    vMatches12[idx1] = bestIdx2;
                    vMatchesPoints12[idx1] = bestPoint;
                    nmatches++;
                    if (bRight1)
                        right++;

                    if (mbCheckOrientation)
                    {
                        float rot = kp1.angle - kp2.angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vMatches12[rotHist[i][j]] = -1;
                nmatches--;
            }
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (vMatches12[i] < 0)
            continue;
        vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
        vMatchedPoints.push_back(vMatchesPoints12[i]);
    }
    return nmatches;
}

/** 
 * @brief 将MapPoints投影到关键帧pKF中，并判断是否有重复的MapPoints 
 * 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
 * 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
 * @param pKF 关键帧
 * @param vpMapPoints 待融合的mp
 * @param th 搜索范围
 * @param bRight 是否是右目
 */
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
{
    cv::Mat Rcw, tcw, Ow;
    GeometricCamera *pCamera;

    if (bRight)
    {
        Rcw = pKF->GetRightRotation();
        tcw = pKF->GetRightTranslation();
        Ow = pKF->GetRightCameraCenter();

        pCamera = pKF->mpCamera2;
    }
    else
    {
        Rcw = pKF->GetRotation();
        tcw = pKF->GetTranslation();
        Ow = pKF->GetCameraCenter();

        pCamera = pKF->mpCamera;
    }
    // 内参
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    int nFused = 0; // 重复MapPoints的数量，要返回的数值

    const int nMPs = vpMapPoints.size();

    // For debbuging，测试用的
    int count_notMP = 0, count_bad = 0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal = 0, count_notidx = 0, count_thcheck = 0;
    // 遍历所有的MapPoints
    for (int i = 0; i < nMPs; i++)
    {
        MapPoint *pMP = vpMapPoints[i];

        if (!pMP)
        {
            count_notMP++;
            continue;
        }

        /*if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
        continue;*/
        if (pMP->isBad())
        {
            count_bad++;
            continue;
        }
        else if (pMP->IsInKeyFrame(pKF)) // 在这个关键帧中能找到MP，跳过
        {
            count_isinKF++;
            continue;
        }

        cv::Mat p3Dw = pMP->GetWorldPos(); //获取MP在世界坐标系3D坐标
        cv::Mat p3Dc = Rcw * p3Dw + tcw;   //求取MP在相机坐标系下的坐标

        // Depth must be positive
        if (p3Dc.at<float>(2) < 0.0f)
        {
            count_negdepth++;
            continue;
        }

        const float invz = 1 / p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        // 步骤1：得到MapPoint在图像上的投影坐标(此时不存在匹配关系)
        const cv::Point2f uv = pCamera->project(cv::Point3f(x, y, z));

        // Point must be inside the image
        //如果Point不在图片内
        if (!pKF->IsInImage(uv.x, uv.y))
        {
            count_notinim++;
            continue;
        }

        const float ur = uv.x - bf * invz; // 单目时bf为0，计算点在右目的横坐标

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw - Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image深度必须在图像的尺度金字塔内
        if (dist3D < minDistance || dist3D > maxDistance)
        {
            count_dist++;
            continue;
        }

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        // 当前视线与平均观测视线的角度不能大于60度，这个约束条件要求很低了
        if (PO.dot(Pn) < 0.5 * dist3D)
        {
            count_normal++;
            continue;
        }
        // 根据距离及最大距离估计出这个点在这个帧可能在的层数
        int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

        // Search in a radius
        // 步骤2：根据MapPoint的深度确定尺度，从而确定搜索范围
        const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
        // 在范围内搜索候选点,先确定点可能在的框，再把所有框里面点符合范围的全部返回
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x, uv.y, radius, bRight);

        if (vIndices.empty())
        {
            count_notidx++;
            continue;
        }

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        // 步骤3：遍历搜索范围内的features
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            size_t idx = *vit;
            const cv::KeyPoint &kp = (pKF->NLeft == -1) ? pKF->mvKeysUn[idx]
                                                        : (!bRight) ? pKF->mvKeys[idx]
                                                                    : pKF->mvKeysRight[idx];

            const int &kpLevel = kp.octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;
            // 计算MapPoint投影的坐标与这个区域特征点的距离，如果偏差很大，直接跳过特征点匹配
            if (pKF->mvuRight[idx] >= 0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = uv.x - kpx;
                const float ey = uv.y - kpy;
                const float er = ur - kpr;
                const float e2 = ex * ex + ey * ey + er * er;

                if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = uv.x - kpx;
                const float ey = uv.y - kpy;
                const float e2 = ex * ex + ey * ey;

                if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                    continue;
            }

            if (bRight)
                idx += pKF->NLeft;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);
            // 找MapPoint在该区域最佳匹配的特征点
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        // 找到了MapPoint在该区域最佳匹配的特征点
        if (bestDist <= TH_LOW) // 50
        {
            MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
            // 如果这个点有对应的MapPoint
            if (pMPinKF)
            {
                // 如果这个MapPoint不是bad，选择哪一个呢？哪个观测的多选哪一个
                if (!pMPinKF->isBad())
                {
                    if (pMPinKF->Observations() > pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else // 如果这个点没有对应的MapPoint，那么为该点添加MapPoint
            {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            nFused++;
        }
        else
            count_thcheck++;
    }

    /*cout << "count_notMP = " << count_notMP << endl;
cout << "count_bad = " << count_bad << endl;
cout << "count_isinKF = " << count_isinKF << endl;
cout << "count_negdepth = " << count_negdepth << endl;
cout << "count_notinim = " << count_notinim << endl;
cout << "count_dist = " << count_dist << endl;
cout << "count_normal = " << count_normal << endl;
cout << "count_notidx = " << count_notidx << endl;
cout << "count_thcheck = " << count_thcheck << endl;
cout << "tot fused points: " << nFused << endl;*/
    return nFused;
}

/** 
 * @brief 在LoopClosing::SearchAndFuse中地图融合使用
 * 将MapPoints投影到关键帧pKF中，并判断是否有重复的MapPoints 
 * 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么存入vpReplacePoint
 * 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
 * @param pKF 当前地图的当前关键帧及5个共视关键帧
 * @param Scw 关键帧的位姿Tcw
 * @param vpPoints 待融合地图的融合帧及其5个共视关键帧对应的mp（1000个以内）（注意此时所有kf与mp全部移至当前地图，这里的待融合地图的说法只为区分，因为还没有融合）
 * @param th 搜索范围
 * @param vpReplacePoint 数量与vpPoints相同，
 */
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    // BUG 去掉尺度，因为在这之前已经将当前地图的kf与mp的位姿移动到了待融合地图的坐标系下，里面并不包含尺度信息
    // Scw = Tcy scw就是1
    cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw / scw;
    cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
    cv::Mat Ow = -Rcw.t() * tcw; // 关键帧在新世界坐标系（待融合地图坐标系）下的位置

    // Set of MapPoints already found in the KeyFrame
    // 1. 先取出关键帧现有的mp
    const set<MapPoint *> spAlreadyFound = pKF->GetMapPoints();

    int nFused = 0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    // 2. 遍历vpPoints
    for (int iMP = 0; iMP < nPoints; iMP++)
    {
        MapPoint *pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if (pMP->isBad() || spAlreadyFound.count(pMP)) // TODO 两个地图的mp能有对应的吗？
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        // 2.1 将新坐标系下的mp转到关键帧坐标系下，差了一个尺度？？？？不用担心，尺度总是为1
        cv::Mat p3Dc = Rcw * p3Dw + tcw;
        // 后面就是一连串的投影加验证投影的好坏
        // Depth must be positive
        if (p3Dc.at<float>(2) < 0.0f)
            continue;

        // Project into Image
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x, y, z));

        // Point must be inside the image
        if (!pKF->IsInImage(uv.x, uv.y))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw - Ow;
        const float dist3D = cv::norm(PO);

        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        // 观测方向与平均方向的夹角要小于60度
        if (PO.dot(Pn) < 0.5 * dist3D)
            continue;

        // Compute predicted scale level
        // 根据距离与观测最大距离预测尺度金字塔层数
        const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

        // Search in a radius
        const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x, uv.y, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(); vit != vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if (bestDist <= TH_LOW)
        {
            MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
            // BUG 这里改成 if(pMPinKF && !pMPinKF->isBad()) 是不是好一点
            if (pMPinKF)
            {
                if (!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12,
                                const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12 * R12;
    cv::Mat sR21 = (1.0 / s12) * R12.t();
    cv::Mat t21 = -sR21 * t12;

    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1, false);
    vector<bool> vbAlreadyMatched2(N2, false);

    for (int i = 0; i < N1; i++)
    {
        MapPoint *pMP = vpMatches12[i];
        if (pMP)
        {
            vbAlreadyMatched1[i] = true;
            int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
            if (idx2 >= 0 && idx2 < N2)
                vbAlreadyMatched2[idx2] = true;
        }
    }

    vector<int> vnMatch1(N1, -1);
    vector<int> vnMatch2(N2, -1);

    // Transform from KF1 to KF2 and search
    for (int i1 = 0; i1 < N1; i1++)
    {
        MapPoint *pMP = vpMapPoints1[i1];

        if (!pMP || vbAlreadyMatched1[i1])
            continue;

        if (pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w * p3Dw + t1w;
        cv::Mat p3Dc2 = sR21 * p3Dc1 + t21;

        // Depth must be positive
        if (p3Dc2.at<float>(2) < 0.0)
            continue;

        const float invz = 1.0 / p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0) * invz;
        const float y = p3Dc2.at<float>(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF2->IsInImage(u, v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

        // Search in a radius
        const float radius = th * pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH)
        {
            vnMatch1[i1] = bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for (int i2 = 0; i2 < N2; i2++)
    {
        MapPoint *pMP = vpMapPoints2[i2];

        if (!pMP || vbAlreadyMatched2[i2])
            continue;

        if (pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w * p3Dw + t2w;
        cv::Mat p3Dc1 = sR12 * p3Dc2 + t12;

        // Depth must be positive
        if (p3Dc1.at<float>(2) < 0.0)
            continue;

        const float invz = 1.0 / p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0) * invz;
        const float y = p3Dc1.at<float>(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF1->IsInImage(u, v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th * pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH)
        {
            vnMatch2[i2] = bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for (int i1 = 0; i1 < N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if (idx2 >= 0)
        {
            int idx1 = vnMatch2[idx2];
            if (idx1 == i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

/** 
 * @brief 跟踪匀速模型时使用的，通过对所有候选MP投影到当前关键帧查找匹配的特征点
 * @param CurrentFrame 当前帧
 * @param LastFrame 上一帧
 * @param th 投影后搜索范围
 * @param bMono 是否是单目
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    // 旋转一致性用的直方图，这里有个不影响结果的小bug
    // 原作者的意思想弄一个30长度的直方图，也就是每30度一个，360度大约需要12~13个，结果数量也给弄成30个了，顶多就是占地方，别的没事
    // TODO 原作者的意思也可能想将360度分30格，每格代表12度，这个不好说，不懂原作者的意思，不过结果是跟上一行一样，有机会可以试试
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

    const cv::Mat twc = -Rcw.t() * tcw;

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);

    const cv::Mat tlc = Rlw * twc + tlw;

    // 双目时，一般来说mb（基线长度）都是正的，应该也有负的比较隔瑟。
    const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;   // 表示当前帧相对于上一帧前进的值大于基线
    const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono; // 表示当前帧相对于上一帧后退的值大于基线

    // 取出上一帧每个特征点对应的有效的MP
    for (int i = 0; i < LastFrame.N; i++)
    {
        MapPoint *pMP = LastFrame.mvpMapPoints[i];
        if (pMP)
        {
            if (!LastFrame.mvbOutlier[i])
            {
                // Project
                // 给转到了当前相机坐标系下
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw * x3Dw + tcw;

                const float xc = x3Dc.at<float>(0);          // 没用到～
                const float yc = x3Dc.at<float>(1);          // 没用到～
                const float invzc = 1.0 / x3Dc.at<float>(2); // 归一化使用

                // 判断这个点是否在相机前面
                if (invzc < 0)
                    continue;
                // 投影到相机坐标系
                cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);

                // 查看是否过界
                if (uv.x < CurrentFrame.mnMinX || uv.x > CurrentFrame.mnMaxX)
                    continue;
                if (uv.y < CurrentFrame.mnMinY || uv.y > CurrentFrame.mnMaxY)
                    continue;

                // 查看这个点在上一帧所对应的金字塔层数
                int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                // Search in a window. Size depends on scale
                // 根据层数确定搜索窗口大小
                float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;
                // 向前走同一个点所在的金字塔层数变化应该变大，反之应该变小，走的不远就在层数附近。
                // 值得注意一点全部在CurrentFrame左目提取（如果是双目）
                if (bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, nLastOctave);
                else if (bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, nLastOctave - 1, nLastOctave + 1);

                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                {
                    const size_t i2 = *vit;

                    // 如果这个点已经有了MP且观测到这个点的帧数大于0表示已经匹配过了，所以跳过
                    if (CurrentFrame.mvpMapPoints[i2])
                        if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
                            continue;

                    // TODO 这里没咋懂CurrentFrame.Nleft == -1应该是表示单目模式，但又整出个右目，可能回看双目部分就能懂了
                    if (CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2] > 0)
                    {
                        const float ur = uv.x - CurrentFrame.mbf * invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if (er > radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP, d);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                    : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                            : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                        cv::KeyPoint kpCF = (CurrentFrame.Nleft == -1) ? CurrentFrame.mvKeysUn[bestIdx2]
                                                                        : (bestIdx2 < CurrentFrame.Nleft) ? CurrentFrame.mvKeys[bestIdx2]
                                                                                                            : CurrentFrame.mvKeysRight[bestIdx2 - CurrentFrame.Nleft];
                        float rot = kpLF.angle - kpCF.angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
                // 涉及双目的，回头再看～
                if (CurrentFrame.Nleft != -1)
                {
                    cv::Mat x3Dr = CurrentFrame.mTrl.colRange(0, 3).rowRange(0, 3) * x3Dc + CurrentFrame.mTrl.col(3);

                    cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dr);

                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                        : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

                    vector<size_t> vIndices2;

                    if (bForward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, nLastOctave, -1, true);
                    else if (bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, 0, nLastOctave, true);
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, nLastOctave - 1, nLastOctave + 1, true);

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                    {
                        const size_t i2 = *vit;
                        if (CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                            if (CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations() > 0)
                                continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                        const int dist = DescriptorDistance(dMP, d);

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }

                    if (bestDist <= TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft] = pMP;
                        nmatches++;
                        if (mbCheckOrientation)
                        {
                            cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                        : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                            cv::KeyPoint kpCF = CurrentFrame.mvKeysRight[bestIdx2];

                            float rot = kpLF.angle - kpCF.angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2 + CurrentFrame.Nleft);
                        }
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    // 旋转一致性检验
    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

/** 
 * @brief Tracking::Relocalization() 中使用，针对候选关键帧对应的MP在当前帧中寻找对应的特征点，如果事先已经有找到的则跳过，相当于额外增加一些匹配点
 * @param CurrentFrame 当前帧
 * @param pKF 候选关键帧
 * @param sAlreadyFound 已经找到的MP
 * @param th 投影后搜索范围
 * @param ORBdist 描述子距离阈值
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint *> &sAlreadyFound, const float th, const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);
    const cv::Mat Ow = -Rcw.t() * tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMPs[i];

        if (pMP)
        {
            // 如果已经找到了，就不考虑了
            if (!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw * x3Dw + tcw;

                const cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);

                // 图像范围考虑
                if (uv.x < CurrentFrame.mnMinX || uv.x > CurrentFrame.mnMaxX)
                    continue;
                if (uv.y < CurrentFrame.mnMinY || uv.y > CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw - Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                // 距离筛选
                if (dist3D < minDistance || dist3D > maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

                // Search in a window
                const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];
                // 通过投影及范围加上尺度搜索出一批满足条件的特征点
                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, nPredictedLevel - 1, nPredictedLevel + 1);

                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    // 如果这个特征点已经有对应点了，跳过
                    if (CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP, d);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]] = NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

/** 
 * @brief 取直方图中数量排前三列，如果不够则返回两列甚至一列
 * @param histo 直方图
 * @param L 直方图长度
 * @param ind1 最大的下标
 * @param ind2 第二大的下标，不存在则返回-1
 * @param ind3 第三大的下标，不存在则返回-1
 */
void ORBmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }
    // 如果过少，则宁缺毋滥
    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
/** 
 * @brief 计算描述子距离
 * @param a 描述子1
 * @param b 描述子2
 */
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} // namespace ORB_SLAM3
