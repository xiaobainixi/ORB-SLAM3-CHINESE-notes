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

#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM3
{

Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale, vector<KeyFrame *> vpKeyFrameMatchedMP) : mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale), pCamera1(pKF1->mpCamera), pCamera2(pKF2->mpCamera)
{
    bool bDifferentKFs = false;
    // vpKeyFrameMatchedMP这个不会为空吧？ 反正在回环里面不为空
    if (vpKeyFrameMatchedMP.empty())
    {
        bDifferentKFs = true;
        vpKeyFrameMatchedMP = vector<KeyFrame *>(vpMatched12.size(), pKF2);
    }

    mpKF1 = pKF1;
    mpKF2 = pKF2;

    vector<MapPoint *> vpKeyFrameMP1 = pKF1->GetMapPointMatches();
    // mN1 大小也就是vpKeyFrameMP1大小
    mN1 = vpMatched12.size();

    mvpMapPoints1.reserve(mN1);
    mvpMapPoints2.reserve(mN1);
    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    cv::Mat Rcw1 = pKF1->GetRotation();
    cv::Mat tcw1 = pKF1->GetTranslation();
    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat tcw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(mN1);

    size_t idx = 0;
    // mN1为pKF1特征点的个数
    KeyFrame *pKFm = pKF2; //Default variable
    for (int i1 = 0; i1 < mN1; i1++)
    {
        // 如果该特征点在pKF1中有匹配
        if (vpMatched12[i1])
        {
            // step1: pMP1
            MapPoint *pMP1 = vpKeyFrameMP1[i1];
            MapPoint *pMP2 = vpMatched12[i1];

            if (!pMP1)
                continue;

            if (pMP1->isBad() || pMP2->isBad())
                continue;

            if (bDifferentKFs)
                pKFm = vpKeyFrameMatchedMP[i1];

            // step2：计算允许的重投影误差阈值：mvnMaxError1和mvnMaxError2
            // 注：是相对当前位姿投影3D点得到的图像坐标，见step6
            // step2.1：根据匹配的MapPoint找到对应匹配特征点的索引：indexKF1和indexKF2
            int indexKF1 = get<0>(pMP1->GetIndexInKeyFrame(pKF1));
            int indexKF2 = get<0>(pMP2->GetIndexInKeyFrame(pKFm));

            if (indexKF1 < 0 || indexKF2 < 0)
                continue;

            // step2.2：取出匹配特征点的引用：kp1和kp2
            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];
            const cv::KeyPoint &kp2 = pKFm->mvKeysUn[indexKF2];

            // step2.3：根据特征点的尺度计算对应的误差阈值：mvnMaxError1和mvnMaxError2
            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float sigmaSquare2 = pKFm->mvLevelSigma2[kp2.octave];
            // 2自由度百分之99，理解一下，当误差大于这个时，我有百分之99的把握认为这个点是外点，也就是说也不一定，但概率很大，其他地方一般都用95%
            mvnMaxError1.push_back(9.210 * sigmaSquare1);
            mvnMaxError2.push_back(9.210 * sigmaSquare2);

            // mvpMapPoints1和mvpMapPoints2是匹配的MapPoints容器
            mvpMapPoints1.push_back(pMP1);
            mvpMapPoints2.push_back(pMP2);
            mvnIndices1.push_back(i1);

            // step4：将MapPoint从世界坐标系变换到相机坐标系：mvX3Dc1和mvX3Dc2
            cv::Mat X3D1w = pMP1->GetWorldPos();
            mvX3Dc1.push_back(Rcw1 * X3D1w + tcw1);

            cv::Mat X3D2w = pMP2->GetWorldPos();
            mvX3Dc2.push_back(Rcw2 * X3D2w + tcw2);

            mvAllIndices.push_back(idx);
            idx++;
        }
    }

    // step5：两个关键帧的内参
    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    // step6：记录计算两针Sim3之前3D mappoint在图像上的投影坐标：mvP1im1和mvP2im2
    FromCameraToImage(mvX3Dc1, mvP1im1, pCamera1);
    FromCameraToImage(mvX3Dc2, mvP2im2, pCamera2);

    SetRansacParameters();
}

/** 
 * @brief 计算ransac相关参数，主要是迭代次数
 * @param probability 点为内点的概率
 * @param minInliers 内点数量的最低要求
 * @param maxIterations 最高迭代次数
 */
void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;

    N = mvpMapPoints1.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers / N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if (mRansacMinInliers == N)
        nIterations = 1;
    else
        nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(epsilon, 3)));

    mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

    mnIterations = 0;
}

/** 
 * @brief Ransac求解mvX3Dc1和mvX3Dc2之间Sim3，函数返回mvX3Dc2到mvX3Dc1的Sim3变换
 * @param nIterations 迭代次数
 * @param bNoMore 迭代停止标志
 * @param vbInliers 存放内点
 * @param nInliers 内点个数
 * @return 
 */
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers = vector<bool>(mN1, false);
    nInliers = 0;

    // 如果最小内点数比总数还多，还算个啥
    if (N < mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    cv::Mat P3Dc1i(3, 3, CV_32F);
    cv::Mat P3Dc2i(3, 3, CV_32F);

    int nCurrentIterations = 0;
    // 单次迭代次数（这个函数迭代的次数）少于nIterations，总迭代次数（该类一共迭代了多少次）少于mRansacMaxIts
    while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations)
    {
        nCurrentIterations++; // 这个函数中迭代的次数
        mnIterations++;       // 总的迭代次数，默认为最大为300

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        // 步骤1：任意取三组点算Sim矩阵
        for (short i = 0; i < 3; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            // P3Dc1i和P3Dc2i中点的排列顺序：
            // x1 x2 x3 ...
            // y1 y2 y3 ...
            // z1 z2 z3 ...
            mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));
            // 删除选过的点
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
        // 步骤2：根据两组匹配的3D点，计算之间的Sim3变换
        ComputeSim3(P3Dc1i, P3Dc2i);
        // 步骤3：通过投影误差进行inlier检测
        CheckInliers();

        if (mnInliersi >= mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            if (mnInliersi > mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for (int i = 0; i < N; i++)
                    if (mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;

                // ！！！！！note: 1. 只要计算得到一次合格的Sim变换，就直接返回 2. 没有对所有的inlier进行一次refine操作
                return mBestT12;
            }
        }
    }

    if (mnIterations >= mRansacMaxIts)
        bNoMore = true;

    return cv::Mat();
}

/** 
 * @brief Ransac求解mvX3Dc1和mvX3Dc2之间Sim3，函数返回mvX3Dc2到mvX3Dc1的Sim3变换，和上面函数一样，就多了个bool输出
 * @param nIterations 迭代次数
 * @param bNoMore 迭代停止标志
 * @param vbInliers 存放内点
 * @param nInliers 内点个数
 * @return 
 */
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers, bool &bConverge)
{
    bNoMore = false;
    bConverge = false;
    vbInliers = vector<bool>(mN1, false);
    nInliers = 0;
    // 如果最小内点数比总数还多，还算个啥
    if (N < mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    cv::Mat P3Dc1i(3, 3, CV_32F);
    cv::Mat P3Dc2i(3, 3, CV_32F);

    int nCurrentIterations = 0;

    cv::Mat bestSim3;
    // 单次迭代次数（这个函数迭代的次数）少于nIterations，总迭代次数（该类一共迭代了多少次）少于mRansacMaxIts
    while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations)
    {
        nCurrentIterations++; // 这个函数中迭代的次数
        mnIterations++;       // 总的迭代次数，默认为最大为300

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        // 步骤1：任意取三组点算Sim矩阵
        for (short i = 0; i < 3; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            // P3Dc1i和P3Dc2i中点的排列顺序：
            // x1 x2 x3 ...
            // y1 y2 y3 ...
            // z1 z2 z3 ...
            mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));
            // 删除选过的点
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // 步骤2：根据两组匹配的3D点，计算之间的Sim3变换
        ComputeSim3(P3Dc1i, P3Dc2i);
        // 步骤3：通过投影误差进行inlier检测
        CheckInliers();

        if (mnInliersi >= mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            if (mnInliersi > mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for (int i = 0; i < N; i++)
                    if (mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;
                bConverge = true;
                // ！！！！！note: 1. 只要计算得到一次合格的Sim变换，就直接返回 2. 没有对所有的inlier进行一次refine操作
                return mBestT12;
            }
            else
            {
                bestSim3 = mBestT12;
            }
        }
    }

    if (mnIterations >= mRansacMaxIts)
        bNoMore = true;

    return bestSim3;
}

cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts, bFlag, vbInliers12, nInliers);
}

/** 
 * @brief 求质心，再减去
 * @param P 点以列向量的形式组合成矩阵
 * @param Pr 去掉质心的P
 * @param C 存放质心
 */
void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    // 这两句可以使用CV_REDUCE_AVG选项来搞定
    cv::reduce(P, C, 1, CV_REDUCE_SUM); // 矩阵P每一行求和
    C = C / P.cols;                     // 求平均，得到质心

    for (int i = 0; i < P.cols; i++)
    {
        Pr.col(i) = P.col(i) - C; //减去质心
    }
}

/** 
 * @brief 计算两组点的sim3
 * @param P1 当前关键帧坐标系下点的坐标
 * @param P2 候选关键帧坐标系下点的坐标
 */
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    // ！！！！！！！这段代码一定要看这篇论文！！！！！！！！！！！
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates（模型坐标系）

    cv::Mat Pr1(P1.size(), P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(), P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3, 1, Pr1.type());      // Centroid of P1
    cv::Mat O2(3, 1, Pr2.type());      // Centroid of P2

    // O1和O2分别为P1和P2矩阵中3D点的质心
    // Pr1和Pr2为减去质心后的3D点
    ComputeCentroid(P1, Pr1, O1);
    ComputeCentroid(P2, Pr2, O2);

    // Step 2: Compute M matrix
    cv::Mat M = Pr2 * Pr1.t();

    // Step 3: Compute N matrix

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4, 4, P1.type());

    N11 = M.at<float>(0, 0) + M.at<float>(1, 1) + M.at<float>(2, 2);
    N12 = M.at<float>(1, 2) - M.at<float>(2, 1);
    N13 = M.at<float>(2, 0) - M.at<float>(0, 2);
    N14 = M.at<float>(0, 1) - M.at<float>(1, 0);
    N22 = M.at<float>(0, 0) - M.at<float>(1, 1) - M.at<float>(2, 2);
    N23 = M.at<float>(0, 1) + M.at<float>(1, 0);
    N24 = M.at<float>(2, 0) + M.at<float>(0, 2);
    N33 = -M.at<float>(0, 0) + M.at<float>(1, 1) - M.at<float>(2, 2);
    N34 = M.at<float>(1, 2) + M.at<float>(2, 1);
    N44 = -M.at<float>(0, 0) - M.at<float>(1, 1) + M.at<float>(2, 2);

    N = (cv::Mat_<float>(4, 4) << N11, N12, N13, N14,
            N12, N22, N23, N24,
            N13, N23, N33, N34,
            N14, N24, N34, N44);

    // Step 4: Eigenvector of the highest eigenvalue

    cv::Mat eval, evec;
    // eval特征值，evec特征向量
    cv::eigen(N, eval, evec); //evec[0] is the quaternion of the desired rotation

    // N矩阵最大特征值（第一个特征值）对应特征向量就是要求的四元数死（q0 q1 q2 q3）
    // 将(q1 q2 q3)放入vec行向量，vec就是四元数旋转轴乘以sin(ang/2)
    cv::Mat vec(1, 3, evec.type());
    (evec.row(0).colRange(1, 4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    // tan(ang) = sin(θ/2) / cos(θ/2)  ang = θ/2
    double ang = atan2(norm(vec), evec.at<float>(0, 0));
    // 转成旋转向量 vec = θ*(nx, ny, nz)
    vec = 2 * ang * vec / norm(vec); //Angle-axis representation. quaternion angle is the half

    mR12i.create(3, 3, P1.type());
    // 转成旋转矩阵
    cv::Rodrigues(vec, mR12i); // computes the rotation matrix from angle-axis

    // Step 5: Rotate set 2

    cv::Mat P3 = mR12i * Pr2;

    // Step 6: Scale

    if (!mbFixScale)
    {
        // 论文中还有一个求尺度的公式，p632右中的位置，那个公式不用考虑旋转
        double nom = Pr1.dot(P3);
        cv::Mat aux_P3(P3.size(), P3.type());
        aux_P3 = P3;
        // 每个元素取平方
        cv::pow(P3, 2, aux_P3);
        double den = 0;

        for (int i = 0; i < aux_P3.rows; i++)
        {
            for (int j = 0; j < aux_P3.cols; j++)
            {
                den += aux_P3.at<float>(i, j);
            }
        }

        ms12i = nom / den;
    }
    else
        ms12i = 1.0f;

    // Step 7: Translation

    mt12i.create(1, 3, P1.type());
    mt12i = O1 - ms12i * mR12i * O2;

    // Step 8: Transformation

    // Step 8.1 T12
    mT12i = cv::Mat::eye(4, 4, P1.type());

    cv::Mat sR = ms12i * mR12i;

    sR.copyTo(mT12i.rowRange(0, 3).colRange(0, 3));
    mt12i.copyTo(mT12i.rowRange(0, 3).col(3));

    // Step 8.2 T21

    mT21i = cv::Mat::eye(4, 4, P1.type());

    cv::Mat sRinv = (1.0 / ms12i) * mR12i.t();

    sRinv.copyTo(mT21i.rowRange(0, 3).colRange(0, 3));
    cv::Mat tinv = -sRinv * mt12i;
    tinv.copyTo(mT21i.rowRange(0, 3).col(3));
}

void Sim3Solver::CheckInliers()
{
    vector<cv::Mat> vP1im2, vP2im1;
    Project(mvX3Dc2, vP2im1, mT12i, pCamera1); // 把2系中的3D经过Sim3变换(mT12i)到1系中计算重投影坐标
    Project(mvX3Dc1, vP1im2, mT21i, pCamera2); // 把1系中的3D经过Sim3变换(mT21i)到2系中计算重投影坐标

    mnInliersi = 0;

    for (size_t i = 0; i < mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i] - vP2im1[i];
        cv::Mat dist2 = vP1im2[i] - mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if (err1 < mvnMaxError1[i] && err2 < mvnMaxError2[i])
        {
            mvbInliersi[i] = true;
            mnInliersi++;
        }
        else
            mvbInliersi[i] = false;
    }
}

cv::Mat Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation.clone();
}

float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}

/** 
 * @brief 世界坐标转到像素坐标
 * @param vP3Dw 世界坐标系下的三维点坐标
 * @param vP2D 输出的像素坐标
 * @param Tcw Tcw
 * @param pCamera 相机模型
 */
void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, GeometricCamera *pCamera)
{
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for (size_t i = 0, iend = vP3Dw.size(); i < iend; i++)
    {
        cv::Mat P3Dc = Rcw * vP3Dw[i] + tcw;
        const float invz = 1 / (P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0);
        const float y = P3Dc.at<float>(1);
        const float z = P3Dc.at<float>(2);

        vP2D.push_back(pCamera->projectMat(cv::Point3f(x, y, z)));
    }
}

/** 
 * @brief 相机坐标转到像素坐标
 * @param vP3Dc 相机坐标系下的三维点坐标
 * @param vP2D 输出的像素坐标
 * @param pCamera 相机模型
 */
void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, GeometricCamera *pCamera)
{
    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for (size_t i = 0, iend = vP3Dc.size(); i < iend; i++)
    {
        const float invz = 1 / (vP3Dc[i].at<float>(2));
        const float x = vP3Dc[i].at<float>(0);
        const float y = vP3Dc[i].at<float>(1);
        const float z = vP3Dc[i].at<float>(2);

        vP2D.push_back(pCamera->projectMat(cv::Point3f(x, y, z)));
    }
}

} // namespace ORB_SLAM3
