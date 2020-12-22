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


#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM3
{

/**
 * @brief 根据词袋文件里的单词数量resizemvInvertedFile
 * @param voc 词袋文件
 */
KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}

/**
 * @brief 根据关键帧的词包，更新数据库的倒排索引
 * @param pKF 关键帧
 */
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);
    // 加载的词典有多少个，然后每个新的关键帧进来后需要查看里面有哪些单词，在worldID对应的mvInvertedFile中添加关键帧，最后的情况就是看每个单词下面对应哪些关键帧
    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

/**
 * @brief 关键帧被删除后，更新数据库的倒排索引
 * @param pKF 关键帧
 */
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    // 可以理解为上个函数的相反，删除对应的KF指针
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

/**
 * @brief 清空
 */
void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

/**
 * @brief 删除地图中所有关键帧的信息
 * @param pMap 地图
 */
void KeyFrameDatabase::clearMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(std::vector<list<KeyFrame*> >::iterator vit=mvInvertedFile.begin(), vend=mvInvertedFile.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =  *vit;

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend;)
        {
            KeyFrame* pKFi = *lit;
            if(pMap == pKFi->GetMap())
            {
                lit = lKFs.erase(lit);
                // Dont delete the KF because the class Map clean all the KF when it is destroyed
            }
            else
            {
                ++lit;
            }
        }
    }
}

/**
 * @brief 没用到，暂时不看
 */
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->GetMap()==pKF->GetMap()) // For consider a loop candidate it a candidate it must be in the same map
                {
                    if(pKFi->mnLoopQuery!=pKF->mnId)
                    {
                        pKFi->mnLoopWords=0;
                        if(!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->mnLoopQuery=pKF->mnId;
                            lKFsSharingWords.push_back(pKFi);
                        }
                    }
                    pKFi->mnLoopWords++;
                }


            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float, KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;

            float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float, KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for(list<pair<float, KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for(list<pair<float, KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

/**
 * @brief 没用到，暂时不看
 */
void KeyFrameDatabase::DetectCandidates(KeyFrame* pKF, float minScore, vector<KeyFrame*>& vpLoopCand, vector<KeyFrame*>& vpMergeCand)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWordsLoop, lKFsSharingWordsMerge;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->GetMap()==pKF->GetMap()) // For consider a loop candidate it a candidate it must be in the same map
                {
                    if(pKFi->mnLoopQuery!=pKF->mnId)
                    {
                        pKFi->mnLoopWords=0;
                        if(!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->mnLoopQuery=pKF->mnId;
                            lKFsSharingWordsLoop.push_back(pKFi);
                        }
                    }
                    pKFi->mnLoopWords++;
                }
                else if(!pKFi->GetMap()->IsBad())
                {
                    if(pKFi->mnMergeQuery!=pKF->mnId)
                    {
                        pKFi->mnMergeWords=0;
                        if(!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->mnMergeQuery=pKF->mnId;
                            lKFsSharingWordsMerge.push_back(pKFi);
                        }
                    }
                    pKFi->mnMergeWords++;
                }
            }
        }
    }

    if(lKFsSharingWordsLoop.empty() && lKFsSharingWordsMerge.empty())
        return;

    if(!lKFsSharingWordsLoop.empty())
    {
        list<pair<float, KeyFrame*> > lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords=0;
        for(list<KeyFrame*>::iterator lit=lKFsSharingWordsLoop.begin(), lend= lKFsSharingWordsLoop.end(); lit!=lend; lit++)
        {
            if((*lit)->mnLoopWords>maxCommonWords)
                maxCommonWords=(*lit)->mnLoopWords;
        }

        int minCommonWords = maxCommonWords*0.8f;

        int nscores=0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for(list<KeyFrame*>::iterator lit=lKFsSharingWordsLoop.begin(), lend= lKFsSharingWordsLoop.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;

            if(pKFi->mnLoopWords>minCommonWords)
            {
                nscores++;

                float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);

                pKFi->mLoopScore = si;
                if(si>=minScore)
                    lScoreAndMatch.push_back(make_pair(si, pKFi));
            }
        }

        if(!lScoreAndMatch.empty())
        {
            list<pair<float, KeyFrame*> > lAccScoreAndMatch;
            float bestAccScore = minScore;

            // Lets now accumulate score by covisibility
            for(list<pair<float, KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
            {
                KeyFrame* pKFi = it->second;
                vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                float bestScore = it->first;
                float accScore = it->first;
                KeyFrame* pBestKF = pKFi;
                for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
                {
                    KeyFrame* pKF2 = *vit;
                    if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
                    {
                        accScore+=pKF2->mLoopScore;
                        if(pKF2->mLoopScore>bestScore)
                        {
                            pBestKF=pKF2;
                            bestScore = pKF2->mLoopScore;
                        }
                    }
                }

                lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
                if(accScore>bestAccScore)
                    bestAccScore=accScore;
            }

            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = 0.75f*bestAccScore;

            set<KeyFrame*> spAlreadyAddedKF;
            vpLoopCand.reserve(lAccScoreAndMatch.size());

            for(list<pair<float, KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
            {
                if(it->first>minScoreToRetain)
                {
                    KeyFrame* pKFi = it->second;
                    if(!spAlreadyAddedKF.count(pKFi))
                    {
                        vpLoopCand.push_back(pKFi);
                        spAlreadyAddedKF.insert(pKFi);
                    }
                }
            }
        }

    }

    if(!lKFsSharingWordsMerge.empty())
    {
        //cout << "BoW candidates: " << lKFsSharingWordsMerge.size() << endl;
        list<pair<float, KeyFrame*> > lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords=0;
        for(list<KeyFrame*>::iterator lit=lKFsSharingWordsMerge.begin(), lend=lKFsSharingWordsMerge.end(); lit!=lend; lit++)
        {
            if((*lit)->mnMergeWords>maxCommonWords)
                maxCommonWords=(*lit)->mnMergeWords;
        }
        //cout << "Max common words: " << maxCommonWords << endl;

        int minCommonWords = maxCommonWords*0.8f;

        int nscores=0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for(list<KeyFrame*>::iterator lit=lKFsSharingWordsMerge.begin(), lend=lKFsSharingWordsMerge.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;

            if(pKFi->mnMergeWords>minCommonWords)
            {
                nscores++;

                float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
                //cout << "KF score: " << si << endl;

                pKFi->mMergeScore = si;
                if(si>=minScore)
                    lScoreAndMatch.push_back(make_pair(si, pKFi));
            }
        }
        //cout << "BoW candidates2: " << lScoreAndMatch.size() << endl;

        if(!lScoreAndMatch.empty())
        {
            list<pair<float, KeyFrame*> > lAccScoreAndMatch;
            float bestAccScore = minScore;

            // Lets now accumulate score by covisibility
            for(list<pair<float, KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
            {
                KeyFrame* pKFi = it->second;
                vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                float bestScore = it->first;
                float accScore = it->first;
                KeyFrame* pBestKF = pKFi;
                for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
                {
                    KeyFrame* pKF2 = *vit;
                    if(pKF2->mnMergeQuery==pKF->mnId && pKF2->mnMergeWords>minCommonWords)
                    {
                        accScore+=pKF2->mMergeScore;
                        if(pKF2->mMergeScore>bestScore)
                        {
                            pBestKF=pKF2;
                            bestScore = pKF2->mMergeScore;
                        }
                    }
                }

                lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
                if(accScore>bestAccScore)
                    bestAccScore=accScore;
            }

            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = 0.75f*bestAccScore;

            //cout << "Min score to retain: " << minScoreToRetain << endl;

            set<KeyFrame*> spAlreadyAddedKF;
            vpMergeCand.reserve(lAccScoreAndMatch.size());

            for(list<pair<float, KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
            {
                if(it->first>minScoreToRetain)
                {
                    KeyFrame* pKFi = it->second;
                    if(!spAlreadyAddedKF.count(pKFi))
                    {
                        vpMergeCand.push_back(pKFi);
                        spAlreadyAddedKF.insert(pKFi);
                    }
                }
            }
            //cout << "Candidates: " << vpMergeCand.size() << endl;
        }

    }

    //----
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
    {
        list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi=*lit;
            pKFi->mnLoopQuery=-1;
            pKFi->mnMergeQuery=-1;
        }
    }

}

/**
 * @brief 没用到，暂时不看
 */
void KeyFrameDatabase::DetectBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nMinWords)
{
    list<KeyFrame*> lKFsSharingWords;
    set<KeyFrame*> spConnectedKF;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        spConnectedKF = pKF->GetConnectedKeyFrames();
        // 遍历当前帧所包含的词汇
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            // 从mvInvertedFile里取出拥有这个词汇的关键帧们
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];
            // 遍历这些关键帧
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend=lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(spConnectedKF.find(pKFi) != spConnectedKF.end())
                {
                    continue;
                }
                if(pKFi->mnPlaceRecognitionQuery!=pKF->mnId)
                {
                    pKFi->mnPlaceRecognitionWords=0;
                    pKFi->mnPlaceRecognitionQuery=pKF->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
               pKFi->mnPlaceRecognitionWords++;

            }
        }
    }
    if(lKFsSharingWords.empty())
        return;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnPlaceRecognitionWords>maxCommonWords)
            maxCommonWords=(*lit)->mnPlaceRecognitionWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    if(minCommonWords < nMinWords)
    {
        minCommonWords = nMinWords;
    }

    list<pair<float, KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnPlaceRecognitionWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
            pKFi->mPlaceRecognitionScore=si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return;

    list<pair<float, KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float, KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnPlaceRecognitionQuery!=pKF->mnId)
                continue;

            accScore+=pKF2->mPlaceRecognitionScore;
            if(pKF2->mPlaceRecognitionScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mPlaceRecognitionScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vpLoopCand.reserve(lAccScoreAndMatch.size());
    vpMergeCand.reserve(lAccScoreAndMatch.size());
    for(list<pair<float, KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                if(pKF->GetMap() == pKFi->GetMap())
                {
                    vpLoopCand.push_back(pKFi);
                }
                else
                {
                    vpMergeCand.push_back(pKFi);
                }
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }
}

/**
 * @brief 相似度分数比较
 */
bool compFirst(const pair<float, KeyFrame*> & a, const pair<float, KeyFrame*> & b)
{
    return a.first > b.first;
}

/**
 * @brief 回环使用，找到候选关键帧，不局限于当前地图
 *
 * 1. 找出和当前帧具有公共单词的所有关键帧
 * 2. 只和具有共同单词较多的关键帧进行相似度计算
 * 3. 将与关键帧相连（权值最高）的前十个关键帧归为一组，计算累计得分
 * 4. 只返回累计得分较高的组中分数最高的关键帧
 * @param pKF 找这个pKF的候选关键帧
 * @param vpLoopCand 存放回环的候选关键帧
 * @param vpMergeCand 存放融合的候选关键帧
 * @param nNumCandidates 候选关键帧的数量限制
 * @see III-E Bags of Words Place Recognition
 */
void KeyFrameDatabase::DetectNBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nNumCandidates)
{
    list<KeyFrame*> lKFsSharingWords;
    //set<KeyFrame*> spInsertedKFsSharing;
    set<KeyFrame*> spConnectedKF;  // 存放与pKF相连的关键帧

    // Search all keyframes that share a word with current frame
    // 1. 找到所有与当前关键帧有相同词汇的所有关键帧，并且标记这些关键帧与统计有相同单词的数量
    {
        unique_lock<mutex> lock(mMutex);

        spConnectedKF = pKF->GetConnectedKeyFrames();

        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            // 1.1 从每个单词中找到所有有这个单词的关键帧
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];
            // 1.2 遍历每个关键帧
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi = *lit;
                /*if(spConnectedKF.find(pKFi) != spConnectedKF.end())
                {
                    continue;
                }*/
                // mnPlaceRecognitionQuery 防止重复添加
                if(pKFi->mnPlaceRecognitionQuery != pKF->mnId)
                {
                    // 统计拥有共同单词的数量
                    pKFi->mnPlaceRecognitionWords=0;
                    // 这个帧还不能与pKF相连，因为相连就代表他们离的很近，不存在回环或者融合
                    if(!spConnectedKF.count(pKFi))
                    {

                        pKFi->mnPlaceRecognitionQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnPlaceRecognitionWords++;
                /*if(spInsertedKFsSharing.find(pKFi) == spInsertedKFsSharing.end())
                {
                    lKFsSharingWords.push_back(pKFi);
                    spInsertedKFsSharing.insert(pKFi);
                }*/

            }
        }
    }
    // 2. 一帧都没有还玩个蛋，扯呼
    if(lKFsSharingWords.empty())
        return;

    // Only compare against those keyframes that share enough words
    // 3. 前面统计了数量，现在就要统计点数最高的那个值，话说这段是不是可以融到步骤一里面，就少了一波循环，不过也许这么做实际的计算量比融到步骤一少
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnPlaceRecognitionWords>maxCommonWords)
            maxCommonWords=(*lit)->mnPlaceRecognitionWords;
    }
    // 给一定范围来准备选
    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float, KeyFrame*> > lScoreAndMatch;

    int nscores=0;  // 没啥用

    // Compute similarity score.
    // 4. 根据最低单词数minCommonWords挑选关键帧存放至 lScoreAndMatch
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend=lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnPlaceRecognitionWords>minCommonWords)
        {
            nscores++;
            // 计算相似度分数
            float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
            pKFi->mPlaceRecognitionScore = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }
    // 一帧都没有还玩个蛋，扯呼
    if(lScoreAndMatch.empty())
        return;

    list<pair<float, KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    // 5. 又来遍历lScoreAndMatch这个，数量比lKFsSharingWords少很多，对于每一个lScoreAndMatch中的帧包含其10个共视关键帧为一组，中间可能有重叠，但不影响
    // 计算每一组的分数与相似度得分最高的关键帧，所以lAccScoreAndMatch个数与lScoreAndMatch相同
    for(list<pair<float, KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        // 5.1 找到这些帧对应的10个共视关键帧
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        // 5.2 遍历遍历
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            // 5.2.1 lScoreAndMatch中的帧的共视帧不在lKFsSharingWords这里，也就是不存在共同单词则跳过
            if(pKF2->mnPlaceRecognitionQuery != pKF->mnId)
                continue;
            // 计算累计分数
            accScore += pKF2->mPlaceRecognitionScore;
            // 找到最优关键帧
            if(pKF2->mPlaceRecognitionScore>bestScore)
            {
                pBestKF = pKF2;
                bestScore = pKF2->mPlaceRecognitionScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if(accScore>bestAccScore)
            bestAccScore = accScore;
    }

    //cout << "Amount of candidates: " << lAccScoreAndMatch.size() << endl;
    // 6. 按照分数从大到小排序
    lAccScoreAndMatch.sort(compFirst);

    vpLoopCand.reserve(nNumCandidates);
    vpMergeCand.reserve(nNumCandidates);
    set<KeyFrame*> spAlreadyAddedKF;
    //cout << "Candidates in score order " << endl;
    //for(list<pair<float, KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    int i = 0;
    list<pair<float, KeyFrame*> >::iterator it=lAccScoreAndMatch.begin();
    // TODO
    // 7. 循环，前提是数量不超过lAccScoreAndMatch的数量，且小于nNumCandidates，这里是不是应该改成小于等于？
    while(i < lAccScoreAndMatch.size() && (vpLoopCand.size() < nNumCandidates || vpMergeCand.size() < nNumCandidates))
    {
        //cout << "Accum score: " << it->first << endl;
        KeyFrame* pKFi = it->second;
        if(pKFi->isBad())
            continue;
        // 防止重复添加
        if(!spAlreadyAddedKF.count(pKFi))
        {
            if(pKF->GetMap() == pKFi->GetMap() && vpLoopCand.size() < nNumCandidates)
            {
                vpLoopCand.push_back(pKFi);
            }
            else if(pKF->GetMap() != pKFi->GetMap() && vpMergeCand.size() < nNumCandidates && !pKFi->GetMap()->IsBad())
            {
                vpMergeCand.push_back(pKFi);
            }
            spAlreadyAddedKF.insert(pKFi);
        }
        i++;
        it++;
    }
    //-------

    // Return all those keyframes with a score higher than 0.75*bestScore
    /*float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vpLoopCand.reserve(lAccScoreAndMatch.size());
    vpMergeCand.reserve(lAccScoreAndMatch.size());
    for(list<pair<float, KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                if(pKF->GetMap() == pKFi->GetMap())
                {
                    vpLoopCand.push_back(pKFi);
                }
                else
                {
                    vpMergeCand.push_back(pKFi);
                }
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }*/
}

// 从地图中选择一些跟帧F相关度比较高的候选关键帧，对这个函数已跪，太绝了。。
/**
 * @brief 在重定位中找到与该帧相似的关键帧
 *
 * 1. 找出和当前帧具有公共单词的所有关键帧
 * 2. 只和具有共同单词较多的关键帧进行相似度计算
 * 3. 将与关键帧相连（权值最高）的前十个关键帧归为一组，计算累计得分
 * 4. 只返回累计得分较高的组中分数最高的关键帧
 * @param F 需要重定位的帧
 * @param pMap 选择地图
 * @return  相似的关键帧
 * @see III-E Bags of Words Place Recognition
 */
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F, Map* pMap)
{
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    // 步骤1：找出和当前帧具有公共单词的所有关键帧
    {
        unique_lock<mutex> lock(mMutex);
        // 遍历当前帧所包含的词汇
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            // 从mvInvertedFile里取出拥有这个词汇的关键帧们
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];
            // 遍历这些关键帧
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery != F->mnId)  // mnRelocQuery主要是查看是否已经被F帧标记成了候选关键帧
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);  // 存入跟F有相同词汇的关键帧
                }
                pKFi->mnRelocWords++;  // 统计相同词汇的数量
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    // 步骤2：统计所有闭环候选帧中与当前帧F具有共同单词最多的单词数，并以此决定阈值
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend=lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }
    // 取20%的lKFsSharingWords
    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float, KeyFrame*> > lScoreAndMatch;

    int nscores = 0;

    // Compute similarity score.
    // 步骤3：遍历所有闭环候选帧，挑选出共有单词数大于阈值minCommonWords且单词匹配度大于minScore存入lScoreAndMatch
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend=lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        // 当前帧F只和具有共同单词较多的关键帧进行比较，需要大于minCommonWords
        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
            pKFi->mRelocScore = si;  // 相似度赋值
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float, KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    // 步骤4：计算候选帧组得分，得到最高组得分bestAccScore，并以此决定阈值minScoreToRetain
    // 单单计算当前帧和某一关键帧的相似性是不够的，这里将与关键帧相连（权值最高，共视程度最高）的前十个关键帧归为一组，计算累计得分
    // 具体而言：lScoreAndMatch中每一个KeyFrame都把与自己共视程度较高的帧归为一组，每一组会计算组得分并记录该组分数最高的KeyFrame，记录于lAccScoreAndMatch
    for(list<pair<float, KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);  // 与lAccScoreAndMatch里面每一帧共视度最高的10个

        float bestScore = it->first;  // 每一组最高分数
        float accScore = bestScore;   // 每一组累计分数
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery != F->mnId)  // 如果F与lAccScoreAndMatch里面的帧的共视帧并没有共同的词汇，跳过
                continue;

            accScore+=pKF2->mRelocScore;  // 计算累计分数
            if(pKF2->mRelocScore>bestScore)  // 重新确定这10+1帧里面与F得分最高的帧
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));  // 添加
        if(accScore>bestAccScore)
            bestAccScore=accScore;  // 确定分数
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    // 步骤5：得到组得分大于阈值的，组内得分最高的关键帧
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float, KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        // 只返回累计得分大于minScoreToRetain的组中分数最高的关键帧 0.75*bestScore
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;  // 地图得是一个～
            if (pKFi->GetMap() != pMap)
                continue;
            if(!spAlreadyAddedKF.count(pKFi))  // 判断该pKFi是否已经在队列中了
            {
                vpRelocCandidates.push_back(pKFi);  // 最终返回的关键帧，有可能是空的，那定位将会失败
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

void KeyFrameDatabase::PreSave()
{
    //Save the information about the inverted index of KF to node
    mvBackupInvertedFileId.resize(mvInvertedFile.size());
    for(int i = 0, numEl = mvInvertedFile.size(); i < numEl; ++i)
    {
        for(std::list<KeyFrame*>::const_iterator it = mvInvertedFile[i].begin(), end = mvInvertedFile[i].end(); it != end; ++it)
        {
            mvBackupInvertedFileId[i].push_back((*it)->mnId);
        }
    }
}

void KeyFrameDatabase::PostLoad(map<long unsigned int, KeyFrame*> mpKFid)
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
    for(unsigned int i = 0; i < mvBackupInvertedFileId.size(); ++i)
    {
        for(long unsigned int KFid : mvBackupInvertedFileId[i])
        {
            if(mpKFid.find(KFid) != mpKFid.end())
            {
                mvInvertedFile[i].push_back(mpKFid[KFid]);
            }
        }
    }

}

void KeyFrameDatabase::SetORBVocabulary(ORBVocabulary* pORBVoc)
{
    ORBVocabulary** ptr;
    ptr = (ORBVocabulary**)( &mpVoc );
    *ptr = pORBVoc;

    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

} //namespace ORB_SLAM
