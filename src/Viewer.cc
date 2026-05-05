/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
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


#include "Viewer.h"
#include <rerun.hpp>

#include <mutex>
#include <set>
#include <thread>
#include <chrono>
#include <vector>

namespace ORB_SLAM3
{

Viewer::~Viewer() = default;

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath, Settings* settings):
    mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false),
    mrec(std::make_unique<rerun::RecordingStream>("orb_slam3"))
{
    mrec->spawn().exit_on_failure();
    if(settings){
        newParameterLoader(settings);
    }
    else{

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

}

void Viewer::newParameterLoader(Settings *settings) {
    float fps = settings->fps();
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::Size imSize = settings->newImSize();
    mImageHeight = imSize.height;
    mImageWidth = imSize.width;
}

bool Viewer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::FileNode node = fSettings["Camera.width"];
    if(!node.empty())
    {
        mImageWidth = node.real();
    }
    else
    {
        std::cerr << "*Camera.width parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Camera.height"];
    if(!node.empty())
    {
        mImageHeight = node.real();
    }
    else
    {
        std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    std::vector<std::array<float, 3>> mPath;
    std::set<long unsigned int> mLoggedKFBias;
    std::string mLastStatus;

    cout << "Starting the Viewer" << endl;
    while(1)
    {
        Eigen::Matrix4f Twc = mpMapDrawer->GetCurrentPose().matrix();

        mrec->log("world/camera/image",
            rerun::archetypes::Pinhole(
                rerun::components::PinholeProjection(std::array<float,9>{
                    Frame::fx,  0.f,        0.f,
                    0.f,        Frame::fy,  0.f,
                    Frame::cx,  Frame::cy,  1.f
                })
            ).with_resolution((float)mImageWidth, (float)mImageHeight)
        );

        mrec->log("world/camera",
            rerun::archetypes::Transform3D(
                rerun::components::Translation3D(Twc(0,3), Twc(1,3), Twc(2,3)),
                rerun::components::TransformMat3x3(std::array<float,9>{
                    Twc(0,0), Twc(1,0), Twc(2,0),
                    Twc(0,1), Twc(1,1), Twc(2,1),
                    Twc(0,2), Twc(1,2), Twc(2,2)
                })
            )
        );

        if(mpTracker->mLastProcessedState == Tracking::OK)
        {
            mPath.push_back({Twc(0,3), Twc(1,3), Twc(2,3)});
            if(mPath.size() >= 2)
                mrec->log("world/path", rerun::LineStrips3D(rerun::components::LineStrip3D(
                    rerun::Collection<rerun::datatypes::Vec3D>::borrow(mPath)
                )));
        }

        // Tracking status text
        {
            std::string status = mpFrameDrawer->GetStatusString();
            if(status != mLastStatus)
            {
                rerun::TextLogLevel level = (status.find("LOST") != std::string::npos)
                    ? rerun::TextLogLevel::Warning
                    : rerun::TextLogLevel::Info;
                mrec->log("slam/status", rerun::TextLog(status).with_level(level));
                mLastStatus = status;
            }
        }

        Map* pMap = mpMapDrawer->mpAtlas->GetCurrentMap();
        if(pMap)
        {
            cv::Mat colorIm = mpFrameDrawer->GetRawImage();

            Eigen::Matrix3f Rcw = Twc.block<3,3>(0,0).transpose();
            Eigen::Vector3f tcw = -Rcw * Twc.block<3,1>(0,3);

            const vector<MapPoint*>& vpMPs    = pMap->GetAllMapPoints();
            const vector<MapPoint*>& vpRefMPs = pMap->GetReferenceMapPoints();
            set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

            std::vector<rerun::components::Position3D> globalPts, activePts;
            std::vector<rerun::Color> globalColors;
            globalPts.reserve(vpMPs.size());
            globalColors.reserve(vpMPs.size());
            activePts.reserve(vpRefMPs.size());

            for(MapPoint* mp : vpMPs)
            {
                if(!mp || mp->isBad()) continue;
                Eigen::Vector3f Xw = mp->GetWorldPos();

                if(spRefMPs.count(mp))
                {
                    activePts.push_back({Xw(0), Xw(1), Xw(2)});
                }
                else
                {
                    rerun::Color ptColor(128, 128, 128);
                    if(!colorIm.empty())
                    {
                        Eigen::Vector3f Xc = Rcw * Xw + tcw;
                        if(Xc(2) > 0.f)
                        {
                            int iu = (int)std::round(Frame::fx * Xc(0) / Xc(2) + Frame::cx);
                            int iv = (int)std::round(Frame::fy * Xc(1) / Xc(2) + Frame::cy);
                            if(iu >= 0 && iu < colorIm.cols && iv >= 0 && iv < colorIm.rows)
                            {
                                if(colorIm.channels() == 1)
                                {
                                    uchar g = colorIm.at<uchar>(iv, iu);
                                    ptColor = rerun::Color(g, g, g);
                                }
                                else
                                {
                                    cv::Vec3b bgr = colorIm.at<cv::Vec3b>(iv, iu);
                                    ptColor = rerun::Color(bgr[2], bgr[1], bgr[0]);
                                }
                            }
                        }
                    }
                    globalPts.push_back({Xw(0), Xw(1), Xw(2)});
                    globalColors.push_back(ptColor);
                }
            }
            mrec->log("world/map/global_map/points",
                rerun::Points3D(globalPts).with_colors(globalColors));
            mrec->log("world/map/active_map/points",
                rerun::Points3D(activePts).with_colors(rerun::Color(0, 255, 0)));

            const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

            std::vector<rerun::components::LineStrip3D> covisLines, treeLines, loopLines, inertialLines;
            std::vector<rerun::datatypes::Vec3D> velOrigins, velVectors;

            for(KeyFrame* pKF : vpKFs)
            {
                if(!pKF || pKF->isBad()) continue;
                Eigen::Vector3f Ow = pKF->GetCameraCenter();

                for(KeyFrame* pKF2 : pKF->GetCovisiblesByWeight(100))
                {
                    if(pKF2->mnId < pKF->mnId || pKF2->isBad()) continue;
                    Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
                    covisLines.push_back(rerun::components::LineStrip3D(
                        std::vector<rerun::datatypes::Vec3D>{
                            {Ow(0),Ow(1),Ow(2)}, {Ow2(0),Ow2(1),Ow2(2)}}));
                }

                KeyFrame* pParent = pKF->GetParent();
                if(pParent && !pParent->isBad())
                {
                    Eigen::Vector3f Owp = pParent->GetCameraCenter();
                    treeLines.push_back(rerun::components::LineStrip3D(
                        std::vector<rerun::datatypes::Vec3D>{
                            {Ow(0),Ow(1),Ow(2)}, {Owp(0),Owp(1),Owp(2)}}));
                }

                for(KeyFrame* pKFl : pKF->GetLoopEdges())
                {
                    if(pKFl->mnId < pKF->mnId || pKFl->isBad()) continue;
                    Eigen::Vector3f Owl = pKFl->GetCameraCenter();
                    loopLines.push_back(rerun::components::LineStrip3D(
                        std::vector<rerun::datatypes::Vec3D>{
                            {Ow(0),Ow(1),Ow(2)}, {Owl(0),Owl(1),Owl(2)}}));
                }

                KeyFrame* pNext = pKF->mNextKF;
                if(pNext && !pNext->isBad())
                {
                    Eigen::Vector3f Owp = pNext->GetCameraCenter();
                    inertialLines.push_back(rerun::components::LineStrip3D(
                        std::vector<rerun::datatypes::Vec3D>{
                            {Ow(0),Ow(1),Ow(2)}, {Owp(0),Owp(1),Owp(2)}}));
                }

                if(!mLoggedKFBias.count(pKF->mnId))
                {
                    if(pKF->bImu)
                    {
                        IMU::Bias b = pKF->GetImuBias();
                        float gyroNorm = std::sqrt(b.bwx*b.bwx + b.bwy*b.bwy + b.bwz*b.bwz);
                        float accNorm  = std::sqrt(b.bax*b.bax + b.bay*b.bay + b.baz*b.baz);
                        mrec->set_time_seconds("slam_time", pKF->mTimeStamp);
                        mrec->log("imu/bias/gyro_norm",  rerun::Scalars(gyroNorm));
                        mrec->log("imu/bias/accel_norm", rerun::Scalars(accNorm));
                        mrec->reset_time();
                    }
                    mLoggedKFBias.insert(pKF->mnId);
                }

                if(pKF->bImu)
                {
                    Sophus::SE3f Twb = pKF->GetImuPose();
                    Eigen::Matrix4f TwbM = Twb.matrix();
                    mrec->log("world/imu/" + std::to_string(pKF->mnId),
                        rerun::archetypes::Transform3D(
                            rerun::components::Translation3D(TwbM(0,3), TwbM(1,3), TwbM(2,3)),
                            rerun::components::TransformMat3x3(std::array<float,9>{
                                TwbM(0,0), TwbM(1,0), TwbM(2,0),
                                TwbM(0,1), TwbM(1,1), TwbM(2,1),
                                TwbM(0,2), TwbM(1,2), TwbM(2,2)
                            })
                        )
                    );
                }

                if(pKF->isVelocitySet())
                {
                    Eigen::Vector3f vel = pKF->GetVelocity();
                    velOrigins.push_back({Ow(0), Ow(1), Ow(2)});
                    velVectors.push_back({vel(0), vel(1), vel(2)});
                }
            }

            mrec->log("world/graph/covisibility",
                rerun::LineStrips3D(covisLines).with_colors(rerun::Color(0, 255, 0)));
            mrec->log("world/graph/spanning_tree",
                rerun::LineStrips3D(treeLines).with_colors(rerun::Color(0, 200, 0)));
            mrec->log("world/graph/loops",
                rerun::LineStrips3D(loopLines).with_colors(rerun::Color(255, 165, 0)));
            if(pMap->isImuInitialized())
                mrec->log("world/graph/inertial",
                    rerun::LineStrips3D(inertialLines).with_colors(rerun::Color(255, 0, 0)));
            mrec->log("world/keyframes/velocities",
                rerun::Arrows3D::from_vectors(velVectors)
                    .with_origins(velOrigins)
                    .with_colors(rerun::Color(0, 200, 255)));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds((int)mT));

        cv::Mat rawIm = mpFrameDrawer->GetRawImage();
        if (!rawIm.empty())
        {
            cv::Mat rgb;
            if(rawIm.channels() == 1)
                cv::cvtColor(rawIm, rgb, cv::COLOR_GRAY2RGB);
            else
                cv::cvtColor(rawIm, rgb, cv::COLOR_BGR2RGB);
            uint32_t h = static_cast<uint32_t>(rgb.rows);
            uint32_t w = static_cast<uint32_t>(rgb.cols);
            std::vector<uint8_t> imgData(rgb.data, rgb.data + w * h * 3);
            mrec->log("world/camera/image/rgb", rerun::Image::from_rgb24(imgData, {w, h}));
        }

        auto keypoints = mpFrameDrawer->GetCurrentKeypoints();
        std::vector<rerun::components::Position2D> kptPositions;
        kptPositions.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            kptPositions.push_back({kp.pt.x, kp.pt.y});
        }
        mrec->log("world/camera/image/rgb/kpts", rerun::Points2D(kptPositions));

        if(Stop())
        {
            while(isStopped())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }

        if(CheckFinish())
            break;
    }

    SetFinish();
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

}
