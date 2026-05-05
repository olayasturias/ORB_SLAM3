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
#include <pangolin/pangolin.h>
#include <rerun.hpp>

#include <mutex>
#include <set>
#include <vector>

namespace ORB_SLAM3
{

Viewer::~Viewer() = default;

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath, Settings* settings):
    both(false), mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
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

    mbStopTrack = false;
}

void Viewer::newParameterLoader(Settings *settings) {
    mImageViewerScale = 1.f;

    float fps = settings->fps();
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::Size imSize = settings->newImSize();
    mImageHeight = imSize.height;
    mImageWidth = imSize.width;

    mImageViewerScale = settings->imageViewerScale();
    mViewpointX = settings->viewPointX();
    mViewpointY = settings->viewPointY();
    mViewpointZ = settings->viewPointZ();
    mViewpointF = settings->viewPointF();
}

bool Viewer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    mImageViewerScale = 1.f;

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

    node = fSettings["Viewer.imageViewScale"];
    if(!node.empty())
    {
        mImageViewerScale = node.real();
    }

    node = fSettings["Viewer.ViewpointX"];
    if(!node.empty())
    {
        mViewpointX = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointX parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointY"];
    if(!node.empty())
    {
        mViewpointY = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointY parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointZ"];
    if(!node.empty())
    {
        mViewpointZ = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointZ parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointF"];
    if(!node.empty())
    {
        mViewpointF = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointF parameter doesn't exist or is not a real number*" << std::endl;
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

    pangolin::CreateWindowAndBind("ORB-SLAM3: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",false,true);
    pangolin::Var<bool> menuCamView("menu.Camera View",false,false);
    pangolin::Var<bool> menuTopView("menu.Top View",false,false);
    // pangolin::Var<bool> menuSideView("menu.Side View",false,false);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",false,true);
    pangolin::Var<bool> menuShowInertialGraph("menu.Show Inertial Graph",true,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);
    pangolin::Var<bool> menuStop("menu.Stop",false,false);
    pangolin::Var<bool> menuStepByStep("menu.Step By Step",false,true);  // false, true
    pangolin::Var<bool> menuStep("menu.Step",false,false);

    pangolin::Var<bool> menuShowOptLba("menu.Show LBA opt", false, true);
    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();
    cv::namedWindow("ORB-SLAM3: Current Frame");

    bool bFollow = true;
    bool bLocalizationMode = false;
    bool bStepByStep = false;
    bool bCameraView = true;

    if(mpTracker->mSensor == mpSystem->MONOCULAR || mpTracker->mSensor == mpSystem->STEREO || mpTracker->mSensor == mpSystem->RGBD)
    {
        menuShowGraph = true;
    }

    float trackedImageScale = mpTracker->GetImageScale();

    cout << "Starting the Viewer" << endl;
    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc,Ow);

        // K matrix column-major: col0=(fx,0,0), col1=(0,fy,0), col2=(cx,cy,1)
        mrec->log("world/camera/image",
            rerun::archetypes::Pinhole(
                rerun::components::PinholeProjection(std::array<float,9>{
                    Frame::fx,  0.f,        0.f,
                    0.f,        Frame::fy,  0.f,
                    Frame::cx,  Frame::cy,  1.f
                })
            ).with_resolution((float)mImageWidth, (float)mImageHeight)
        );

        // Twc is column-major (pangolin): m[0..2]=col0, m[4..6]=col1, m[8..10]=col2, m[12..14]=t
        mrec->log("world/camera",
            rerun::archetypes::Transform3D(
                rerun::components::Translation3D(
                    (float)Twc.m[12], (float)Twc.m[13], (float)Twc.m[14]
                ),
                rerun::components::TransformMat3x3(std::array<float,9>{
                    (float)Twc.m[0], (float)Twc.m[1], (float)Twc.m[2],
                    (float)Twc.m[4], (float)Twc.m[5], (float)Twc.m[6],
                    (float)Twc.m[8], (float)Twc.m[9], (float)Twc.m[10]
                })
            )
        );

        if(mpTracker->mLastProcessedState == Tracking::OK)
        {
            mPath.push_back({(float)Twc.m[12], (float)Twc.m[13], (float)Twc.m[14]});
            if(mPath.size() >= 2)
                mrec->log("world/path", rerun::LineStrips3D(rerun::components::LineStrip3D(
                    rerun::Collection<rerun::datatypes::Vec3D>::borrow(mPath)
                )));
        }

        if(mbStopTrack)
        {
            menuStepByStep = true;
            mbStopTrack = false;
        }

        if(menuFollowCamera && bFollow)
        {
            if(bCameraView)
                s_cam.Follow(Twc);
            else
                s_cam.Follow(Ow);
        }
        else if(menuFollowCamera && !bFollow)
        {
            if(bCameraView)
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
            }
            else
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,10, 0,0,0,0.0,0.0, 1.0));
                s_cam.Follow(Ow);
            }
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        if(menuCamView)
        {
            menuCamView = false;
            bCameraView = true;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
        }

        if(menuTopView && mpMapDrawer->mpAtlas->isImuInitialized())
        {
            menuTopView = false;
            bCameraView = false;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,50, 0,0,0,0.0,0.0, 1.0));
            s_cam.Follow(Ow);
        }

        if(menuLocalizationMode && !bLocalizationMode)
        {
            mpSystem->ActivateLocalizationMode();
            bLocalizationMode = true;
        }
        else if(!menuLocalizationMode && bLocalizationMode)
        {
            mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }

        if(menuStepByStep && !bStepByStep)
        {
            //cout << "Viewer: step by step" << endl;
            mpTracker->SetStepByStep(true);
            bStepByStep = true;
        }
        else if(!menuStepByStep && bStepByStep)
        {
            mpTracker->SetStepByStep(false);
            bStepByStep = false;
        }

        if(menuStep)
        {
            mpTracker->mbStep = true;
            menuStep = false;
        }


        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        mpMapDrawer->DrawCurrentCamera(Twc);
        if(menuShowKeyFrames || menuShowGraph || menuShowInertialGraph || menuShowOptLba)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph, menuShowInertialGraph, menuShowOptLba);
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();

        pangolin::FinishFrame();

        // Rerun: log all map entities unconditionally (independent of Pangolin menu state)
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

            Map* pMap = mpMapDrawer->mpAtlas->GetCurrentMap();
            if(pMap)
            {
                // Map points colored by image projection
                cv::Mat colorIm = mpFrameDrawer->GetRawImage();

                // Derive Rcw, tcw from the Pangolin Twc (column-major)
                Eigen::Matrix3f Rwc;
                Rwc(0,0)=(float)Twc.m[0]; Rwc(0,1)=(float)Twc.m[4]; Rwc(0,2)=(float)Twc.m[8];
                Rwc(1,0)=(float)Twc.m[1]; Rwc(1,1)=(float)Twc.m[5]; Rwc(1,2)=(float)Twc.m[9];
                Rwc(2,0)=(float)Twc.m[2]; Rwc(2,1)=(float)Twc.m[6]; Rwc(2,2)=(float)Twc.m[10];
                Eigen::Matrix3f Rcw = Rwc.transpose();
                Eigen::Vector3f tcw = -Rcw * Eigen::Vector3f((float)Twc.m[12], (float)Twc.m[13], (float)Twc.m[14]);

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

                // Single pass over all keyframes: graph edges, velocity, body frame, bias
                const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

                std::vector<rerun::components::LineStrip3D> covisLines, treeLines, loopLines, inertialLines;
                std::vector<rerun::datatypes::Vec3D> velOrigins, velVectors;

                for(KeyFrame* pKF : vpKFs)
                {
                    if(!pKF || pKF->isBad()) continue;
                    Eigen::Vector3f Ow = pKF->GetCameraCenter();

                    // Covisibility edges
                    for(KeyFrame* pKF2 : pKF->GetCovisiblesByWeight(100))
                    {
                        if(pKF2->mnId < pKF->mnId || pKF2->isBad()) continue;
                        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
                        covisLines.push_back(rerun::components::LineStrip3D(
                            std::vector<rerun::datatypes::Vec3D>{
                                {Ow(0),Ow(1),Ow(2)}, {Ow2(0),Ow2(1),Ow2(2)}}));
                    }

                    // Spanning tree
                    KeyFrame* pParent = pKF->GetParent();
                    if(pParent && !pParent->isBad())
                    {
                        Eigen::Vector3f Owp = pParent->GetCameraCenter();
                        treeLines.push_back(rerun::components::LineStrip3D(
                            std::vector<rerun::datatypes::Vec3D>{
                                {Ow(0),Ow(1),Ow(2)}, {Owp(0),Owp(1),Owp(2)}}));
                    }

                    // Loop edges
                    for(KeyFrame* pKFl : pKF->GetLoopEdges())
                    {
                        if(pKFl->mnId < pKF->mnId || pKFl->isBad()) continue;
                        Eigen::Vector3f Owl = pKFl->GetCameraCenter();
                        loopLines.push_back(rerun::components::LineStrip3D(
                            std::vector<rerun::datatypes::Vec3D>{
                                {Ow(0),Ow(1),Ow(2)}, {Owl(0),Owl(1),Owl(2)}}));
                    }

                    // Inertial chain
                    KeyFrame* pNext = pKF->mNextKF;
                    if(pNext && !pNext->isBad())
                    {
                        Eigen::Vector3f Owp = pNext->GetCameraCenter();
                        inertialLines.push_back(rerun::components::LineStrip3D(
                            std::vector<rerun::datatypes::Vec3D>{
                                {Ow(0),Ow(1),Ow(2)}, {Owp(0),Owp(1),Owp(2)}}));
                    }

                    // IMU bias — log once per KF at its timestamp
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

                    // IMU body frame (separate tree — avoids transform inheritance with camera)
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

                    // Velocity arrows
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
        }

        cv::Mat toShow;
        cv::Mat im = mpFrameDrawer->DrawFrame(trackedImageScale);

        if(both){
            cv::Mat imRight = mpFrameDrawer->DrawRightFrame(trackedImageScale);
            cv::hconcat(im,imRight,toShow);
        }
        else{
            toShow = im;
        }

        if(mImageViewerScale != 1.f)
        {
            int width = toShow.cols * mImageViewerScale;
            int height = toShow.rows * mImageViewerScale;
            cv::resize(toShow, toShow, cv::Size(width, height));
        }

        cv::imshow("ORB-SLAM3: Current Frame",toShow);
        cv::waitKey(mT);
        cv::Mat rawIm = mpFrameDrawer->GetRawImage();
        if (!rawIm.empty())
        {
            cv::Mat rgb;
            cv::cvtColor(rawIm, rgb, cv::COLOR_BGR2RGB);
            uint32_t height = static_cast<uint32_t>(rgb.rows);
            uint32_t width  = static_cast<uint32_t>(rgb.cols);
            std::vector<uint8_t> imgData(rgb.data, rgb.data + width * height * 3);
            mrec->log("world/camera/image/rgb", rerun::Image::from_rgb24(imgData, {width, height}));
        }

        auto keypoints = mpFrameDrawer->GetCurrentKeypoints();
        std::vector<rerun::components::Position2D> kptPositions;
        kptPositions.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            kptPositions.push_back({kp.pt.x, kp.pt.y});
        }

        mrec->log("world/camera/image/rgb/kpts", rerun::Points2D(kptPositions));

        if(menuReset)
        {
            menuShowGraph = true;
            menuShowInertialGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            mpSystem->ResetActiveMap();
            menuReset = false;
        }

        if(menuStop)
        {
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();

            // Stop all threads
            mpSystem->Shutdown();

            // Save camera trajectory
            mpSystem->SaveTrajectoryEuRoC("CameraTrajectory.txt");
            mpSystem->SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
            menuStop = false;
        }

        if(Stop())
        {
            while(isStopped())
            {
                usleep(3000);
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

/*void Viewer::SetTrackingPause()
{
    mbStopTrack = true;
}*/

}
