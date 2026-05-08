/**
* ORB-SLAM3 example for the TartanAir dataset (monocular, left camera).
*
* Dataset structure expected:
*   <sequence_root>/image_left/   — images named 000000.png, 000001.png, …
*   <sequence_root>/imu/cam_time.txt  — optional; one timestamp (seconds) per line.
*                                        If absent, 10 Hz is assumed.
*
* Usage:
*   mono_tartanair path_to_vocabulary path_to_settings path_to_sequence [trajectory_name]
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<string>
#include<vector>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strImagePath, const string &strTimesPath,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{
    if(argc < 4 || argc > 5)
    {
        cerr << "\nUsage: ./mono_tartanair path_to_vocabulary path_to_settings"
                " path_to_sequence [trajectory_name]\n"
             << "  path_to_sequence : folder containing image_left/ "
                "and optionally imu/cam_time.txt\n";
        return 1;
    }

    const bool bFileName = (argc == 5);
    const string file_name = bFileName ? string(argv[4]) : "";

    vector<string> vstrImages;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]) + "/image_left",
               string(argv[3]) + "/imu/cam_time.txt",
               vstrImages, vTimestamps);

    const int nImages = (int)vstrImages.size();
    if(nImages == 0){ cerr << "No images found in " << string(argv[3]) << "/image_left\n"; return 1; }
    cout << "Images: " << nImages << endl;

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    vector<float> vTimesTrack(nImages);
    cv::Mat im;
    for(int ni = 0; ni < nImages; ni++)
    {
        im = cv::imread(vstrImages[ni], cv::IMREAD_UNCHANGED);
        if(im.empty()){ cerr << "Failed to load: " << vstrImages[ni] << "\n"; return 1; }
        if(imageScale != 1.f)
            cv::resize(im, im, cv::Size((int)(im.cols*imageScale), (int)(im.rows*imageScale)));

        const double tframe = vTimestamps[ni];

#ifdef COMPILEDWITHC11
        auto t1 = std::chrono::steady_clock::now();
#else
        auto t1 = std::chrono::monotonic_clock::now();
#endif
        SLAM.TrackMonocular(im, tframe);
#ifdef COMPILEDWITHC11
        auto t2 = std::chrono::steady_clock::now();
#else
        auto t2 = std::chrono::monotonic_clock::now();
#endif
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
        vTimesTrack[ni] = (float)ttrack;

        double T = (ni < nImages-1) ? vTimestamps[ni+1] - tframe
                 : (ni > 0)         ? tframe - vTimestamps[ni-1]
                 :                    0.1;
        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    SLAM.Shutdown();

    sort(vTimesTrack.begin(), vTimesTrack.end());
    float total = 0.f;
    for(int ni = 0; ni < nImages; ni++) total += vTimesTrack[ni];
    cout << "-------\nmedian tracking time: " << vTimesTrack[nImages/2]
         << "\nmean tracking time: " << total/nImages << "\n";

    if(bFileName)
    {
        SLAM.SaveTrajectoryTUM("f_"  + file_name + ".txt");
        SLAM.SaveKeyFrameTrajectoryTUM("kf_" + file_name + ".txt");
    }
    else
    {
        SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    }
    return 0;
}

void LoadImages(const string &strImagePath, const string &strTimesPath,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    // Collect sorted .png images via OpenCV glob
    vector<cv::String> files;
    cv::glob(strImagePath + "/*.png", files, false);
    sort(files.begin(), files.end());
    vstrImages.assign(files.begin(), files.end());

    const int N = (int)vstrImages.size();
    vTimeStamps.resize(N);

    ifstream fTimes(strTimesPath.c_str());
    if(fTimes.is_open())
    {
        for(int i = 0; i < N && !fTimes.eof(); i++)
        {
            string s;
            getline(fTimes, s);
            if(!s.empty()) vTimeStamps[i] = stod(s);
        }
        fTimes.close();
    }
    else
    {
        // No timestamp file: assume 10 Hz
        for(int i = 0; i < N; i++)
            vTimeStamps[i] = i * 0.1;
    }
}
