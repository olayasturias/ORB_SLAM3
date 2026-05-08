/**
* ORB-SLAM3 example for the VBR (Visual-Biometric Reference) dataset (stereo).
*
* Dataset structure expected:
*   <sequence_root>/camera_left/data/        — left images *.png (sorted)
*   <sequence_root>/camera_right/data/       — right images *.png (sorted)
*   <sequence_root>/camera_left/timestamps.txt — nanosecond integer timestamps, one per line
*
* Usage:
*   stereo_vbr path_to_vocabulary path_to_settings path_to_sequence [trajectory_name]
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<string>
#include<vector>
#include<cstdint>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strPathLeft, const string &strPathRight,
                const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight,
                vector<double> &vTimeStamps);

int main(int argc, char **argv)
{
    if(argc < 4 || argc > 5)
    {
        cerr << "\nUsage: ./stereo_vbr path_to_vocabulary path_to_settings"
                " path_to_sequence [trajectory_name]\n"
             << "  path_to_sequence : folder containing camera_left/ and camera_right/\n";
        return 1;
    }

    const bool bFileName = (argc == 5);
    const string file_name = bFileName ? string(argv[4]) : "";

    string seqPath(argv[3]);
    vector<string> vstrImageLeft, vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(seqPath + "/camera_left/data",
               seqPath + "/camera_right/data",
               seqPath + "/camera_left/timestamps.txt",
               vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = (int)vstrImageLeft.size();
    if(nImages == 0){ cerr << "No images found in " << seqPath << "\n"; return 1; }
    cout << "Images: " << nImages << endl;

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, true);
    float imageScale = SLAM.GetImageScale();

    vector<float> vTimesTrack(nImages);
    cv::Mat imLeft, imRight;
    for(int ni = 0; ni < nImages; ni++)
    {
        imLeft  = cv::imread(vstrImageLeft[ni],  cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED);
        if(imLeft.empty()) { cerr << "Failed to load: " << vstrImageLeft[ni]  << "\n"; return 1; }
        if(imRight.empty()){ cerr << "Failed to load: " << vstrImageRight[ni] << "\n"; return 1; }

        if(imageScale != 1.f)
        {
            cv::Size sz((int)(imLeft.cols*imageScale), (int)(imLeft.rows*imageScale));
            cv::resize(imLeft,  imLeft,  sz);
            cv::resize(imRight, imRight, sz);
        }

        const double tframe = vTimestamps[ni];

#ifdef COMPILEDWITHC11
        auto t1 = std::chrono::steady_clock::now();
#else
        auto t1 = std::chrono::monotonic_clock::now();
#endif
        SLAM.TrackStereo(imLeft, imRight, tframe);
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

void LoadImages(const string &strPathLeft, const string &strPathRight,
                const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight,
                vector<double> &vTimeStamps)
{
    vector<cv::String> filesL, filesR;
    cv::glob(strPathLeft  + "/*.png", filesL, false);
    cv::glob(strPathRight + "/*.png", filesR, false);
    sort(filesL.begin(), filesL.end());
    sort(filesR.begin(), filesR.end());

    const int N = (int)min(filesL.size(), filesR.size());
    vstrImageLeft.assign(filesL.begin(),  filesL.begin() + N);
    vstrImageRight.assign(filesR.begin(), filesR.begin() + N);
    vTimeStamps.resize(N);

    ifstream fTimes(strPathTimes.c_str());
    if(fTimes.is_open())
    {
        for(int i = 0; i < N; i++)
        {
            string line;
            if(!getline(fTimes, line) || line.empty()){ vTimeStamps[i] = i * 0.1; continue; }
            try
            {
                // VBR timestamps are nanosecond integers
                int64_t ns = stoll(line);
                vTimeStamps[i] = ns * 1e-9;
            }
            catch(...)
            {
                vTimeStamps[i] = i * 0.1;
            }
        }
        fTimes.close();
    }
    else
    {
        for(int i = 0; i < N; i++) vTimeStamps[i] = i * 0.1;
    }
}
