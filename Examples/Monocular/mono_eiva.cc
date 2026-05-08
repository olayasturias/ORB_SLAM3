/**
* ORB-SLAM3 example for the EIVA dataset (monocular, left camera).
*
* Dataset structure expected:
*   <sequence_root>/processed/left/   — images *.jpg (Zephyr filenames)
*
* Zephyr filename formats (both handled):
*   YYYY-MM-DDTHHMMSS.fffZ.jpg        — compact time, UTC Z suffix
*   YYYY-MM-DDTHH-MM-SS-fff.jpg       — dashed time with milliseconds
*
* Usage:
*   mono_eiva path_to_vocabulary path_to_settings path_to_sequence [trajectory_name]
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<string>
#include<vector>
#include<ctime>
#include<cstdio>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

static double parseEIVATimestamp(const string &stem)
{
    auto tpos = stem.find('T');
    if(tpos == string::npos) return 0.0;

    int yr = 0, mo = 0, dy = 0, hr = 0, mn = 0, sc = 0;
    if(sscanf(stem.c_str(), "%d-%d-%d", &yr, &mo, &dy) != 3) return 0.0;

    string tp = stem.substr(tpos + 1);
    if(!tp.empty() && tp.back() == 'Z') tp.pop_back();

    double frac = 0.0;
    if(tp.size() >= 8 && tp[2] == '-')
    {
        int ms = 0;
        sscanf(tp.c_str(), "%d-%d-%d-%d", &hr, &mn, &sc, &ms);
        frac = ms * 1e-3;
    }
    else
    {
        if(tp.size() < 6) return 0.0;
        hr = stoi(tp.substr(0, 2));
        mn = stoi(tp.substr(2, 2));
        sc = stoi(tp.substr(4, 2));
        auto dot = tp.find('.');
        if(dot != string::npos) frac = stod(tp.substr(dot));
    }

    struct tm t{};
    t.tm_year = yr - 1900; t.tm_mon = mo - 1; t.tm_mday = dy;
    t.tm_hour = hr; t.tm_min = mn; t.tm_sec = sc; t.tm_isdst = -1;
    return static_cast<double>(mktime(&t)) + frac;
}

void LoadImages(const string &strImagePath,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{
    if(argc < 4 || argc > 5)
    {
        cerr << "\nUsage: ./mono_eiva path_to_vocabulary path_to_settings"
                " path_to_sequence [trajectory_name]\n"
             << "  path_to_sequence : folder containing processed/left/\n";
        return 1;
    }

    const bool bFileName = (argc == 5);
    const string file_name = bFileName ? string(argv[4]) : "";

    vector<string> vstrImages;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]) + "/processed/left", vstrImages, vTimestamps);

    const int nImages = (int)vstrImages.size();
    if(nImages == 0){ cerr << "No images found in " << string(argv[3]) << "/processed/left\n"; return 1; }
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

void LoadImages(const string &strImagePath,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    vector<cv::String> files;
    cv::glob(strImagePath + "/*.jpg", files, false);
    sort(files.begin(), files.end());
    vstrImages.assign(files.begin(), files.end());

    const int N = (int)vstrImages.size();
    vTimeStamps.resize(N);
    for(int i = 0; i < N; i++)
    {
        string path = files[i];
        auto slash = path.find_last_of("/\\");
        string fname = (slash != string::npos) ? path.substr(slash + 1) : path;
        auto dot = fname.rfind('.');
        string stem = (dot != string::npos) ? fname.substr(0, dot) : fname;
        vTimeStamps[i] = parseEIVATimestamp(stem);
    }
}
