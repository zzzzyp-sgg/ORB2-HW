#include "ORB/global_defination/global_defination.h"
#include "ORB/ORBFeature.hpp"

using namespace ORB;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    std::string config_path = WORK_SPACE_PATH + "/config/camera_para.yaml";
    std::string img1 = WORK_SPACE_PATH + "/image/1.png";
    std::string img2 = WORK_SPACE_PATH + "/image/2.png";

    std::shared_ptr<ORBFeature> chapterThree = std::make_shared<ORBFeature>(img1, img2, config_path);
    chapterThree->Run();
    return 0;
}
