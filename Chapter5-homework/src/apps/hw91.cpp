#include "ORB/trainingDictionary/trainingDic.hpp"
#include "ORB/common_include.h"
#include "ORB/global_defination/global_defination.h"


using namespace ORB;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    std::string dataSetPath = "/ssd/DATA/EuRoc/dataset/MH01/mav0/cam0/data";
    std::string dictionaryPath = WORK_SPACE_PATH + "/dictionary/vocabulary.yml.gz";

    LOG(INFO) << dictionaryPath;
    std::shared_ptr<TraingDIC> ob = std::make_shared<TraingDIC>(dataSetPath, dictionaryPath, 5, 10);
    ob->Run();
    return 0;
}