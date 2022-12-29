#include "ORB/trainingDictionary/trainingDic.hpp"
#include "ORB/common_include.h"
#include "ORB/global_defination/global_defination.h"


using namespace ORB;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    std::shared_ptr<TraingDIC> ob = std::make_shared<TraingDIC>();

    // using dictionary
    std::string dictionaryPath = WORK_SPACE_PATH + "/dictionary/vocabulary.yml.gz";
    std::string ImgPath = WORK_SPACE_PATH + "/image/dataSet";
    ob->UsingDictionary(dictionaryPath, ImgPath);
    return 0;
}