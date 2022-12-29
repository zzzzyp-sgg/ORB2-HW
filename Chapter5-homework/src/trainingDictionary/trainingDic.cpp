#include <ORB/trainingDictionary/trainingDic.hpp>

namespace ORB {
    TraingDIC::TraingDIC(const std::string& dataSetPath, const std::string& outDICPath, int Depth, int Branch)
        :k(Branch), d(Depth), outputPath(outDICPath)
    {
        this->dataSetPath = dataSetPath;
    }

    void TraingDIC::Run()
    {
        // 作业， 很简单的
        ReadDataSet(dataSetPath, vImagePath);
        if (vImagePath.size() == 0)
        {
            LOG(ERROR) << "Please check ReadDataSet() function!";
            return;
        }

        
        std::vector<cv::Mat> vDescriptors;
        for (std::string str : vImagePath)
        {
            cv::Mat descriptor;
            DetectDescriptors(str, descriptor);
            vDescriptors.push_back(descriptor);
        }

        // 作业，训练字典，开始你的表演
        traingDictionary(vDescriptors);
    }

    /**
     *  作业 使用自己训练好的字典，找到得分最高的图像，并做特征点匹配
     *  <三件事情>
     *  1， 使用自己训练好的字典，检测两张得分最高的图像（两张图像不一样）
     *  2， 做特征点匹配（用你熟悉的方法）
     *  3， 输出图像
     */
    void TraingDIC::UsingDictionary(const std::string& dicFilePath, const std::string& imgPath)
    {
        // TODO
        // 检测两张得分最高的图像，应该就是指两张最相似的图像？
        // 把图片都读入
        std::vector<cv::String> _ImagePath;
        ReadDataSet(imgPath, _ImagePath);

        std::vector<cv::Mat> vDescriptors;
        for (std::string str : _ImagePath)
        {
            cv::Mat descriptor;
            DetectDescriptors(str, descriptor);
            vDescriptors.push_back(descriptor);
        }

        // 先加载词典
        DBoW3::Vocabulary vocab(dicFilePath);
        double bestScore = 0;                   // 最佳得分
        int idx_1, idx_2;                       // 两张最相似的图片索引
        // std::vector<double> allScore;
        for (size_t i = 0; i < vDescriptors.size(); i++)
        {
            DBoW3::BowVector v1;
            vocab.transform(vDescriptors[i], v1);
            for (size_t j = i +1; j < vDescriptors.size(); j++)
            {
                DBoW3::BowVector v2;
                vocab.transform(vDescriptors[j], v2);
                double score = vocab.score(v1, v2);
                if (score > bestScore)  // 找到最相似的两张图片
                {
                    bestScore = score;
                    idx_1 = i;
                    idx_2 = j;
                }
            }
        }

        // 对两张图片进行特征匹配
        // 先利用ORB提取两张图片的特征点
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
        std::vector<cv::KeyPoint> vkeypoints_1, vkeypoints_2;
        cv::Mat descriptors_1, descriptors_2;
        cv::Mat img_1 = cv::imread(_ImagePath[idx_1]);
        cv::Mat img_2 = cv::imread(_ImagePath[idx_2]);
        detector->detect(img_1, vkeypoints_1);                      // 图1
        descriptor->compute(img_1, vkeypoints_1, descriptors_1);
        detector->detect(img_2, vkeypoints_2);                      // 图2
        descriptor->compute(img_2, vkeypoints_2, descriptors_2);
        // 特征匹配
        std::vector<cv::DMatch> matches, good_matches;
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(descriptors_1, descriptors_2, matches);
        double min_dist = 10000, max_dist = 0;
        for (int i = 0; i < descriptors_1.rows; i++)
        {
            double dist = matches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        static bool showFlag = false;
        if(showFlag == false)
        {
            showFlag = true;
            LOG(INFO) << "-- Max dist : " << max_dist;
            LOG(INFO) << "-- Min dist : " << min_dist;
        }

        for (int i = 0; i < descriptors_1.rows; i++)
        {
            if (matches[i].distance <= std::max(2 * min_dist, 30.0))
            {
                good_matches.push_back(matches[i]);
            }
        }

        // 输出图像
        cv::Mat img;
        cv::drawMatches(img_1, vkeypoints_1, img_2, vkeypoints_2, matches, img);
        cv::imshow("all matches", img);
        cv::waitKey(0);
        cv::Mat good_img;
        cv::drawMatches(img_1, vkeypoints_1, img_2, vkeypoints_2, good_matches, good_img);
        cv::imshow("good matches", good_img);
        cv::waitKey(0);
    }

    /**
     *  读取数据集
     *  dataSetPath 是数据集路径
     *  ImagePath  是读取到图像的路径
     */
    void TraingDIC::ReadDataSet(const std::string& dataSetPath, std::vector<cv::String>& ImagePath)
    {
        // 作业
        ImagePath.clear();
        
        // 使用opencv自带的glob函数
        cv::glob(dataSetPath, ImagePath);
        // std::cout << ImagePath[0];
    }

    void TraingDIC::DetectDescriptors(const cv::String& image_path, cv::Mat& descriptor)
    {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

        cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
        std::vector<cv::KeyPoint> vKeyPoints;
        detector->detectAndCompute(image, cv::Mat(), vKeyPoints, descriptor);
    }

    void TraingDIC::DetectDescriptors(const cv::String& imagePath, std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat& descriptor)
    {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

        cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
        detector->detectAndCompute(image, cv::Mat(), vKeyPoints, descriptor);
    }

    /**
     *     作业，训练字典
     *     请您计时，看一下您的机器上用时多长时间？
     */
    void TraingDIC::traingDictionary(const std::vector<cv::Mat> vdescriptors)
    {
        if (vdescriptors.size() == 0)
        {
            LOG(ERROR) << "please check vdescriptors!";
        }
        // TODO

        // 用了chrono来计时
        // 最终耗时88秒左右
        DBoW3::Vocabulary vocab(k, d);  // 使用指定的k和d(虽然和默认的一样)
        auto t1 = std::chrono::steady_clock::now();
        vocab.create(vdescriptors);
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "train vocabulary time cost = " << time_used.count() << " seconds. " << std::endl;
        // std::cout << "vocabulary info: " << std::endl << vocab << std::endl; // 字典信息
        // save自带的输出太多了，临时禁用一下
        std::cout.setstate(std::ios_base::failbit);
        vocab.save(outputPath);  // 保存字典压缩包
        std::cout.clear();
    }
}