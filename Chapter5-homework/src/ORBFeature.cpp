#include "ORB/ORBFeature.hpp"

namespace ORB {
    ORBFeature::ORBFeature(const std::string& image_one_path, const std::string& image_two_path, const std::string& config_path)
    {
        camera_ptr = std::make_shared<Parameter>(config_path);
        image_one = cv::imread(image_one_path, cv::IMREAD_GRAYSCALE);
        image_two = cv::imread(image_two_path, cv::IMREAD_GRAYSCALE);

        orb_extractor_ptr = std::make_shared<ORBextractor>(camera_ptr->nFeatures, camera_ptr->scalseFactor, camera_ptr->nLevels, camera_ptr->iniThFAST, camera_ptr->minThFAST);
    }

    void ORBFeature::Run()
    {  
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        std::vector<cv::DMatch> matches, des_good_matches, good_matches;
        FindFeatureMatches(image_one, image_two, keypoints_1, keypoints_2, matches, des_good_matches);

        cv::Mat img;
        cv::drawMatches(image_one, keypoints_1, image_two, keypoints_2, matches, img);
        cv::imshow("all matches", img);
        cv::waitKey(0);

        /************************你需要完成的函数*************************************/
        UseHistConsistency(keypoints_1, keypoints_2, matches, good_matches);
        cv::Mat good_img;
        cv::drawMatches(image_one, keypoints_1, image_two, keypoints_2, good_matches, good_img);
        cv::imshow("这是通过直方图筛选的匹配特征点", good_img);
        cv::waitKey(0);

        cv::Mat des_img;
        cv::drawMatches(image_one, keypoints_1, image_two, keypoints_2, des_good_matches, des_img);
        cv::imshow("这是通过描述子距离筛选的匹配特征点", des_img);
        cv::waitKey(0);

    }

    /**
     *  matches: 中存放的是暴力匹配的数据
     *  good_matches: 该数据中，你需要使用直方图滤除不一致的匹配点对，将匹配好的数据放入该容器中
     */
    void ORBFeature::UseHistConsistency(const std::vector<cv::KeyPoint>& keypoints_1,
                                        const std::vector<cv::KeyPoint>& keypoints_2, 
                                        const std::vector<cv::DMatch>& matches, 
                                        std::vector<cv::DMatch>& good_matches)
    {
        /**
         *  请您使用直方图滤除不一致的匹配点对。
         */
        // 构建旋转直方图
        int HISTO_LENGTH = 30;
        std::vector<int> rotHist[HISTO_LENGTH];
        for(int i=0; i < HISTO_LENGTH; i++)
            // 每个bin里预分配500个，因为使用的是vector不够的话可以自动扩展容量
            rotHist[i].reserve(500); 
        
        const float factor = HISTO_LENGTH / 360.0f;

        // 遍历暴力匹配得到的matches，计算旋转角度差所在的直方图
        for (size_t n = 0; n < matches.size(); n++)
        {
            float rot  = keypoints_1[matches[n].queryIdx].angle - keypoints_2[matches[n].trainIdx].angle;
            // 角度小于0就加2Π
            if (rot < 0.0)
                rot = rot + 360.0f;
            
            // 判断在第几个直方图中，使用之前的factor
            // int bin = std::floor(rot * factor);
            int bin = std::round(rot * factor);
            // bin满了的话就再轮回1次
            if (bin == HISTO_LENGTH)
                bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            // rotHist[bin].push_back(match.trainIdx);
            rotHist[bin].push_back(n);
        }

        // 这里用了框架提供的只找到最多的直方图
        int maxIdx;
        ComputeOneMaxima(rotHist, HISTO_LENGTH, maxIdx);

        // 将在最大的直方图内的match给到good_match
        // ! 这种办法存进去的match比直方图对应bin里的多,应该是有重复匹配的原因
        // for (auto match : matches)
        // {
        //     if (std::count(rotHist[maxIdx].begin(), rotHist[maxIdx].end(), match.trainIdx))
        //         good_matches.push_back(match);
        // }
        for (size_t i = 0; i < rotHist[maxIdx].size(); i++)
        {
            good_matches.push_back(matches[rotHist[maxIdx][i]]);
        }
    }

    void ORBFeature::ComputeOneMaxima(std::vector<int>* histo, const int L, int &index)
    {
        int max1=0;

        for(int i=0; i<L; i++)
        {
            const int s = histo[i].size();
            if(s>max1)
            {
                max1=s;
                index=i;
            }
        }
    }

    void ORBFeature::MatchImage()
    {
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat descriptors_1, descriptors_2;
        std::vector<cv::DMatch> matches, good_matches;

        FindFeatureMatches(image_one, image_two, keypoints_1, keypoints_2, matches, good_matches);

        cv::Mat img;
        cv::drawMatches(image_one, keypoints_1, image_two, keypoints_2, matches, img);
        cv::imshow("all matches", img);
        cv::waitKey();

        cv::Mat img_goodmatch;
        cv::drawMatches(image_one, keypoints_1, image_two, keypoints_2, good_matches, img_goodmatch);
        cv::imshow("good matches", img_goodmatch);
        cv::waitKey(0);
    }

    /**
     *  vPoint_one ： 图像1上的匹配特征点
     *  vPoint_two :  图像2上的匹配特征点
     */
    void ORBFeature::PoseEstimation(cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point2f>& vPoint_one, std::vector<cv::Point2f>& vPoint_two)
    {
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat descriptors_1, descriptors_2;
        std::vector<cv::DMatch> matches, good_matches;
        FindFeatureMatches(image_one, image_two, keypoints_1, keypoints_2, matches, good_matches);

        for (int i = 0, id = good_matches.size(); i < id; i++)
        {
            vPoint_one.push_back(keypoints_1[good_matches[i].queryIdx].pt);
            vPoint_two.push_back(keypoints_2[good_matches[i].trainIdx].pt);
        }        

        cv::Point2d principal_point(camera_ptr->cx, camera_ptr->cy);
        double focal_length = camera_ptr->fx;
        cv::Mat essential_matrix = cv::findEssentialMat(vPoint_one, vPoint_two, focal_length, principal_point);

        cv::recoverPose(essential_matrix, vPoint_one, vPoint_two, R21, t21, focal_length, principal_point);
    }

    /**
     *  points 是三维点的坐标
     */ 
    void ORBFeature::Triangulation(std::vector<cv::Point3d>& points)
    {
        cv::Mat R21, t21;
        std::vector<cv::Point2f> vPoint_one, vPoint_two;
        std::vector<cv::Point2f> vPoint_one_temp, vPoint_two_temp;
        PoseEstimation(R21, t21, vPoint_one, vPoint_two);
        vPoint_one_temp = vPoint_one;
        vPoint_two_temp = vPoint_two;

        for (int i = 0, id = vPoint_one.size(); i < id; i++)
        {
            float x1 = (vPoint_one[i].x - camera_ptr->cx)/camera_ptr->fx; 
            float y1 = (vPoint_one[i].y - camera_ptr->cy)/camera_ptr->fy;

            float x2 = (vPoint_two[i].x - camera_ptr->cx)/camera_ptr->fx;
            float y2 = (vPoint_two[i].y - camera_ptr->cy)/camera_ptr->fy;

            vPoint_one[i] = cv::Point2f(x1, y1);
            vPoint_two[i] = cv::Point2f(x2, y2);
        }
        cv::Mat P1 = (cv::Mat_<float>(3, 4) <<
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0);
        cv::Mat P2(3, 4, CV_32F);
        R21.copyTo(P2.rowRange(0, 3).colRange(0, 3));
        t21.copyTo(P2.rowRange(0, 3).col(3));
        cv::Mat pts_4d;
        points.clear();
        cv::triangulatePoints(P1, P2, vPoint_one, vPoint_two, pts_4d);
        for (int i = 0; i < pts_4d.cols; i++)
        {
            cv::Mat x = pts_4d.col(i);
            x /= x.at<float>(3, 0);
            cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
            points.push_back(p);
        }
    }

    void ORBFeature::FindFeatureMatches(const cv::Mat& src_image_one, const cv::Mat& src_image_two,
                                        std::vector<cv::KeyPoint>& vkeypoints_one, 
                                        std::vector<cv::KeyPoint>& vkeypoints_two,
                                        std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& good_matches)
    {
        cv::Mat descriptors_one, descriptors_two;
        if (1)
        {   
            ExtractORB(src_image_one, vkeypoints_one, descriptors_one);
            ExtractORB(src_image_two, vkeypoints_two, descriptors_two);
        }
        else {
            ORBSLAM2ExtractORB(src_image_one, vkeypoints_one, descriptors_one);
            ORBSLAM2ExtractORB(src_image_two, vkeypoints_two, descriptors_two);
        }

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(descriptors_one, descriptors_two, matches);
        double min_dist = 10000, max_dist = 0;
        for (int i = 0; i < descriptors_one.rows; i++)
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

        for (int i = 0; i < descriptors_one.rows; i++)
        {
            if (matches[i].distance <= std::max(2 * min_dist, 30.0))
            {
                good_matches.push_back(matches[i]);
            }
        }    

    }

    cv::Scalar ORBFeature::get_color(float depth)
    {
        float up_th = 50, low_th = 10, th_range = up_th - low_th;
        if (depth > up_th) depth = up_th;
        if (depth < low_th) depth = low_th;
        return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
    }

    void ORBFeature::ExtractORB(const cv::Mat& image, std::vector<cv::KeyPoint>& vkeypoints, cv::Mat& descriptors)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

        detector->detect(image, vkeypoints);
        descriptor->compute(image, vkeypoints, descriptors);
    }

    void ORBFeature::UndistortImage(const cv::Mat& image, cv::Mat& outImage)
    {
        int rows = image.rows;
        int cols = image.cols;
        cv::Mat Image = cv::Mat(rows, cols, CV_8UC1);

        for (int v = 0; v < rows; v++)
        {
            for (int u = 0; u < cols; u++)
            {
                double x = (u - camera_ptr->cx)/camera_ptr->fx;
                double y = (v - camera_ptr->cy)/camera_ptr->fy;

                double r = sqrt(x * x + y * y);
                double r2 = r * r;
                double r4 = r2 * r2;

                double x_dis = x * (1 + camera_ptr->k1 * r2 + camera_ptr->k2 * r4) + 2 * camera_ptr->p1 * x * y + camera_ptr->p2 * (r2 + 2 * x * x);
                double y_dis = y * (1 + camera_ptr->k1 * r2 + camera_ptr->k2 * r4) + camera_ptr->p1 * (r2 + 2 * y * y) + 2 * camera_ptr->p2 * x * y;

                double u_dis = camera_ptr->fx * x_dis + camera_ptr->cx;
                double v_dis = camera_ptr->fy * y_dis + camera_ptr->cy;

                if (u_dis >= 0 && v_dis >= 0 && u_dis < cols && v_dis < rows)
                {
                    Image.at<uchar>(v, u) = image.at<uchar>((int)v_dis, (int)u_dis);
                }
                else {
                    Image.at<uchar>(v, u) = 0;
                }
            }
            outImage = Image;
        }
    }

    void ORBFeature::ORBSLAM2ExtractORB(const cv::Mat& srcImage, std::vector<cv::KeyPoint>& vkeypoints, cv::Mat& descriptors)
    {
        orb_extractor_ptr->operator()(srcImage, cv::Mat(), vkeypoints, descriptors);       
    }
}