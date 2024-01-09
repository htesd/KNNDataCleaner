#include "opencv4/opencv2/opencv.hpp"
#include <fstream>
#include <sys/time.h>
#include <dirent.h>
#include <stdio.h>
using namespace std;
using namespace cv;
using namespace cv::ml;
std::vector<string> get_filelist_from_dir(std::string _path)
{
    DIR* dir;
    std::cout<<_path<<"\n";
    dir = opendir(_path.c_str());
    struct dirent* ptr;
    std::vector<std::string> file;

    while((ptr = readdir(dir)) != NULL)
    {
        string temp_path=_path;
        temp_path.append("/");
        if(ptr->d_name[0] == '.') {continue;}
        file.push_back(temp_path.append(ptr->d_name));

    }
    closedir(dir);
    sort(file.begin(), file.end());
    return file;
}
/*!
 *
 * @param filepath
 * @param train_data
 * @param train_label
 * @param data_num  这个地方应该是每个类别采取的样本数量
 * @param class_num
 */
void get_gray_data(std::string filepath,cv::Mat *train_data,cv::Mat *train_label,int data_num,int class_num){
    std::vector<std::vector<string>> all_file_list;

    int count=0;
    for(int i=0;i!=class_num;i++){
        string temp_file_path=filepath;
        all_file_list.push_back(get_filelist_from_dir(temp_file_path.append(std::to_string(i))));
    }
    int max_size=0;
    for(int i=0;i!=class_num;i++){
        int temp=all_file_list.at(i).size();
        if(temp>max_size){
            max_size=temp;
        }
    }

    for(int i=0;i!=max_size;i++){
        if(i>data_num){
            break;
        }

        for(int j=0;j!=class_num;j++){

            if(i>all_file_list.at(j).size()-1){
                continue;
            }
            if(all_file_list.at(j).size()==0){
                continue;
            }

            cv::Mat temp_img=cv::imread(all_file_list.at(j).at(i));
            if(temp_img.empty()){

                remove(all_file_list.at(j).at(i).c_str());
                continue;
            }
//            cv::imshow("test",temp_img);
//            cv::waitKey(0);
//            std::cout<<j<<"\n";
            cvtColor(temp_img,temp_img,cv::COLOR_BGR2GRAY);
            cv::resize(temp_img,temp_img,cv::Size(28,28));
            train_data->push_back(temp_img.reshape(0, 1));
            train_label->push_back(j);
            count++;

        }

    }
    cout<<"总数:"<<count<<"\n";
    train_data->convertTo(*train_data, CV_32F); //uchar型转换为cv_32f
}


int knn_predict(Mat img,Ptr<KNearest> model){
    cvtColor(img, img, COLOR_BGR2GRAY);
    //threshold(src, src, 0, 255, CV_THRESH_OTSU);

    resize(img, img, Size(28, 28));
    Mat test;
    test.push_back(img.reshape(0, 1));
    test.convertTo(test, CV_32F);
    struct timeval time;
    gettimeofday(&time, NULL);
    //printf("s: %ld, ms: %ld\n", time.tv_sec, (time.tv_sec*1000 + time.tv_usec/1000));
    long int time1=time.tv_usec;
    int result = model->predict(test);

    gettimeofday(&time, NULL);
    long int time2=time.tv_usec;
    std::cout<<time2-time1<<"微秒 ";
    //printf("s: %ld, ms: %ld\n", time.tv_sec, (time.tv_sec*1000 + time.tv_usec/1000));

    cout << "我猜你写的是：" << result << endl;
//    imshow("原图像", img);
//    waitKey(1);
    return result;
}


int main()
{
    string file_path="/home/iiap/桌面/数据集/resnet18/data/";
    Ptr<KNearest> model;
    std::fstream file;
    string fileName = "num_knn_pixel.xml";
    Mat trainData, trainLabels;
    cv::Mat train_data,train_label;
    file.open(fileName.c_str(), ios::in);

    int class_num=7;

    if (file)
    {
        //训练结果存在，加载训练结果
        model = Algorithm::load<KNearest>("num_knn_pixel.xml");
    }
    else
    {	//训练结果不存在，重新训练

        //依据类别遍历文件夹，每类随机选取100个做初始化

        std::cout<<"?????"<<std::endl;
        get_gray_data(file_path,&train_data,&train_label,10,class_num);

        std::cout<<"?????"<<std::endl;
        std::cout<<train_data.size<<"\n";
        std::cout<<train_label.size<<"\n";

        int samplesNum = train_data.rows;

        int trainNum = samplesNum*8/10;

        std::cout<<"train num"<<samplesNum<<std::endl;

        trainData = train_data(Range(0, trainNum), Range::all());
        trainLabels = train_label(Range(0, trainNum), Range::all());

        //使用KNN算法
        //指找最近的个坐标
        int K = 5;
        Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);

        model = KNearest::create();
        model->setDefaultK(K);
        model->setIsClassifier(true);
        model->train(tData);

        //预测分类
        double train_hr = 0, test_hr = 0;
        Mat response;
        // compute prediction error on train and test data
        for (int i = 0; i < samplesNum; i++)
        {
            Mat sample = train_data.row(i);
            float r = model->predict(sample);   //对所有行进行预测
            //预测结果与原结果相比，相等为1，不等为0
            r = std::abs(r - train_label.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

            if (i < trainNum)
                train_hr += r;  //累积正确数
            else
                test_hr += r;
        }

        test_hr /= samplesNum - trainNum;
        train_hr = trainNum > 0 ? train_hr / trainNum : 1.;

        printf("accuracy: train = %.1f%%, test = %.1f%%\n",
               train_hr*100., test_hr*100.);
        //保存训练结果
        //model->save("./num_knn_pixel.xml");
    }
    // ===============================预测部分========================
//同样的思路：按照类别遍历所有文件，预测正确的就删除，失败的就加入并且继续训练


    std::vector<std::vector<string>> all_file_list;
    for(int i=0;i!=class_num;i++){
        string temp_file_path=file_path;

        all_file_list.push_back(get_filelist_from_dir(temp_file_path.append(std::to_string(i))));
    }
    int max_size=0;
    for(int i=0;i!=class_num;i++){
        int temp=all_file_list.at(i).size();
        if(temp>max_size){
            max_size=temp;
        }
    }
for(int i=0;i!=max_size;i++){

    for(int j=0;j!=class_num;j++){

        if (all_file_list.at(j).size()==0){
            continue;
        }

        if(i>all_file_list.at(j).size()-1){
            continue;
        }
        cv::Mat img=cv::imread(all_file_list.at(j).at(i));
        int result=knn_predict(img,model);

        if(result==j){
            remove(all_file_list.at(j).at(i).c_str());
            std::cout<<"remove:"<<all_file_list.at(j).at(i).c_str();
            continue;
        }else{
//            cv::imshow("special img",img);
//            cv::waitKey(0);
            cvtColor(img,img,cv::COLOR_BGR2GRAY);
            cv::resize(img,img,cv::Size(28,28));
            img.convertTo(img, CV_32F);
            train_data.push_back(img.reshape(0, 1));
            train_label.push_back(j);
            Ptr<TrainData> tData = TrainData::create(train_data, ROW_SAMPLE, train_label);
            model->train(tData);
            std::cout<<"现在的训练大小"<<train_data.size<<"\n";


//            model->save("./num_knn_pixel.xml");
        }
    }
}

    return 0;
}

