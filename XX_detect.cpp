/**
 *  This code change from caffe/examples/classification.cpp 
 *  This is a demo code for using a XX model to do detection.  
 *  
 * 
 **/
#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using namespace std;

//定义检查器class
class Detector
{
    public:
    Detector(const string& model_file,//预测阶段的网络模型描述文件  
             const string& weights_file,//已经训练好的caffemodel文件所在的路径  
             const string& mean_file,//均值文件所在的路径
             const string& mean_value);

    //提取人脸,返回结果
    std::vector<vector<float> > Detect(const cv::Mat& img);
        
    private:

    //对mean_进行初始化
    void SetMean(const string& mean_file,const string& mean_value);

    //将input_channels与网络输入绑定
    viod WrapInputLayer(std::vector<cv::Mat>* input_channels);

    //预处理
    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);
    
    private:
    shared_ptr<Net<float> > net_;//网络指针
    cv::Size input_geometry_;//输入图片的Size
    int num_channels_;//输入图片的channels
    cv::Mat mean_;//均值图片
};

//构造函数 
Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value)
{
    //定义工作模式CPU或者GPU
    #ifdef CPU_ONLY 
        Caffe::set_mode(Caffe::CPU);
    #else
        Caffe::set_mode(Caffe::GPU);
    #endif

    //载入网络
    net_.reset(new Net<float>(model_file,TEST));//从model_file中读取网络结构,初始化网络
    net_->CopyTrainedLayerFrom(weights_file);//从weights_file路径下的caffemodel文件读入训练完毕的网络参数 

    CHECK_EQ(net_->num_inputs(),1)  << "Network should have exactly one input."; //核验是不是只输入了一张图像，输入的blob结构为(N,C,H,W)，在这里，N只能为1       
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";  //核验输出的blob结构，输出的blob结构同样为(N,C,W,H)，在这里，N同样只能为1    


    // /定义网络的基本数据单元  
    Blob<float>* input_layer = net_->input_blobs()[0];//获取网络输入的blob，表示网络的数据层  
    num_channels_ = input_layer->channels();  //获取输入的通道数   

    //检验输入图片是否是3通道或者1通道
    CHECK(num_channels_==3||num_channels_==1)
        << "Input layer should have 1 or 3 channels."; 

    //读取输入图片宽高
    input_geometry_=cv::Size(input_layer->width(),input_layer->height());
     
    //初始化均值图片
    SetMean(mean_file,mean_value); 

}


//检测器,输入图片,返回 结果,每个vector代表一个解果,存有位置及信任度信息
std::vector<vector<float> > Detector::Detect(const cv::Mat& img)
{


    Blob<float>* input_layer=net_->input_blobs()[0];
    input_layer->Reshape(1,num_channels_,input_geometry_.height,input_geometry_.width);
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    
    WrapInputLayer(&input_channels);

    //预处理
    Preprocess(img,&input_channels);

    //网络前向传播
    net_->Forward();

    //将输出拷贝至std::vector
    Blob<float>* result_blob=net_->output_blobs()[0];
    const float* result=result_blob->cpu_data();  //读取输出信息
    const int num_det=result_blob->height();//height:检测到的数量

    //定义返回结果
    vector<vector<float> > detections;//格式即为Detect函数返回值格式
    
    //逐个保存每一个结果,每个结果占7个vector
    for(int k=0;k<num_det;k++)
    {
        if(result[0]==-1)
        {
            result+=7; //跳过无用的结果
            continue;
        }
        vector<float> detection(result,result+7);
        detections.push_back(detection);
        result+=7;
    }

    //返回结果
    return detections;
}    


//初始化均值图片mean_
void Detector::SetMean(const string& mean_file, const string& mean_value)
{
    //设置均值有2种方式,一种是通过mean_file一种是通过mean_value
    //mean_file类似于用caffe做图像分类时需要提供的lmdb文件,是关于像素点的均值,也就是对所有图片关于像素点均值
    //mean_value只有三个值,分别代表三个通道的均值

    //定义均值图片
    cv::Scalar channel_mean;

    //通过mean_file设置
    if(!mean_file.empty())
    {
        CHECK(mean_value.empty()) <<  
         "Cannot specify mean_file and mean_value at the same time";  

        BlobProto blob_proto;  
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);  

        Blob<float> mean_blob;  
        mean_blob.FromProto(blob_proto);  
        CHECK_EQ(mean_blob.channels(), num_channels_)  
          << "Number of channels of mean file doesn't match input layer.";  

        std::vector<cv::Mat> channels;  
        float* data = mean_blob.mutable_cpu_data();

        for (int i = 0; i < num_channels_; ++i)
        {
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);  
            data += mean_blob.height() * mean_blob.width();  
        }

        cv::Mat mean;  
        cv::merge(channels, mean);  

        channel_mean = cv::mean(mean);  
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);  
    }

    //通过mean_value设置
    if(!mean_value.empty())
    {
        //检测mean_file是否为空
        CHECK(mean_file.empty())
         << "Cannot specify mean_file and mean_value at the same time";  
        
        //将mean_value读入流中
        stringstream ss(mean_value);
        vector<float> values;
        string item;

        //遇到','截取
        while(getline(ss,item,','))
        {
            float value=std::atof(item.c_str());//字符转化为数字
            values.push_back(value);
        }

        CHECK(values.size() == 1 || values.size() == num_channels_) << 
         "Specify either 1 mean_value or as many as channels: " << num_channels_;  

        //将提取出来均值储存入Mat中
        std::vector<cv::Mat> channels;
        for(int i=0;i<num_channels_;i++)
        {
            cv::Mat channel(input_geometry_.height,input_geometry_.width,CV_32FC1,cv::Scalar(values[i]));//利用scalar对Mat进行初始化,例如Mat（height,width,CV_32F2,Scalar（1，2）），2个channel的图片，一层为1，  
            channels.push_back(channel);
        }

        cv::merge(channels,mean_);
    }
}

//将输入input_channels与网络的输入绑定（wrap）  
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{

    Blob<float>* input_layer=net_->input_blobs()[0];

    int width=input_layer->width();
    int height=input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    
    //将Mat的头指针与input_data指向相同,也就意味着，向Mat里写东西，就等同于向网络的输入写数据  
    for(int i=0;i<input_layer->channels();i++)
    {
        cv::Mat channel(height,width,CV_32FC1,input_data);
        input_channels->push_back(channel);
        input_data+=width*height;
    }
}

//各种初始化
void Detector::Preprocess(const cv::Mat& img,
                          std::vector<cv::Mat>* input_channels)
{
    cv::Mat sample;
    
    //当输入图片的channel与网络规定的channel不同时的操作
    if(img.channels()==3 && num_channels_==1)
        cv::cvtColor(img,sample,cv::COLOR_BGR2GRAY);
    else if(img.channels()==4 && num_channels_==1)
        cv::cvtColor(img,sample,cv::COLOR_BGRA2GRAY);
    else if(img.channels()==4 && num_channels_==3)
        cv::cvtColor(img,sample,cv::COLOR_BGRA2BGR);
    else if(img.channels()==1 && num_channels_==3)
        cv::cvtColor(img,sample,cv::COLOR_GRAY2BGR);
    else
        sample = img;

     cv::Mat sample_resized;

     //当输入图片size与网络规定不同时的操作
     if(sample.size()!=input_geometry_)
        cv::resize(sample,sample_resized,input_geometry_);
    else
        sample_resized=sample;
    
    cv::Mat sample_float;

    //将像素转化为float
    if(num_channels_==3)
        sample_resized.convertTo(sample_float,CV_32FC3);
    else
        sample_resized.convertTo(sample_float,CV_32FC1);
    
    //图像减去均值
    cv::Mat sample_normalized;
    cv::subtract(sample_float,mean_,sample_normalized);

    //数据输入,input_channel已经和网络输入绑定，即指向相同，所以将数据写入input_channel的  
    cv::split(sample_normalized,*input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)==net_->input_blobs()[0]->cpu_data())
         << "Input channels are not wrapping the input layer of the network."; 
}


//一堆定义命令行输入的指令，配置Caffe时用到的一个包，好像是gflags实现的，第一个参数是名称，第二个是默认值，第三个是解释   
//例如在命令行通过 -mean_value=" "就可以对mean_value进行赋值  

//初始化mean_file为空
DEFINE_string(mean_file,"",
     "The mean file used to subtract from the input image.");  

//初始化mean_value
DEFINE_string(mean_value,"104,117,123",
    "If specified, can be one value or can be same as image channels"  
    " - would subtract from the corresponding channel). Separated by ','."  
    "Either mean_file or mean_value should be provided, not both.");

DEFINE_string(file_type, "image",  
    "The file type in the list_file. Currently support image and video."); 

//此处设置输出路径 
DEFINE_string(out_file, "",  
    "If provided, store the detection results in the out_file.");  
DEFINE_double(confidence_threshold, 0.01,  
    "Only store detections with score higher than the threshold.");  
DEFINE_string(detect_type, "trace",  
    "Do detection:detect Do tracing :trace");    




//下面为main函数

int main(int argc,char** argv)
{
    ::google::InitGoogleLogging(argv[0]);//InitGoogleLogging做了一些初始化glog的工作 
    FLAGS_alsologtostderr = 1;  

    #ifdef GFLAGS_GFLAGS_H_
        namespace gflags=google;
    #endif

    gflags::ParseCommandLineFlags(&argc, &argv, true); 

    if (argc < 4) 
    {  
        gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/XX/XX_detect");  
        return 1;  
    }  

    //读取参数
    const string& model_file=argv[1];//这里是否改为固定路径,加载.prototxt
    const string& weights_file=argv[2];//这里是否改为固定路径,加载caffemodel


    //通过这种方式,读入DEFINE_string的赋值
    const string& mean_file=FLAGS_mean_file;
    const string& mean_value=FLAGS_mean_value;
    const string& file_type=FLAGS_file_type;
    const string& out_file=FLAGS_out_file;
    const float confidence_threshold=FLAGS_confidence_threshold;
    const string& detect_type=FLAGS_detect_type;

    //初始化网络
    Detector detector(model_file,weights_file,mean_file,mean_value);

    //设置输出路径
    std::streambuf* buf=std::cout.rdbuf();
    std::ofstream outfile;
    if(!out_file.empty())
    {
        outfile.open(out_file.c_str());
        if(outfile.good())
        {
            buf=outfile.rdbuf();
        }
    }
    std::ostream out(buf);

    //一张图一张图处理
    std::ifstream infile(argv[3]);
    std::string file;
    while(infile>>file)
    {   
        if(file_type=="image")
        {
            cv::Mat img=cv::imread(file,-1);
            CHECK(!img.empty())<< "Unable to decode image " << file;  
            std::vector<vector<float> > detections=detector.Detect(img);

            //打印得到的结果
            for(int i=0;i<detections.size();i++)
            {
                const vector<float>& d=detections[i];
                // Detection中的7个值分别为 [image_id, label, score, xmin, ymin, xmax, ymax].
                CHECK_EQ(d.size(),7);
                const float score=d[2]; //置信度
                if(score>=confidence_threshold)
                {
                    out<<file<<"";
                    out<<static_cast<int>(d[1])<<"";
                    out<<score<<"";
                    out<<static_cast<int>(d[3]*img.cols)<<"";  //xmin
                    out<<static_cast<int>(d[4]*img.rows)<<"";  //ymin
                    out<<static_cast<int>(d[5]*img.cols)<<"";  //xmax
                    out<<static_cast<int>(d[6]*img.rows)<<std::endl;   //ymax
                }
            }
        }else
        {
            return 1;
        }
    }
}

