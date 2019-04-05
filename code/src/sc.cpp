/**
 * COMP 6651
 * Khyatibahen Chaudhary 
 * REFERENCES: http://www.cs.middlebury.edu/~dsilver/vision/seam-carving/
 * http://answers.opencv.org/question/27248/max-and-min-values-in-a-mat/
 * https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html
 */

#include "sc.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace cv;
using namespace std;


//1st. Energy map
void energyImageGeneration(Mat& inputImage, Mat& BW_image){
    	Mat gray_image;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

    //laplacian function
	GaussianBlur(inputImage, gray_image, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(gray_image, gray_image, CV_RGB2GRAY);
	Laplacian(gray_image, BW_image, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(BW_image, BW_image);
    // end of laplacian
    /*GaussianBlur(inputImage, gray_image, Size(3,3), 0, 0, BORDER_DEFAULT);
    cvtColor(gray_image, gray_image, CV_BGR2GRAY);
    
    // use Sobel to calculate the gradient of the image in the x and y direction
    Sobel(gray_image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(gray_image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    
    // convert gradients to abosulte versions of themselves
    convertScaleAbs(grad_x, grad_x);
    convertScaleAbs(grad_y, grad_y);
    
    // total gradient (approx)
    addWeighted( grad_x, 0.5, grad_y, 0.5, 0, BW_image );*/
    
    // convert the default values to double precision
    BW_image.convertTo(BW_image, CV_64F, 1.0/255.0);
    imshow("gradient image", BW_image);
}

//3rd. find seams
vector<int> findOptimalPath(Mat& energyMap, vector<int> optimalPath, char seam_direction){
	double maximum = 2147483647.0;
	int minIndex;
    if(seam_direction == 'v'){
        //min and max location point in Mat.
        //Mat table = energyMap.row(energyMap.rows - 1);
	double min_value, max_value;
	
	for(int minColumn = 0; minColumn<energyMap.cols; minColumn++){
		if(energyMap.at<double>(energyMap.rows-1, minColumn) < maximum){
			maximum = energyMap.at<double>(energyMap.rows-1, minColumn);
			minIndex = minColumn;		
		}
	}
	int min_index;
	min_index = minIndex;
        optimalPath[energyMap.rows - 1] = min_index;
	
	int offset;
	for(int rows = energyMap.rows-2; rows>=0; rows--){
		
	    
	   double a = energyMap.at<double>(rows, max(min_index - 1, 0));
           double b = energyMap.at<double>(rows, min_index);
           double c = energyMap.at<double>(rows, min(min_index + 1, energyMap.cols - 1));
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
                offset = 0;
            }
            else if (min(b, c) > a) {
                offset = -1;
            }
            
            min_index += offset;
            min_index = std::min(std::max(min_index, 0), energyMap.cols - 1); // take care of edge cases
	    optimalPath[rows] = min_index;
	   	
	}
       
    }
    else if(seam_direction == 'h'){
        Mat table = energyMap.col(energyMap.cols - 1);
	double min_value, max_value;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(table, &min_value, &max_value, &min_loc, &max_loc);
	int min_index = min_loc.y;
        optimalPath[energyMap.cols - 1] = min_index;
	
       // int cols = energyMap.cols - 2;
        int offset;
	for(int cols = energyMap.cols - 2; cols >=0; cols--){
		
	    double a = energyMap.at<double>(std::max(min_index - 1, 0), cols);
            double b = energyMap.at<double>(min_index, cols);
            double c = energyMap.at<double>(std::min(min_index + 1, energyMap.rows - 1), cols);
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
                offset = 0;
            }
            else if (min(b, c) > a) {
                offset = -1;
            }
            
            min_index += offset;
            min_index = std::min(std::max(min_index, 0), energyMap.rows - 1); // take care of edge cases
	    optimalPath[cols] = min_index;
	    
	}
    }
   
   return optimalPath;
}

//2nd. cummulative energy/ dynamic programming
vector<int> calculateSeams(Mat& inputImage, Mat& imageIntense, char seam_direction){
    int imageRows = imageIntense.rows;
    int imageCols = imageIntense.cols;
    vector<int> optimalPath;
    Mat energyMap = Mat(imageRows, imageCols, CV_64F, double(0));
 //  imageIntense.copyTo(energyMap);
   
 //  Mat energyMap;
	
   if(seam_direction == 'h'){
	imageIntense.row(0).copyTo(energyMap.row(0));
//cout<<"inside horizontal"<<endl;
       optimalPath.resize(imageCols);
       for(int column = 1; column < imageCols; column++){
            for(int row = 0; row < imageRows; row++){
                if(row == 0)
                    energyMap.at<double>(row,column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row, column-1), energyMap.at<double>(row+1, column-1));
                else if(row == imageRows -1){
			energyMap.at<double>(row,column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row, column-1),energyMap.at<double>(row-1, column-1));
		}
		else{
                    energyMap.at<double>(row, column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row-1, column-1), min(energyMap.at<double>(row, column-1), 				energyMap.at<double>(row+1, column-1)));
                }       
            }
        }

       optimalPath = findOptimalPath(energyMap, optimalPath, seam_direction);
    }
   else if(seam_direction == 'v'){
	imageIntense.col(0).copyTo(energyMap.col(0));  
	//cout<<"inside VERtical"<<endl;
       optimalPath.resize(imageRows);

       for(int row = 1; row < imageRows; row++){
            for(int column = 0; column < imageCols; column++){
                if(column == 0){
                    energyMap.at<double>(row,column) =imageIntense.at<double>(row, column) + min(energyMap.at<double>(row-1, column), energyMap.at<double>(row-1, column+1));
		}
		else if(column == imageCols -1){
			energyMap.at<double>(row, column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row-1, column-1),energyMap.at<double>(row-1, column));
		}
                else{
                    energyMap.at<double>(row, column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row-1, column-1), min(energyMap.at<double>(row-1, column), energyMap.at<double>(row-1,column+1)));
                }       
            }
        }

       optimalPath = findOptimalPath(energyMap, optimalPath, seam_direction);
   }
	
	return optimalPath;
}


bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image){

    // some sanity checks
    // Check 1 -> new_width <= in_image.cols
    if(new_width>in_image.cols){
        cout<<"Invalid request!!! new_width has to be smaller than the current size!"<<endl;
        return false;
    }
    if(new_height>in_image.rows){
        cout<<"Invalid request!!! ne_height has to be smaller than the current size!"<<endl;
        return false;
    }
    
    if(new_width<=0){
        cout<<"Invalid request!!! new_width has to be positive!"<<endl;
        return false;
    }
    
    if(new_height<=0){
        cout<<"Invalid request!!! new_height has to be positive!"<<endl;
        return false;    
    }

    
    return seam_carving_trivial(in_image, new_width, new_height, out_image);
}


// seam carves by removing trivial seams
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image){

    Mat iimage = in_image.clone();
    Mat oimage = in_image.clone();
    Mat BW_image;
    energyImageGeneration(iimage,BW_image);
   
    while(iimage.rows!=new_height || iimage.cols!=new_width){
        // horizontal seam if needed
        if(iimage.rows>new_height){
            reduce_horizontal_seam_trivial(iimage, oimage, BW_image);
            iimage = oimage.clone();
            energyImageGeneration(iimage,BW_image);
        }
        
        if(iimage.cols>new_width){
            reduce_vertical_seam_trivial(iimage, oimage, BW_image);
            iimage = oimage.clone();
            energyImageGeneration(iimage,BW_image);
        }
    }
    
    out_image = oimage.clone();
    return true;
}

// horizontl trivial seam is a seam through the center of the image
bool reduce_horizontal_seam_trivial(Mat& in_image, Mat& out_image, Mat& BW_image){

    // retrieve the dimensions of the new image
    int rows = in_image.rows-1;
    int cols = in_image.cols;
    
    // create an image slighly smaller
    out_image = Mat(rows, cols, CV_8UC3);
    vector<int> optimalPathArray;
   optimalPathArray = calculateSeams(in_image,BW_image, 'h' );
//cout<<"khkj"<<endl;
    //populate the image
    
    //optimalPathArray[i]
   for(int i=0;i<cols;++i){
        for(int j=0;j<optimalPathArray[i];++j){
//		cout<<i<<endl;
            Vec3b pixel = in_image.at<Vec3b>(j, i);
            out_image.at<Vec3b>(j,i) = pixel;
        }
}
 for(int i=0;i<cols;++i){
        for(int j=optimalPathArray[i];j<rows;++j){
//cout<<"ljkjkljkl"<<endl;
            Vec3b pixel = in_image.at<Vec3b>(j+1, i);
            out_image.at<Vec3b>(j,i) = pixel;
        }
        
    }

    return true;
}

// vertical trivial seam is a seam through the center of the image
bool reduce_vertical_seam_trivial(Mat& in_image, Mat& out_image, Mat& BW_image){
    // retrieve the dimensions of the new image
    int rows = in_image.rows;
    int cols = in_image.cols-1;
    
    // create an image slighly smaller
    out_image = Mat(rows, cols, CV_8UC3);
    vector<int> optimalPathArray;
   optimalPathArray = calculateSeams(in_image,BW_image, 'v' );
    //populate the image
    
    
    for(int i=0;i<rows;++i)
        for(int j=0;j<optimalPathArray[i];++j){
//cout<<"ver ---"<<endl;
            Vec3b pixel = in_image.at<Vec3b>(i, j);
            out_image.at<Vec3b>(i,j) = pixel;
        }
    //optimalPathArray[i]
    for(int i=0;i<rows;++i)
        for(int j=optimalPathArray[i];j<cols;++j){
            Vec3b pixel = in_image.at<Vec3b>(i, j+1);
            out_image.at<Vec3b>(i,j) = pixel;
        }
    
    return true;
}
