/**
 * COMP 6651
 * Khyatibahen Chaudhary 
 * REFERENCES: http://www.cs.middlebury.edu/~dsilver/vision/seam-carving/
 * http://answers.opencv.org/question/27248/max-and-min-values-in-a-mat/
 * https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html
 */

#include "sc.h"

using namespace cv;
using namespace std;


//1st. Energy map
void energyImageGeneration(Mat& inputImage, Mat& BW_image){
    Mat gray_image;

    //laplacian function
	GaussianBlur(inputImage, gray_image, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(inputImage, gray_image, CV_RGB2GRAY);
	Laplacian(gray_image, BW_image, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(BW_image, BW_image);
    // end of laplacian
    imshow("gradient image", BW_image);
}

//3rd. find seams
int* findOptimalPath(Mat& energyMap, int* optimalPath, char seam_direction){

    if(seam_direction == 'v'){
        //min and max location point in Mat.
        Mat table = energyMap.row(energyMap.rows - 1);
        Point min_loc, max_loc;
        minMaxLoc(table, &min, &max, &min_loc, &max_loc);
        optimalPath[energyImage.rows - 1] = min_loc.x;

        int rows = energyImage.rows - 2;
        for(int from = rows; from > 0; from-- ){
            //backtrack
        }
    }
    else if(seam_direction == 'h'){
        Mat table = energyMap.col(energyMap.cols - 1);
        Point min_loc, max_loc;
        minMaxLoc(energyMap, &min, &max, &min_loc, &max_loc);
        optimalPath[energyImage.cols - 1] = min_loc.y;

        int cols = energyImage.cols - 2;
        for(int from = cols; from > 0; from-- ){
            //backtrack
        }
    }
   
   return optimalPath;
}

//2nd. cummulative energy/ dynamic programming
int* calculateSeams(Mat& inputImage, Mat& imageIntense, char seam_direction){
    int imageRows = imageIntense.rows;
    int imageCols = imageIntense.cols;

    Mat energyMap = Mat(inputImage.rows, inputImage.cols,CV_64F, double(0));
    BW_image.copyTo(energyMap);
    
   if(seam_direction == 'h'){
       int* optimalPath = new int[imageCols];
       for(int column = 1; column < imageCols; column++){
            for(int row = 0; row < imageRows; row++){
                if(row == 0)
                    energyMap.at<double>(row,column) += min(energyMap.at<double>(row, column-1), energyMap.at<double>(row+1, column-1));
                else{
                    energyMap.at<double>(row, column) += min(energyMap.at<double>(row-1, column-1), min(energyMap.at<double>(row, column-1), energyMap.at<double>(min(row+1, imageCols-1), column-1)));
                }       
            }
        }

        optimalPath = findOptimalPath(energyMap, optimalPath, seam_direction);
    }
   else if(seam_direction == 'v'){
       int* optimalPath = new int[imageRows];

       for(int row = 1; row < imageRows; row++){
            for(int column = 0; column < imageCols; column++){
                if(column == 0)
                    energyMap.at<double>(row,column) += min(energyMap.at<double>(row-1, column), energyMap.at<double>(row-1, column+1));
                else{
                    energyMap.at<double>(row, column) += min(energyMap.at<double>(row-1, column-1), min(energyMap.at<double>(row-1, column-1), energyMap.at<double>(row-1, min(column+1, imageRows-1)));
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
    int* optimalPathArray;
    optimalPathArray = calculateSeams(in_image,BW_image, 'h' );
    //populate the image
    int middle = in_image.rows / 2;
    
   for(int i=0;j<cols;++i){
        for(int j=0;j<=optimalPathArray[i];++j){
            Vec3b pixel = in_image.at<Vec3b>(j, i);
            out_image.at<Vec3b>(i,j) = pixel;
        }

        for(int j=optimalPathArray[i]+1;j<rows;++j){
            Vec3b pixel = in_image.at<Vec3b>(j+1, i);
            out_image.at<Vec3b>(i,j) = pixel;
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
    int* optimalPathArray;
    optimalPathArray = calculateSeams(in_image,BW_image, 'v' );
    //populate the image
    int middle = in_image.cols / 2;
    
    for(int i=0;i<rows;++i)
        for(int j=0;j<=optimalPathArray[i];++j){
            Vec3b pixel = in_image.at<Vec3b>(i, j);
            out_image.at<Vec3b>(i,j) = pixel;
        }
    
    for(int i=0;i<rows;++i)
        for(int j=optimalPathArray[i]+1;j<cols;++j){
            Vec3b pixel = in_image.at<Vec3b>(i, j+1);
            out_image.at<Vec3b>(i,j) = pixel;
        }
    
    return true;
}
