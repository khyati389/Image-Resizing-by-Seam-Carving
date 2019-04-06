/**
 * COMP 6651 : project2 Seam - Carving
 * Khyatibahen Chaudhary 40071098
 * REFERENCES: http://www.cs.middlebury.edu/~dsilver/vision/seam-carving/
 * http://answers.opencv.org/question/27248/max-and-min-values-in-a-mat/
 * https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html
 */

#include "sc.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

int ENERGYMAP_ROWS;
int ENERGYMAP_COLS;

using namespace cv;
using namespace std;

//1st. Energy map
void energyImageGeneration(Mat& inputImage, Mat& BW_image) {
    Mat gray_image;
	double value = 1.0/255.0;
    //laplacian function
    GaussianBlur(inputImage, gray_image, Size(3, 3), 0, 0, BORDER_DEFAULT);
    cvtColor(gray_image, gray_image, CV_RGB2GRAY);
    Laplacian(gray_image, BW_image, CV_16S, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(BW_image, BW_image);
    
    BW_image.convertTo(BW_image, CV_64F, value);
    //imshow("gradient image", BW_image);
}

//3rd. find seams
vector<int> findOptimalPath(Mat& energyMap, vector<int> optimalPath, char seam_location) {
	
    double maximum = 2147483647.0;
    double minIndex;
	int minimum_Index;
    if(seam_location == 'v') {
        double min_value, max_value;
        for(int minColumn = 0; minColumn<ENERGYMAP_COLS; minColumn++) {
            if(energyMap.at<double>(ENERGYMAP_ROWS-1, minColumn) < maximum) {
                maximum = energyMap.at<double>(ENERGYMAP_ROWS-1, minColumn);
                minIndex = minColumn;
            }
        }
        
        minimum_Index = minIndex;
        optimalPath[ENERGYMAP_ROWS - 1] = minimum_Index;
		int rows = ENERGYMAP_ROWS-2;
       //backtrack from bottom to top
        while(rows>=0) {
            double pix_a = energyMap.at<double>(rows, max(minimum_Index - 1, 0));
            double pix_b = energyMap.at<double>(rows, minimum_Index);
            double pix_c = energyMap.at<double>(rows, min(minimum_Index + 1, ENERGYMAP_COLS - 1));
			double minPointValue = std::min(pix_a, std::min(pix_b,pix_c));
			
			if(minPointValue == pix_a){
				minimum_Index = std::max(minimum_Index - 1, 0);
			}else if(minPointValue == pix_b){
				minimum_Index = minimum_Index;
			}else if(minPointValue == pix_c){
				minimum_Index = std::min(minimum_Index + 1, ENERGYMAP_COLS - 1);
			}
            optimalPath.at(rows) = minimum_Index;
			rows--;
        }
    }
    else if(seam_location == 'h') {
        Mat table = energyMap.col(ENERGYMAP_COLS - 1);
        double min_value, max_value;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(table, &min_value, &max_value, &min_loc, &max_loc);
        int minimum_Index = min_loc.y;
        optimalPath.at(ENERGYMAP_COLS - 1) = minimum_Index;
		int cols = ENERGYMAP_COLS - 2;
        //backtrack from right to left
		while(cols >=0){
            double pix_a = energyMap.at<double>(std::max(minimum_Index - 1, 0), cols);
            double pix_b = energyMap.at<double>(minimum_Index, cols);
            double pix_c = energyMap.at<double>(std::min(minimum_Index + 1, ENERGYMAP_ROWS - 1), cols);
			double minPointValue = std::min(pix_a, std::min(pix_b,pix_c));
			
			if(minPointValue == pix_a){
				minimum_Index = std::max(minimum_Index - 1, 0);
			}else if(minPointValue == pix_b){
				minimum_Index = minimum_Index;
			}else if(minPointValue == pix_c){
				minimum_Index = std::min(minimum_Index + 1, ENERGYMAP_ROWS - 1);
			}
            optimalPath.at(cols) = minimum_Index;
			cols--;
        }
    }

    return optimalPath;
}

//2nd. cummulative energy/ dynamic programming
vector<int> calculateSeams(Mat& inputImage, Mat& imageIntense, char seam_location) {
	ENERGYMAP_ROWS = imageIntense.rows;
	ENERGYMAP_COLS = imageIntense.cols;
   
    vector<int> optimalPath;
    Mat energyMap = Mat::zeros(ENERGYMAP_ROWS, ENERGYMAP_COLS, CV_64F);

    if(seam_location == 'h') {
		Mat firstEnergyRow = imageIntense.row(0);
        firstEnergyRow.copyTo(energyMap.row(0));
//cout<<"inside horizontal"<<endl;
        optimalPath.resize(ENERGYMAP_COLS);
        for(int column = 1; column < ENERGYMAP_COLS; column++) {
            for(int row = 0; row < ENERGYMAP_ROWS; row++) {
                if(row == 0)
                    energyMap.at<double>(row,column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row, column-1), energyMap.at<double>(row+1, column-1));
                else if(row == ENERGYMAP_ROWS -1) {
                    energyMap.at<double>(row,column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row, column-1),energyMap.at<double>(row-1, column-1));
                }
                else {
                    energyMap.at<double>(row, column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row-1, column-1), min(energyMap.at<double>(row, column-1), 				energyMap.at<double>(row+1, column-1)));
                }
            }
        }

        optimalPath = findOptimalPath(energyMap, optimalPath, seam_location);
    }
    else if(seam_location == 'v') {
		Mat firstEnergyCol = imageIntense.col(0);
        firstEnergyCol.copyTo(energyMap.col(0));
        //cout<<"inside VERtical"<<endl;
        optimalPath.resize(ENERGYMAP_ROWS);

        for(int row = 1; row < ENERGYMAP_ROWS; row++) {
            for(int column = 0; column < ENERGYMAP_COLS; column++) {
                if(column == 0) {
                    energyMap.at<double>(row,column) =imageIntense.at<double>(row, column) + min(energyMap.at<double>(row-1, column), energyMap.at<double>(row-1, column+1));
                }
                else if(column == ENERGYMAP_COLS -1) {
                    energyMap.at<double>(row, column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row-1, column-1),energyMap.at<double>(row-1, column));
                }
                else {
                    energyMap.at<double>(row, column) = imageIntense.at<double>(row, column) + min(energyMap.at<double>(row-1, column-1), min(energyMap.at<double>(row-1, column), energyMap.at<double>(row-1,column+1)));
                }
            }
        }
        optimalPath = findOptimalPath(energyMap, optimalPath, seam_location);
    }

    return optimalPath;
}


bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image) {

    // some sanity checks
    if(new_width>in_image.cols) {
        cout<<"Invalid request!!! new_width has to be smaller than the current size!"<<endl;
        return false;
    }
    if(new_height>in_image.rows) {
        cout<<"Invalid request!!! ne_height has to be smaller than the current size!"<<endl;
        return false;
    }

    if(new_width<=0) {
        cout<<"Invalid request!!! new_width has to be positive!"<<endl;
        return false;
    }

    if(new_height<=0) {
        cout<<"Invalid request!!! new_height has to be positive!"<<endl;
        return false;
    }
    return seam_carving_trivial(in_image, new_width, new_height, out_image);
}


// seam carves by removing trivial seams
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image) {

    Mat iimage = in_image.clone();
    Mat oimage = in_image.clone();
    Mat BW_image;
    energyImageGeneration(iimage,BW_image);

    while(iimage.rows!=new_height || iimage.cols!=new_width) {
        // horizontal seam if needed
        if(iimage.rows>new_height) {
            reduce_horizontal_seam_trivial(iimage, oimage, BW_image);
            iimage = oimage.clone();
            energyImageGeneration(iimage,BW_image);
        }

        if(iimage.cols>new_width) {
            reduce_vertical_seam_trivial(iimage, oimage, BW_image);
            iimage = oimage.clone();
            energyImageGeneration(iimage,BW_image);
        }
    }
    out_image = oimage.clone();
    return true;
}

// horizontl trivial seam is a seam through the center of the image
bool reduce_horizontal_seam_trivial(Mat& in_image, Mat& out_image, Mat& BW_image) {

    // retrieve the dimensions of the new image
    int rows = in_image.rows-1;
    int cols = in_image.cols;

    // create an image slighly smaller
    out_image = Mat(rows, cols, CV_8UC3);
    vector<int> optimalPathArray;
    optimalPathArray = calculateSeams(in_image,BW_image, 'h' );
//cout<<"khkj"<<endl;
   
    for(int i=0; i<cols; ++i) {
        for(int j=0; j<optimalPathArray[i]; ++j) {
//		cout<<i<<endl;
            Vec3b pixel = in_image.at<Vec3b>(j, i);
            out_image.at<Vec3b>(j,i) = pixel;
        }
    }
    for(int i=0; i<cols; ++i) {
        for(int j=optimalPathArray[i]; j<rows; ++j) {
//cout<<"ljkjkljkl"<<endl;
            Vec3b pixel = in_image.at<Vec3b>(j+1, i);
            out_image.at<Vec3b>(j,i) = pixel;
        }
    }

    return true;
}

// vertical trivial seam is a seam through the center of the image
bool reduce_vertical_seam_trivial(Mat& in_image, Mat& out_image, Mat& BW_image) {
    // retrieve the dimensions of the new image
    int rows = in_image.rows;
    int cols = in_image.cols-1;

    // create an image slighly smaller
    out_image = Mat(rows, cols, CV_8UC3);
    vector<int> optimalPathArray;
    optimalPathArray = calculateSeams(in_image,BW_image, 'v' );
    
    for(int i=0; i<rows; ++i)
        for(int j=0; j<optimalPathArray[i]; ++j) {
//cout<<"ver ---"<<endl;
            Vec3b pixel = in_image.at<Vec3b>(i, j);
            out_image.at<Vec3b>(i,j) = pixel;
        }
    //optimalPathArray[i]
    for(int i=0; i<rows; ++i)
        for(int j=optimalPathArray[i]; j<cols; ++j) {
            Vec3b pixel = in_image.at<Vec3b>(i, j+1);
            out_image.at<Vec3b>(i,j) = pixel;
        }

    return true;
}
