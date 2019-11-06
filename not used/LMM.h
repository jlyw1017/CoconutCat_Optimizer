#pragma  once
#include <stdio.h>
#include <opencv2\core\core.hpp>
#pragma comment(lib,"opencv_core248d.lib")

const int  MAXTIME = 50;

using namespace cv;
FileStorage fs;

Mat jacobin(const Mat& pk/*[a,b]*/, const Mat& x); //f = a*exp(-b*x)
Mat yEstimate(const Mat& p, const Mat& x);
inline void outData(FileStorage& fs, Mat & m, char* filename)
{
    fs.open(filename, FileStorage::FORMAT_XML | FileStorage::WRITE);
    char *temp = new char[10];
    strcpy_s(temp, 10,filename);
    *strchr(temp, '.') = '\0';
    fs << temp << m;
    fs.release();
    delete[] temp;
}

void LM(double* p0, int pN, double* x, int xN, double* y, double lamda, double step, double ep = 0.0001)
{
    int iters = 0;
    int updateJ = 1;
    double ek = 0.0, ekk = 0.0;//估计误差
    Mat_<double> xM(xN, 1, x), yM(xN, 1, y), pM(pN, 1, p0), JM, yEM, yEMM, dM, gM, dMM, dpM;//至少需要JM,gM,dpM,pM
    for (; iters < MAXTIME; iters++)
    {
        if (updateJ == 1)
        {
            JM = jacobin(pM, xM);
            //outData(fs, JM, "J.xml");
            yEM = yEstimate(pM, xM);
            dM = yM - yEM;
            gM = JM.t()*dM;
            if (iters == 0)
                ek = dM.dot(dM);
            //outData(fs, dM, "d.xml");
        }
        Mat_<double> NM = JM.t()*JM + lamda*(Mat::eye(pN, pN, CV_64F));
        if (solve(NM, gM, dpM))
        {
            Mat_<double> pMM = pM + dpM;
            yEMM = yEstimate(pMM, xM);
            dMM = yM - yEMM;
            ekk = dMM.dot(dMM);
            //outData(fs, dMM, "dlm.xml");
            //outData(fs, dpM, "dp.xml");
            if (ekk < ek)//成功则更新向量与估计误差
            {
                printf("the %d iterator result\n", iters);
                if (dpM.dot(dpM) < ep)
                {
                    outData(fs, pM, "p.xml");
                    return;
                }
                else
                {
                    pM = pMM;
                    ek = ekk;
                    lamda = lamda / step;
                    updateJ = 1;
                    continue;
                }
            }
            else
            {
                lamda = lamda*step;
                updateJ = 0;
            }
        }
        else
        {
            outData(fs, JM, "badJ.xml");
            //return;
        }
    }
}

Mat jacobin(const Mat& pk/*[a,b]*/, const Mat& x)
{
    Mat_<double> J(x.rows, pk.rows), da, db;
    exp(-pk.at<double>(1)*x, da);
    exp(-pk.at<double>(1)*x, db);
    db = -pk.at<double>(0)*x.mul(db);
    //outData(fs, da, "da.xml");
    //outData(fs, db, "db.xml");
    da.copyTo(J(Rect(0, 0, 1, J.rows)));
    db.copyTo(J(Rect(1, 0, 1, J.rows)));
    return J;
}

Mat yEstimate(const Mat& p, const Mat& x)
{
    Mat_<double> Y(x.rows, x.cols);
    exp(-p.at<double>(1)*x, Y);
    Y = p.at<double>(0)*Y;
    return Y;
}
