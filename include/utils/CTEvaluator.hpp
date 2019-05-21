#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "mkl.h"

#include "SPLINE/SplineFitter.hpp"
#include "utils/Attenuation4DEvaluatorI.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

namespace CTL::util {

class CTEvaluator : public Attenuation4DEvaluatorI
{
public:
    /**Evaluation of the attenuation values based on the Engineered basis.
     *
     *@param[in] sampledBasisFunctions Sampled basis functions in a DEN file the number of sampling
     *points is equal to dimx and dimz is a number of functions.
     *@param[in] coefficientVolumeFiles Files with fitted coefficient volumes.
     *@param[in] intervalStart Start time.
     *@param[in] intervalEnd End time.
     */
    CTEvaluator(std::vector<std::string>& coefficientVolumeFiles,
                std::vector<std::string>& tickFiles);

    /**Destructor of CTEvaluator class
     *
     */
    ~CTEvaluator();

    /**Function to obtain time discretization as float array.
     *
     *This function evaluates the time instants in which other functions of the implementing class
     *fills the arrays with values in that times.
     *@param[in] granularity Number of time points to fill in timePoints array with.
     *@param[out] timePoints Time discretization.
     */
    void timeDiscretization(const uint32_t granularity, float* timePoints) override;

    /**Test if the time is discretized evenly.
     *
     * This property is required for assembling pseudoinverse matrix.
     */
    bool isTimeDiscretizedEvenly() override;

    /**Function to evaluate time series in given point.
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] granularity Number of time points to fill in aif array with.
     *@param[out] aif Prealocated array to put the values at particular times.
     */
    void timeSeriesIn(const uint16_t x,
                      const uint16_t y,
                      const uint16_t z,
                      const uint32_t granularity,
                      float* aif) override;

    /**Function to evaluate the value of attenuation at (x,y,z,t).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     */
    float valueAt(const uint16_t x, const uint16_t y, const uint16_t z, const float t) override;

    /**Function to evaluate the value of attenuation for the whole frame (z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     *@param[out] val Prealocated array of size dimx*dimy to put the values of the frame at time t
     *at frame z.
     */
    void frameAt(const uint16_t z, const float t, float* val) override;

    /**Function to evaluate time series of frames of attenuation coefficients.
     *
     *@param[in] vz Zero based z coordinate of the volume.
     *@param[in] granularity Number of time points to fill in aif array with.
     *@param[out] val Prealocated array to put the values at particular times of the size
     *granularity*dimx*dimy.
     */
    void frameTimeSeries(const uint16_t z, const uint32_t granularity, float* val) override;

private:
    /**Function to obtain time discretization as double array.
     *
     *This function evaluates the time instants in which other functions of the implementing class
     *fills the arrays with values in that times.
     *@param[in] granularity Number of time points to fill in timePoints array with.
     *@param[out] timePoints Time discretization.
     */
    void timeDiscretizationDouble(const uint32_t granularity, double* timePoints);
    /**Function to evaluate the value of Engineer polynomial without constant at given point
     *(x,y,z,t).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     */
    float valueAt_withoutOffset(const uint16_t x, const uint16_t y, const uint16_t z, float t);

    /**Function to evaluate the value of Engineer polynomial without constant at given point
     *(x,y,z,intervalStart).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     */
    float valueAt_intervalStart(const uint16_t x, const uint16_t y, const uint16_t z);

    /**Function to evaluate the value of Engineer polynomial without constant at given frame
     *(z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[out] val Prealocated array to put the values at particular times.
     */
    void frameAt_intervalStart(const uint16_t z, float* val);

    /**
     * Fill breakpoint values for given coordinates.
     *
     * @param x
     * @param y
     */
    void fillBreakpointsY(const uint16_t x, const uint16_t y);

    uint16_t breakpointsNum;
    float sweepTime;
    float sweepOffset;
    float intervalStart, intervalEnd;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> attenuationVolumes;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> tickData;
    uint16_t dimx, dimy, dimz;
    std::shared_ptr<math::SplineFitter> fitter;

    // Spline fitting settings
    MKL_INT bc_type = DF_BC_1ST_LEFT_DER | DF_BC_1ST_RIGHT_DER;
    double bc[2] = { 0.0, 0.0 };
    double* breakpointsT = nullptr;
    double* breakpointsY = nullptr;

    // Global parameters to support faster operations for certain functions.
    std::mutex globalsAccess;
    uint32_t storedGranularity = 0;
    double* storedTimeDiscretization = nullptr;
    double* storedInterpolationBuffer = nullptr;

    /** Function updates the stored discretization to match given granularity.
     *
     *@param[in] t Time to update.
     *
     */
    void updateStoredDiscretization(const uint32_t granularity);

    /**Current processed slice*/
    std::vector<std::shared_ptr<io::Frame2DI<float>>> storedVals;
    /**Times of current processed slice*/
    std::vector<float> storedTimes;
    /**Z coordinate of current processed slice*/
    uint16_t storedZ;
    /** Function updates the framesStored to match given z frame.
     *
     *@param[in] z Frame to update.
     *
     */
    void updateStoredVals(const uint16_t z);
};

/*The interval starts by the maximum time value of the first volume and ends by the minimum time
 * value of the last volume*/
CTEvaluator::CTEvaluator(std::vector<std::string>& attenuationVolumeFiles,
                         std::vector<std::string>& tickFiles)
{

    if(attenuationVolumeFiles.size() < 2)
    {
        io::throwerr(
            "Number of files with attenuation information needs to be at least 2 but is %d.",
            attenuationVolumeFiles.size());
    }
    breakpointsNum = attenuationVolumeFiles.size();
    breakpointsT = new double[breakpointsNum];
    breakpointsY = new double[breakpointsNum];
    io::DenFileInfo di(attenuationVolumeFiles[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    std::shared_ptr<io::Frame2DReaderI<float>> pr;
    std::shared_ptr<io::Frame2DReaderI<float>> tickr;
    for(std::size_t i = 0; i != breakpointsNum; i++)
    {
        pr = std::make_shared<io::DenFrame2DReader<float>>(attenuationVolumeFiles[i]);
        tickr = std::make_shared<io::DenFrame2DReader<float>>(tickFiles[i]);
        attenuationVolumes.push_back(pr);
        tickData.push_back(tickr);
    }
    for(std::shared_ptr<io::Frame2DReaderI<float>>& f : attenuationVolumes)
    {
        if(f->dimx() != dimx || f->dimy() != dimy || f->dimz() != dimz)
        {
            io::throwerr("There are incompatible coefficients!");
        }
    }
    std::shared_ptr<io::Frame2DReaderI<float>>& startData = tickData[0];
    std::shared_ptr<io::Frame2DReaderI<float>>& endData = tickData[breakpointsNum - 1];
    std::shared_ptr<io::Frame2DI<float>> fs, fe;
    fs = startData->readFrame(0);
    fe = endData->readFrame(0);
    // Fifth element of frame is time
    intervalStart = fs->get(4, 0);
    intervalEnd = fe->get(4, 0);
    float start, end;
    for(std::size_t z = 0; z != dimz; z++)
    {
        fs = startData->readFrame(z);
        fe = endData->readFrame(z);
        start = fs->get(4, 0);
        end = fe->get(4, 0);
        if(start > intervalStart)
        {
            intervalStart = start;
        }
        if(end < intervalEnd)
        {
            intervalEnd = end;
        }
    }
    if(!(intervalStart < intervalEnd))
    {
        std::string msg = io::xprintf(
            "Start of interval %f needs to be strictly before its end %f but its not!",
            intervalStart, intervalEnd);
        LOGE << msg;
        throw new std::runtime_error(msg);
    }
    for(std::shared_ptr<io::Frame2DReaderI<float>>& f : attenuationVolumes)
    {
        if(f->dimx() != dimx || f->dimy() != dimy || f->dimz() != dimz)
        {
            io::throwerr("There are incompatible coefficients!");
        }
    }
    fitter = std::make_shared<math::SplineFitter>(attenuationVolumeFiles.size(), DF_PP_CUBIC,
                                                  DF_PP_AKIMA);
    storedZ = 1;
    updateStoredVals(0); // Just to have something in thye storedInitVal pointer
}

CTEvaluator::~CTEvaluator()
{
    if(storedTimeDiscretization != nullptr)
    {
        delete[] storedTimeDiscretization;
        storedTimeDiscretization = nullptr;
    }
    if(storedInterpolationBuffer != nullptr)
    {
        delete[] storedInterpolationBuffer;
        storedInterpolationBuffer = nullptr;
    }
    if(breakpointsT != nullptr)
    {
        delete[] breakpointsT;
        breakpointsT = nullptr;
    }
    if(breakpointsY != nullptr)
    {
        delete breakpointsY;
        breakpointsY = nullptr;
    }
    storedVals.clear();
    storedTimes.clear();
}

void CTEvaluator::timeDiscretization(const uint32_t granularity, float* timePoints)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    for(uint32_t i = 0; i != granularity; i++)
    {
        timePoints[i] = float(time);
        time += increment;
    }
}

void CTEvaluator::timeDiscretizationDouble(const uint32_t granularity, double* timePoints)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    for(uint32_t i = 0; i != granularity; i++)
    {
        timePoints[i] = time;
        time += increment;
    }
}

void CTEvaluator::updateStoredDiscretization(const uint32_t granularity)
{
    if(granularity != storedGranularity)
    {
        if(storedTimeDiscretization != nullptr)
        {
            delete[] storedTimeDiscretization;
            storedTimeDiscretization = nullptr;
        }
        if(storedInterpolationBuffer != nullptr)
        {
            delete[] storedInterpolationBuffer;
            storedInterpolationBuffer = nullptr;
        }
        storedTimeDiscretization = new double[granularity];
        storedInterpolationBuffer = new double[granularity];
        timeDiscretizationDouble(granularity, storedTimeDiscretization);
        storedGranularity = granularity;
    }
}

bool CTEvaluator::isTimeDiscretizedEvenly() { return true; }

void CTEvaluator::timeSeriesIn(
    const uint16_t x, const uint16_t y, const uint16_t z, const uint32_t granularity, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredDiscretization(granularity);
    updateStoredVals(z);
    fillBreakpointsY(x, y);
    LOGD << io::xprintf("Investigating point (x,y,z) = (%d, %d, %d)", x, y, z);
    LOGW << io::xprintf("The AIF consists of the values (t_0,a[0])=(%f,%f) (t_1, a[1])=(%f, %f), "
                        "(t_2,a[2])=(%f, %f), (t_3,a[3])=(%f, %f), (t_4,a[4])=(%f, %f)"
                        "(t_5,a[5])=(%f, %f), (t_6,a[6])=(%f, %f), (t_7,a[7])=(%f, %f)"
                        "(t_8,a[8])=(%f, %f), (t_9,a[9])=(%f, %f), (t_10,a[10])=(%f, %f)",
                        breakpointsT[0], breakpointsY[0], breakpointsT[1], breakpointsY[1],
                        breakpointsT[2], breakpointsY[2], breakpointsT[3], breakpointsY[3],
                        breakpointsT[4], breakpointsY[4], breakpointsT[5], breakpointsY[5],
                        breakpointsT[6], breakpointsY[6], breakpointsT[7], breakpointsY[7],
                        breakpointsT[8], breakpointsY[8], breakpointsT[9], breakpointsY[9],
                        breakpointsT[10], breakpointsY[10]);
    fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
    // See
    // https://software.intel.com/en-us/mkl-developer-reference-c-df-interpolate1d-df-interpolateex1d
    fitter->interpolateAt(granularity, storedTimeDiscretization, storedInterpolationBuffer);
        std::vector<double> taxis;
        std::vector<double> plotme;
        for(uint32_t i = 0; i != granularity; i++)
        {
            plotme.push_back(storedInterpolationBuffer[i]);
            taxis.push_back(storedTimeDiscretization[i]);
        }
        plt::plot(taxis, plotme);
        std::vector<double> taxis_scatter;
        std::vector<double> plotme_scatter;
	for(uint32_t i = 0; i != storedVals.size(); i++)
	{
		taxis_scatter.push_back(breakpointsT[i]);
		plotme_scatter.push_back(breakpointsY[i]);
	}
	plt::plot(taxis_scatter, plotme_scatter);
        plt::show();
    float v0 = storedInterpolationBuffer[0];
    for(uint32_t i = 0; i != granularity; i++)
    {
	LOGD << io::xprintf("The granularity %d that corresponds to t=%f the value is %f.", i, storedTimeDiscretization[i], storedInterpolationBuffer[i]);
        val[i] = std::max(float(0), float(storedInterpolationBuffer[i] - v0));
    }
}

void CTEvaluator::fillBreakpointsY(const uint16_t x, const uint16_t y)
{
    for(uint32_t i = 0; i != breakpointsNum; i++)
    {
        breakpointsY[i] = storedVals[i]->get(x, y);
    }
}

float CTEvaluator::valueAt(const uint16_t x, const uint16_t y, const uint16_t z, const float t)
{
    float val0 = valueAt_withoutOffset(x, y, z, intervalStart);
    float v = valueAt_withoutOffset(x, y, z, t);
    return std::max(float(0), v - val0);
}

void CTEvaluator::updateStoredVals(const uint16_t z)
{

    if(storedZ != z)
    {
        storedVals.clear();
        storedTimes.clear();
        float t;
        for(uint32_t i = 0; i != breakpointsNum; i++)
        {
            t = tickData[i]->readFrame(z)->get(4, 0);
            breakpointsT[i] = t;
            storedTimes.push_back(tickData[i]->readFrame(z)->get(4, 0));
            storedVals.push_back(attenuationVolumes[i]->readFrame(z));
        }
        storedZ = z;
    }
}

float CTEvaluator::valueAt_intervalStart(const uint16_t x, const uint16_t y, const uint16_t z)
{

    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredVals(z);
    return storedVals[0]->get(x, y);
}

float CTEvaluator::valueAt_withoutOffset(const uint16_t x,
                                         const uint16_t y,
                                         const uint16_t z,
                                         const float t)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredVals(z);
    fillBreakpointsY(x, y);
    fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
    // See
    // https://software.intel.com/en-us/mkl-developer-reference-c-df-interpolate1d-df-interpolateex1d
    double val;
    double at = (double)t;

    fitter->interpolateAt(1, &at, &val);
    return (float)val;
}

void CTEvaluator::frameAt(const uint16_t z, const float t, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredVals(z);
    double v;
    double at = (double)t;
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            fillBreakpointsY(x, y);
            fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
            fitter->interpolateAt(1, &at, &v);
            val[y * dimx + x] = std::min(float(0), float(v - storedVals[0]->get(x, y)));
        }
    }
}

void CTEvaluator::frameTimeSeries(const uint16_t z, const uint32_t granularity, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredDiscretization(granularity);
    updateStoredVals(z);
    for(int x = 0; x != dimx; x++)
    {
        for(int y = 0; y != dimy; y++)
        {
            fillBreakpointsY(x, y);
            fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
            // See
            // https://software.intel.com/en-us/mkl-developer-reference-c-df-interpolate1d-df-interpolateex1d
            fitter->interpolateAt(granularity, storedTimeDiscretization, storedInterpolationBuffer);
            float v0 = storedInterpolationBuffer[0];
            for(uint32_t i = 0; i != granularity; i++)
            {
                val[y * dimx + x + i * dimx * dimy]
                    = std::max(float(0), float(storedInterpolationBuffer[i] - v0));
            }
        }
    }
}
} // namespace CTL::util
