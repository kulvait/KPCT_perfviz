#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "mkl.h"

#include "SPLINE/SplineFitter.hpp"
#include "matplotlibcpp.h"
#include "utils/Attenuation4DEvaluatorI.hpp"

namespace plt = matplotlibcpp;

namespace KCT::util {

class CTEvaluator : public Attenuation4DEvaluatorI
{
public:
    /**Evaluation of the attenuation values based on CT data with tick files.
     *
     * @param coefficientVolumeFiles CT data files
     * @param tickFiles Annotation of CT data
     * @param allowExtrapolation If true t can be extrapolated, if false t will be restricted to the
     * spline range.
     * @param zeroStartOffset If the reported TACs by Attenuation4DEvaluatorI public functions
     * should be offsetted such that at start of the discretization that corresponds to the time of
     * the acquisition of the first frame from given z stack they should be reported 0.0 or original
     * value if false. Defaults to true.
     * @param minimumAttenuationValue Minimum attenuation value to report by Attenuation4DEvaluatorI
     * public functions, if the attenuation is smaller, this value is reported. Defaults to 0.0.
     */
    CTEvaluator(std::vector<std::string>& coefficientVolumeFiles,
                std::vector<std::string>& tickFiles,
                bool allowExtrapolation = true,
                bool zeroStartOffset = true,
                float minimumAttenuationValue = 0.0);

    /** Destructor of CTEvaluator class
     *
     * Virtual by default.
     * https://stackoverflow.com/a/7403943
     *
     */
    virtual ~CTEvaluator();

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
    void timeSeriesIn(const uint32_t x,
                      const uint32_t y,
                      const uint32_t z,
                      const uint32_t granularity,
                      float* aif) override;

    /**Function to vizualize time series in given point.
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] granularity Number of time points to fill in aif array with.
     */
    void
    vizualizeIn(const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t granularity);

    /**Function to evaluate the value of attenuation at (x,y,z,t).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     */
    float valueAt(const uint32_t x, const uint32_t y, const uint32_t z, const float t) override;

    /**Function to evaluate the value of attenuation for the whole frame (z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     *@param[out] val Prealocated array of size dimx*dimy to put the values of the frame at time t
     *at frame z.
     */
    void frameAt(const uint32_t z, const float t, float* val) override;

    /**Function to evaluate the value of attenuation for the whole volume at point t.
     *
     * @param[in] t Time of evaluation.
     * @param[out] volume Writter to write volume to
     * @param[in] subtractZeroVolume If the values at zero shall be zeros and all other volumes
     * shall be evaluated with respect to this. Good for representing TACs.
     */
    void volumeAt(const float t,
                  std::shared_ptr<io::AsyncFrame2DWritterI<float>> volume,
                  const bool subtractZeroVolume = true) override;

    /**Function to evaluate time series of frames of attenuation coefficients.
     *
     *@param[in] vz Zero based z coordinate of the volume.
     *@param[in] granularity Number of time points to fill in aif array with.
     *@param[out] val Prealocated array to put the values at particular times of the size
     *granularity*dimx*dimy.
     */
    void frameTimeSeries(const uint32_t z, const uint32_t granularity, float* val) override;

    /** Time points in which we have exact information
     *
     *@param[in] z Zero based z coordinate of the volume.
     *
     * @return
     */
    std::vector<double> nativeTimeDiscretization(const uint32_t z);

    /** Values of the fuction in the nativeTimeDiscretization times in a given point.
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *
     * @return
     */
    std::vector<double> nativeValuesIn(const uint32_t x, const uint32_t y, const uint32_t z);

    /**
     * @brief Native attenuation values
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] granularity Number of time points to fill in aif array with.
     *@param[out] val Prealocated array to put the values at particular times of the discretization
     *of the size granularity.
     */
    void timeSeriesNativeNoOffsetNoTruncationIn(const uint32_t x,
                                                const uint32_t y,
                                                const uint32_t z,
                                                const uint32_t granularity,
                                                float* val);

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
    float valueAt_withoutOffset(const uint32_t x, const uint32_t y, const uint32_t z, float t);

    /**Function to evaluate the value of Engineer polynomial without constant at given point
     *(x,y,z,intervalStart).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     */
    float valueAt_intervalStart(const uint32_t x, const uint32_t y, const uint32_t z);

    /**Function to evaluate the value of Engineer polynomial without constant at given frame
     *(z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[out] val Prealocated array to put the values at particular times.
     */
    void frameAt_intervalStart(const uint32_t z, float* val);

    /**
     * Fill breakpoint values for given coordinates.
     *
     * @param x
     * @param y
     */
    void fillBreakpointsY(const uint32_t x, const uint32_t y);

    uint32_t breakpointsNum;
    float sweepTime;
    float sweepOffset;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> attenuationVolumes;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> tickData;
    uint32_t dimx, dimy, dimz;
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
    uint32_t storedZ;
    /** Function updates the framesStored to match given z frame.
     *
     *@param[in] z Frame to update.
     *
     */
    void updateStoredVals(const uint32_t z);

    bool allowExtrapolation;
    bool zeroStartOffset;
    float minimumAttenuationValue;
};

/*The interval starts by the maximum time value of the first volume and ends by the minimum time
 * value of the last volume*/
CTEvaluator::CTEvaluator(std::vector<std::string>& attenuationVolumeFiles,
                         std::vector<std::string>& tickFiles,
                         bool allowExtrapolation,
                         bool zeroStartOffset,
                         float minimumAttenuationValue)
    : Attenuation4DEvaluatorI(0.0, 1.0)
    , allowExtrapolation(allowExtrapolation)
    , zeroStartOffset(zeroStartOffset)
    , minimumAttenuationValue(minimumAttenuationValue)
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
        throw std::runtime_error(msg);
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

void CTEvaluator::vizualizeIn(const uint32_t x,
                              const uint32_t y,
                              const uint32_t z,
                              const uint32_t granularity)
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
}

std::vector<double>
CTEvaluator::nativeValuesIn(const uint32_t x, const uint32_t y, const uint32_t z)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    std::vector<double> values;
    updateStoredVals(z);
    fillBreakpointsY(x, y);
    for(uint32_t i = 0; i != breakpointsNum; i++)
    {
        values.push_back(breakpointsY[i]);
    }
    return values;
}

std::vector<double> CTEvaluator::nativeTimeDiscretization(const uint32_t z)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredVals(z);
    std::vector<double> nativeTimes;
    for(uint32_t i = 0; i != breakpointsNum; i++)
    {
        nativeTimes.push_back(breakpointsT[i]);
    }
    return nativeTimes;
}

void CTEvaluator::timeSeriesIn(
    const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t granularity, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredDiscretization(granularity);
    updateStoredVals(z);
    fillBreakpointsY(x, y);
    fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
    // See
    // https://software.intel.com/en-us/mkl-developer-reference-c-df-interpolate1d-df-interpolateex1d
    fitter->interpolateAt(granularity, storedTimeDiscretization, storedInterpolationBuffer);
    float startOffset = 0.0;
    if(zeroStartOffset)
    {
        startOffset = storedInterpolationBuffer[0];
    }
    for(uint32_t i = 0; i != granularity; i++)
    {
        // LOGD << io::xprintf("The granularity %d that corresponds to t=%f the value is %f.", i,
        //                    storedTimeDiscretization[i], storedInterpolationBuffer[i]);
        val[i]
            = std::max(minimumAttenuationValue, float(storedInterpolationBuffer[i] - startOffset));
    }
}

void CTEvaluator::timeSeriesNativeNoOffsetNoTruncationIn(
    const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t granularity, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredDiscretization(granularity);
    updateStoredVals(z);
    fillBreakpointsY(x, y);
    fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
    // See
    // https://software.intel.com/en-us/mkl-developer-reference-c-df-interpolate1d-df-interpolateex1d
    fitter->interpolateAt(granularity, storedTimeDiscretization, storedInterpolationBuffer);
    for(uint32_t i = 0; i != granularity; i++)
    {
        // LOGD << io::xprintf("The granularity %d that corresponds to t=%f the value is %f.", i,
        //                    storedTimeDiscretization[i], storedInterpolationBuffer[i]);
        val[i] = storedInterpolationBuffer[i];
    }
}

void CTEvaluator::fillBreakpointsY(const uint32_t x, const uint32_t y)
{
    for(uint32_t i = 0; i != breakpointsNum; i++)
    {
        breakpointsY[i] = storedVals[i]->get(x, y);
    }
}

float CTEvaluator::valueAt(const uint32_t x, const uint32_t y, const uint32_t z, const float t)
{
    float startOffset = 0.0;
    if(zeroStartOffset)
    {
        startOffset = valueAt_intervalStart(x, y, z);
    }
    double at;
    if(allowExtrapolation)
    {
        at = (double)t;
    } else
    {
        at = std::max(breakpointsT[0], std::min(breakpointsT[breakpointsNum - 1], (double)t));
    }
    float v = valueAt_withoutOffset(x, y, z, at);
    return std::max(minimumAttenuationValue, v - startOffset);
}

void CTEvaluator::updateStoredVals(const uint32_t z)
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
            storedTimes.push_back(t);
            storedVals.push_back(attenuationVolumes[i]->readFrame(z));
        }
        storedZ = z;
    }
}

float CTEvaluator::valueAt_intervalStart(const uint32_t x, const uint32_t y, const uint32_t z)
{

    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredVals(z);
    return storedVals[0]->get(x, y);
}

float CTEvaluator::valueAt_withoutOffset(const uint32_t x,
                                         const uint32_t y,
                                         const uint32_t z,
                                         const float t)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredVals(z);
    fillBreakpointsY(x, y);
    fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
    // See
    // https://software.intel.com/en-us/mkl-developer-reference-c-df-interpolate1d-df-interpolateex1d
    double val, at;
    if(allowExtrapolation)
    {
        at = (double)t;
    } else
    {
        at = std::max(breakpointsT[0], std::min(breakpointsT[breakpointsNum - 1], (double)t));
    }

    fitter->interpolateAt(1, &at, &val);
    return (float)val;
}

void CTEvaluator::frameAt(const uint32_t z, const float t, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredVals(z);
    double v, at;
    if(allowExtrapolation)
    {
        at = (double)t;
    } else
    {
        at = std::max(breakpointsT[0], std::min(breakpointsT[breakpointsNum - 1], (double)t));
    }
    float startOffset = 0.0;
    for(uint32_t y = 0; y != dimy; y++)
    {
        for(uint32_t x = 0; x != dimx; x++)
        {
            if(zeroStartOffset)
            {
                startOffset = storedVals[0]->get(x, y);
            }
            fillBreakpointsY(x, y);
            fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
            fitter->interpolateAt(1, &at, &v);
            val[y * dimx + x] = std::max(minimumAttenuationValue, float(v - startOffset));
        }
    }
}

void CTEvaluator::frameTimeSeries(const uint32_t z, const uint32_t granularity, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredDiscretization(granularity);
    updateStoredVals(z);
    float startOffset = 0.0;
    for(uint32_t x = 0; x != dimx; x++)
    {
        for(uint32_t y = 0; y != dimy; y++)
        {
            fillBreakpointsY(x, y);
            fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
            // See
            // https://software.intel.com/en-us/mkl-developer-reference-c-df-interpolate1d-df-interpolateex1d
            fitter->interpolateAt(granularity, storedTimeDiscretization, storedInterpolationBuffer);
            if(zeroStartOffset)
            {
                startOffset = storedInterpolationBuffer[0];
            }
            for(uint32_t i = 0; i != granularity; i++)
            {
                val[y * dimx + x + i * dimx * dimy] = std::max(
                    minimumAttenuationValue, float(storedInterpolationBuffer[i] - startOffset));
            }
        }
    }
}

void CTEvaluator::volumeAt(const float t,
                           std::shared_ptr<io::AsyncFrame2DWritterI<float>> volume,
                           const bool subtractZeroVolume)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    double v, at;
    if(allowExtrapolation)
    {
        at = (double)t;
    } else
    {
        at = std::max(breakpointsT[0], std::min(breakpointsT[breakpointsNum - 1], (double)t));
    }
    float startOffset = 0.0;
    io::BufferedFrame2D<float> frame(float(0), dimx, dimy);
    for(uint32_t z = 0; z != dimz; z++)
    {
        updateStoredVals(z);
        for(uint32_t y = 0; y != dimy; y++)
        {
            for(uint32_t x = 0; x != dimx; x++)
            {
                fillBreakpointsY(x, y);
                fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
                fitter->interpolateAt(1, &at, &v);
                if(subtractZeroVolume)//Ignoring zeroStartOffset value of the object
                {
                    startOffset = storedVals[0]->get(x, y);
                    frame.set(v - startOffset, x, y);
                } else
                {
                    frame.set(v, x, y);
                }
            }
        }
        volume->writeFrame(frame, z);
    }
}

} // namespace KCT::util
