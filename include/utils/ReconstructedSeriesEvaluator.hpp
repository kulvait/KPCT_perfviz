#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "mkl.h"

#include "SPLINE/SplineFitter.hpp"
#include "utils/Attenuation4DEvaluatorI.hpp"

namespace KCT::util {

class ReconstructedSeriesEvaluator : public Attenuation4DEvaluatorI
{
public:
    /**Evaluation of the attenuation values based on the Static reconstructions.
     *
     *@param[in] sampledBasisFunctions Sampled basis functions in a DEN file the number of sampling
     *points is equal to dimx and dimz is a number of functions.
     *@param[in] coefficientVolumeFiles Files with fitted coefficient volumes.
     *@param[in] intervalStart Start time.
     *@param[in] intervalEnd End time.
     */
    ReconstructedSeriesEvaluator(std::vector<std::string>& attenuationVolumeFiles,
                                 float sweepTime,
                                 float sweepOffset);

    ReconstructedSeriesEvaluator(std::string staticDir,
                                 uint32_t sweepCount,
                                 float sweepTime,
                                 float sweepOffset);

    /**Destructor of ReconstructedSeriesEvaluator class
     *
     */
    ~ReconstructedSeriesEvaluator();

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

    /** Time points in which we have exact information
     *
     *
     * @return
     */
    std::vector<double> nativeTimeDiscretization();

    /** Values of the fuction in the nativeTimeDiscretization times in a given point.
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *
     * @return
     */
    std::vector<double> nativeValuesIn(const uint32_t x, const uint32_t y, const uint32_t z);

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

    void timeSeriesNativeNoOffsetNoTruncationIn(const uint32_t x,
                                                const uint32_t y,
                                                const uint32_t z,
                                                const uint32_t granularity,
                                                float* aif);

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

private:
    void initialize(std::vector<std::string>& coefficientVolumeFiles);
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

    std::vector<std::shared_ptr<io::Frame2DI<float>>> storedVals;
    uint32_t storedZ;
    /** Function updates the framesStored to match given z frame.
     *
     *@param[in] z Frame to update.
     *
     */
    void updateStoredVals(const uint32_t z);
};

void ReconstructedSeriesEvaluator::initialize(std::vector<std::string>& attenuationVolumeFiles)
{
    if(attenuationVolumeFiles.size() < 2)
    {
        std::string msg = io::xprintf(
            "Number of files with attenuation information needs to be at least 2 but is %d.",
            attenuationVolumeFiles.size());
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    breakpointsNum = attenuationVolumeFiles.size();
    breakpointsT = new double[breakpointsNum];
    timeDiscretizationDouble(breakpointsNum, breakpointsT);
    breakpointsY = new double[breakpointsNum];
    io::DenFileInfo di(attenuationVolumeFiles[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    std::shared_ptr<io::Frame2DReaderI<float>> pr;
    for(std::size_t i = 0; i != breakpointsNum; i++)
    {
        pr = std::make_shared<io::DenFrame2DReader<float>>(attenuationVolumeFiles[i]);
        attenuationVolumes.push_back(pr);
    }
    for(std::shared_ptr<io::Frame2DReaderI<float>>& f : attenuationVolumes)
    {
        if(f->dimx() != dimx || f->dimy() != dimy || f->dimz() != dimz)
        {
            io::throwerr("There are incompatible coefficients!");
        }
    }
    if(breakpointsNum >= 5)
    {
        fitter = std::make_shared<math::SplineFitter>(attenuationVolumeFiles.size(), DF_PP_CUBIC,
                                                      DF_PP_AKIMA);
    } else if(breakpointsNum == 4)
    {
        fitter = std::make_shared<math::SplineFitter>(attenuationVolumeFiles.size(), DF_PP_CUBIC,
                                                      DF_PP_BESSEL);

    } else
    {
        fitter = std::make_shared<math::SplineFitter>(attenuationVolumeFiles.size(), DF_PP_CUBIC,
                                                      DF_PP_NATURAL);
    }
    storedZ = 1;
    updateStoredVals(0); // Just to have something in thye storedInitVal pointer
}

ReconstructedSeriesEvaluator::ReconstructedSeriesEvaluator(std::string attenuationVolumeDir,
                                                           uint32_t sweepCount,
                                                           float sweepTime,
                                                           float sweepOffset)
    : Attenuation4DEvaluatorI(sweepOffset, (sweepCount - 1) * sweepTime + sweepOffset)
    , sweepTime(sweepTime)
    , sweepOffset(sweepOffset)
{

    std::vector<std::string> attenuationVolumeFiles;
    for(uint32_t i = 0; i != sweepCount; i++)
    {
        attenuationVolumeFiles.emplace_back(
            io::xprintf("%s/RUN%02d.den", attenuationVolumeDir.c_str(), i + 1));
    }
    initialize(attenuationVolumeFiles);
}

ReconstructedSeriesEvaluator::ReconstructedSeriesEvaluator(
    std::vector<std::string>& attenuationVolumeFiles, float sweepTime, float sweepOffset)
    : Attenuation4DEvaluatorI(sweepOffset,
                              sweepOffset + (attenuationVolumeFiles.size() - 1) * sweepTime)
    , sweepTime(sweepTime)
    , sweepOffset(sweepOffset)
{
    initialize(attenuationVolumeFiles);
}

ReconstructedSeriesEvaluator::~ReconstructedSeriesEvaluator()
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
}

void ReconstructedSeriesEvaluator::timeDiscretization(const uint32_t granularity, float* timePoints)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    for(uint32_t i = 0; i != granularity; i++)
    {
        timePoints[i] = float(time);
        time += increment;
    }
}

std::vector<double>
ReconstructedSeriesEvaluator::nativeValuesIn(const uint32_t x, const uint32_t y, const uint32_t z)
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

std::vector<double> ReconstructedSeriesEvaluator::nativeTimeDiscretization()
{
    std::vector<double> nativeTimes;
    for(uint32_t i = 0; i != breakpointsNum; i++)
    {
        nativeTimes.push_back(breakpointsT[i]);
    }
    return nativeTimes;
}

void ReconstructedSeriesEvaluator::timeDiscretizationDouble(const uint32_t granularity,
                                                            double* timePoints)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    for(uint32_t i = 0; i != granularity; i++)
    {
        timePoints[i] = time;
        time += increment;
    }
}

void ReconstructedSeriesEvaluator::updateStoredDiscretization(const uint32_t granularity)
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

bool ReconstructedSeriesEvaluator::isTimeDiscretizedEvenly() { return true; }

void ReconstructedSeriesEvaluator::timeSeriesNativeNoOffsetNoTruncationIn(
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
        val[i] = float(storedInterpolationBuffer[i]);
    }
}

void ReconstructedSeriesEvaluator::timeSeriesIn(
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
    float v0 = storedInterpolationBuffer[0];
    for(uint32_t i = 0; i != granularity; i++)
    {
        val[i] = std::max(float(0), float(storedInterpolationBuffer[i] - v0));
    }
}

void ReconstructedSeriesEvaluator::fillBreakpointsY(const uint32_t x, const uint32_t y)
{
    for(uint32_t i = 0; i != breakpointsNum; i++)
    {
        breakpointsY[i] = storedVals[i]->get(x, y);
    }
}

float ReconstructedSeriesEvaluator::valueAt(const uint32_t x,
                                            const uint32_t y,
                                            const uint32_t z,
                                            const float t)
{
    float val0 = valueAt_intervalStart(x, y, z);
    float v = valueAt_withoutOffset(x, y, z, t);
    return std::max(float(0), v - val0);
}

void ReconstructedSeriesEvaluator::updateStoredVals(const uint32_t z)
{

    if(storedZ != z)
    {
        storedVals.clear();
        for(uint32_t i = 0; i != breakpointsNum; i++)
        {
            storedVals.push_back(attenuationVolumes[i]->readFrame(z));
        }
        storedZ = z;
    }
}

float ReconstructedSeriesEvaluator::valueAt_intervalStart(const uint32_t x,
                                                          const uint32_t y,
                                                          const uint32_t z)
{

    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredVals(z);
    return storedVals[0]->get(x, y);
}

float ReconstructedSeriesEvaluator::valueAt_withoutOffset(const uint32_t x,
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
    double val;
    double at = (double)t;

    fitter->interpolateAt(1, &at, &val);
    return (float)val;
}

void ReconstructedSeriesEvaluator::frameAt(const uint32_t z, const float t, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateStoredVals(z);
    double v;
    double at = (double)t;
    for(uint32_t y = 0; y != dimy; y++)
    {
        for(uint32_t x = 0; x != dimx; x++)
        {
            fillBreakpointsY(x, y);
            fitter->buildSpline(breakpointsT, breakpointsY, bc_type, bc);
            fitter->interpolateAt(1, &at, &v);
            val[y * dimx + x] = std::max(float(0), float(v - storedVals[0]->get(x, y)));
        }
    }
}

void ReconstructedSeriesEvaluator::volumeAt(const float t,
                                            std::shared_ptr<io::AsyncFrame2DWritterI<float>> volume,
                                            const bool subtractZeroVolume)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    double v, at;
    at = (double)t;
    float startOffset = 0.0f;
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
                if(subtractZeroVolume)
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

void ReconstructedSeriesEvaluator::frameTimeSeries(const uint32_t z,
                                                   const uint32_t granularity,
                                                   float* val)
{
    updateStoredDiscretization(granularity);
    std::vector<std::shared_ptr<io::Frame2DI<float>>> localFrames;
    double* breakpointsYLocal = new double[breakpointsNum];
    double* storedInterpolationBufferLocal = new double[granularity];
    double* storedTimeDiscretizationLocal = new double[granularity];
    timeDiscretizationDouble(granularity, storedTimeDiscretizationLocal);

    std::shared_ptr<math::SplineFitter> fitter
        = std::make_shared<math::SplineFitter>(attenuationVolumes.size(), DF_PP_CUBIC, DF_PP_AKIMA);
    for(uint32_t i = 0; i != breakpointsNum; i++)
    {
        localFrames.push_back(attenuationVolumes[i]->readFrame(z));
    }
    for(uint32_t x = 0; x != dimx; x++)
    {
        for(uint32_t y = 0; y != dimy; y++)
        {
            for(uint32_t i = 0; i != breakpointsNum; i++)
            {
                breakpointsYLocal[i] = localFrames[i]->get(x, y);
            }
            fitter->buildSpline(breakpointsT, breakpointsYLocal, bc_type, bc);
            // See
            // https://software.intel.com/en-us/mkl-developer-reference-c-df-interpolate1d-df-interpolateex1d
            fitter->interpolateAt(granularity, storedTimeDiscretizationLocal,
                                  storedInterpolationBufferLocal);
            float v0 = storedInterpolationBufferLocal[0];
            for(uint32_t i = 0; i != granularity; i++)
            {
                val[y * dimx + x + i * dimx * dimy]
                    = std::max(float(0), float(storedInterpolationBufferLocal[i] - v0));
            }
        }
    }
    delete[] breakpointsYLocal;
    delete[] storedInterpolationBufferLocal;
    delete[] storedTimeDiscretizationLocal;
}
} // namespace KCT::util
