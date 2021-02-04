#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "utils/Attenuation4DEvaluatorI.hpp"
#include "FUN/VectorFunctionI.h"
#include "FUN/FourierSeries.hpp"

namespace CTL::util {

class FourierSeriesEvaluator : public Attenuation4DEvaluatorI
{
public:
    /**Evaluation of the attenuation values based on the Legendre coeficients fiting.
     *
     *@param[in] coefficientVolumeFiles Files with fitted coefficient volumes of related functions
     *from given Fourier basis.
     *@param[in] intervalStart Start time.
     *@param[in] intervalEnd End time.
     *@param[in] negativeAsZero Negative value is treated as zero.
     *@param[in] halfPeriodicFunctions Use basis where the first functions have a half period.
     */
    FourierSeriesEvaluator(std::vector<std::string> coefficientVolumeFiles,
                           float intervalStart,
                           float intervalEnd,
                           bool negativeAsZero = true,
                           bool halfPeriodicFunctions = false);

    /**Destructor of FourierSeriesEvaluator class
     *
     */
    ~FourierSeriesEvaluator();

    /**Function to obtain time discretization.
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

    /**Function to evaluate the value of Legendre polynomial without constant at given point
     *(x,y,z,intervalStart).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     */
    float valueAt_intervalStart(const uint16_t x, const uint16_t y, const uint16_t z);

private:
    /**Function to evaluate the value of Legendre polynomial without constant at given point
     *(x,y,z,t).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     */
    float valueAt_withoutOffset(const uint16_t x, const uint16_t y, const uint16_t z, float t);

    /**Function to evaluate the value of Legendre polynomial without constant at given frame
     *(z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     *@param[in] offset Specified offset to subtract and crop at 0.
     *@param[out] val Prealocated array to put the values at particular times.
     */
    void frameAt_customOffset(const uint16_t z, const float t, float* offset, float* val);

    /**Function to evaluate the value of Legendre polynomial without constant at given frame
     *(z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[out] val Prealocated array to put the values at particular times.
     */
    void frameAt_intervalStart(const uint16_t z, float* val);

    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> coefficientVolumes;
    uint32_t basisSize;
    uint16_t dimx, dimy, dimz;
    std::shared_ptr<util::VectorFunctionI> fourierEvaluatorWithoutConstant;
    float* valuesAtStart;
    float* fourierCoefficientsAtIntervalStartWithoutConstant;

    // Global parameters to support faster operations for certain functions.
    std::mutex globalsAccess;
    float* fourierCoefficientsAtStoredTimeWithoutConstant;
    float storedTime;
    /** Function updates the fourierCoefficientsAtStoredTimeWithoutConstant to match given time.
     *
     *@param[in] t Time to update.
     *
     */
    void updateCoefficientsStored(const float t);
    std::vector<std::shared_ptr<io::Frame2DI<float>>> framesStored;
    uint16_t storedZ;
    /** Function updates the framesStored to match given z frame.
     *
     *@param[in] z Frame to update.
     *
     */
    void updateFramesStored(const uint16_t z);
    bool negativeAsZero;
    bool halfPeriodicFunctions;
};

FourierSeriesEvaluator::FourierSeriesEvaluator(std::vector<std::string> coefficientVolumeFiles,
                                               float intervalStart,
                                               float intervalEnd,
                                               bool negativeAsZero,
                                               bool halfPeriodicFunctions)
    : Attenuation4DEvaluatorI(intervalStart, intervalEnd)
    , negativeAsZero(negativeAsZero)
    , halfPeriodicFunctions(halfPeriodicFunctions)
{
    basisSize = coefficientVolumeFiles.size();
    if(basisSize < 2)
    {
        std::string ERR = io::xprintf("There must be at least one more function than constant, "
                                      "menas basisSize >= 2, but basisSize =%d.",
                                      basisSize);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    std::shared_ptr<io::Frame2DReaderI<float>> pr;
    for(std::size_t i = 0; i < basisSize; i++)
    {
        pr = std::make_shared<io::DenFrame2DReader<float>>(coefficientVolumeFiles[i]);
        coefficientVolumes.push_back(pr);
    }
    io::DenFileInfo di(coefficientVolumeFiles[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();

    for(std::shared_ptr<io::Frame2DReaderI<float>>& f : coefficientVolumes)
    {
        if(f->dimx() != dimx || f->dimy() != dimy || f->dimz() != dimz)
        {

            std::string ERR = "There are incompatible coefficients!";
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
    }
    fourierEvaluatorWithoutConstant = std::make_shared<util::FourierSeries>(
        basisSize, intervalStart, intervalEnd, 1, halfPeriodicFunctions);
    fourierCoefficientsAtStoredTimeWithoutConstant = new float[basisSize - 1];
    fourierCoefficientsAtIntervalStartWithoutConstant = new float[basisSize - 1];
    fourierEvaluatorWithoutConstant->valuesAt(intervalStart,
                                              fourierCoefficientsAtStoredTimeWithoutConstant);
    fourierEvaluatorWithoutConstant->valuesAt(intervalStart,
                                              fourierCoefficientsAtIntervalStartWithoutConstant);
    storedTime = intervalStart;
    storedZ = 0;
    for(uint32_t i = 0; i < basisSize; i++)
    {
        framesStored.push_back(coefficientVolumes[i]->readFrame(storedZ));
    }
    valuesAtStart = new float[dimx * dimy](); // Constructor is not filling this array
}

FourierSeriesEvaluator::~FourierSeriesEvaluator()
{
    if(fourierCoefficientsAtStoredTimeWithoutConstant != nullptr)
    {
        delete[] fourierCoefficientsAtStoredTimeWithoutConstant;
    }
    if(fourierCoefficientsAtIntervalStartWithoutConstant != nullptr)
    {
        delete[] fourierCoefficientsAtIntervalStartWithoutConstant;
    }
    if(valuesAtStart != nullptr)
    {
        delete[] valuesAtStart;
    }
}

void FourierSeriesEvaluator::timeDiscretization(const uint32_t granularity, float* timePoints)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    for(uint32_t i = 0; i != granularity; i++)
    {
        timePoints[i] = float(time);
        time += increment;
    }
}

bool FourierSeriesEvaluator::isTimeDiscretizedEvenly() { return true; }

void FourierSeriesEvaluator::timeSeriesIn(
    const uint16_t x, const uint16_t y, const uint16_t z, const uint32_t granularity, float* val)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    for(uint32_t i = 0; i != granularity; i++)
    {
        val[i] = valueAt(x, y, z, time);
        time += increment;
    }
}

float FourierSeriesEvaluator::valueAt(const uint16_t x,
                                      const uint16_t y,
                                      const uint16_t z,
                                      const float t)
{
    float val0 = valueAt_intervalStart(x, y, z);
    float v = valueAt_withoutOffset(x, y, z, t);
    if(negativeAsZero)
        return std::max(0.0f, v - val0);
    else
        return v - val0;
}

void FourierSeriesEvaluator::updateCoefficientsStored(const float t)
{
    if(storedTime != t)
    {
        fourierEvaluatorWithoutConstant->valuesAt(t,
                                                  fourierCoefficientsAtStoredTimeWithoutConstant);
        storedTime = t;
    }
}

void FourierSeriesEvaluator::updateFramesStored(const uint16_t z)
{

    if(storedZ != z)
    {
        for(uint32_t i = 1; i < basisSize; i++)
        {
            framesStored[i] = coefficientVolumes[i]->readFrame(z);
        }
        storedZ = z;
    }
}

float FourierSeriesEvaluator::valueAt_intervalStart(const uint16_t x,
                                                    const uint16_t y,
                                                    const uint16_t z)
{

    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z);
    float val = 0.0;
    for(uint32_t i = 1; i < basisSize; i++)
    {
        val += fourierCoefficientsAtIntervalStartWithoutConstant[i - 1]
            * framesStored[i]->get(x, y);
    }
    return val;
}

float FourierSeriesEvaluator::valueAt_withoutOffset(const uint16_t x,
                                                    const uint16_t y,
                                                    const uint16_t z,
                                                    const float t)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateCoefficientsStored(t);
    updateFramesStored(z);
    float val = 0.0;
    for(uint32_t i = 1; i != basisSize; i++)
    {
        val += fourierCoefficientsAtStoredTimeWithoutConstant[i - 1] * framesStored[i]->get(x, y);
    }
    return val;
}

void FourierSeriesEvaluator::frameAt(const uint16_t z, const float t, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z);
    std::fill_n(valuesAtStart, dimx * dimy, float(0.0));
    std::fill_n(val, dimx * dimy, float(0.0));
    // Initialize values of legendre polynomials at the time intervalStart
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 1; d < basisSize; d++)
            {
                valuesAtStart[y * dimx + x]
                    += fourierCoefficientsAtIntervalStartWithoutConstant[d - 1]
                    * framesStored[d]->get(x, y);
            }
        }
    }
    updateCoefficientsStored(t);
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 1; d < basisSize; d++)
            {
                val[y * dimx + x] += fourierCoefficientsAtStoredTimeWithoutConstant[d - 1]
                    * framesStored[d]->get(x, y);
            }
            if(negativeAsZero)
            {
                val[y * dimx + x]
                    = std::max(float(0), val[y * dimx + x] - valuesAtStart[y * dimx + x]);
            } else
            {
                val[y * dimx + x] = val[y * dimx + x] - valuesAtStart[y * dimx + x];
            }
        }
    }
}

void FourierSeriesEvaluator::frameAt_customOffset(const uint16_t z,
                                                  const float t,
                                                  float* offset,
                                                  float* val)
{
    std::fill_n(val, dimx * dimy, float(0.0));
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z);
    updateCoefficientsStored(t);
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 1; d < basisSize; d++)
            {
                val[y * dimx + x] += fourierCoefficientsAtStoredTimeWithoutConstant[d - 1]
                    * framesStored[d]->get(x, y);
            }
            if(negativeAsZero)
            {
                val[y * dimx + x] = std::max(float(0), val[y * dimx + x] - offset[y * dimx + x]);
            } else
            {
                val[y * dimx + x] = val[y * dimx + x] - offset[y * dimx + x];
            }
        }
    }
}

void FourierSeriesEvaluator::frameAt_intervalStart(const uint16_t z, float* val)
{

    std::fill_n(val, dimx * dimy, float(0.0));
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z);

    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 1; d < basisSize; d++)
            {
                val[y * dimx + x] += fourierCoefficientsAtIntervalStartWithoutConstant[d - 1]
                    * framesStored[d]->get(x, y);
            }
        }
    }
}

void FourierSeriesEvaluator::frameTimeSeries(const uint16_t z,
                                             const uint32_t granularity,
                                             float* val)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    frameAt_intervalStart(z, val);
    for(uint32_t i = 1; i < granularity; i++)
    {
        time += increment;
        frameAt_customOffset(z, time, val, &val[i * dimx * dimy]);
    }
    std::fill_n(val, dimx * dimy, float(0.0)); // At time zero is concentration zero
}
} // namespace CTL::util
