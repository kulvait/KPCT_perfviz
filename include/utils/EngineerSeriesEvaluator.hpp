#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "utils/Attenuation4DEvaluatorI.hpp"
#include "FUN/StepFunction.hpp"

namespace CTL::util {

class EngineerSeriesEvaluator : public Attenuation4DEvaluatorI
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
    EngineerSeriesEvaluator(std::string sampledBasisFunctions,
                            std::vector<std::string> &coefficientVolumeFiles,
                            float intervalStart,
                            float intervalEnd);

    /**Destructor of EngineerSeriesEvaluator class
     *
     */
    ~EngineerSeriesEvaluator();

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

private:
    /**Function to evaluate the value of Engineer polynomial without constant at given point
     *(x,y,z,t).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     */
    float valueAt_withoutOffset(const uint16_t x, const uint16_t y, const uint16_t z, float t);

    /**Function to evaluate the value of Engineer polynomial without constant at given frame
     *(z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     *@param[in] offset Specified offset to subtract and crop at 0.
     *@param[out] val Prealocated array to put the values at particular times.
     */
    void frameAt_customOffset(const uint16_t z, const float t, float* offset, float* val);

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

    uint32_t degree;
    uint16_t samplingPointsCount;
    float intervalStart;
    float intervalEnd;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> coefficientVolumes;
    uint16_t dimx, dimy, dimz;
    std::shared_ptr<util::VectorFunctionI> basisEvaluator;
    float* valuesAtStart;
    float* legendreValuesIntervalStart;

    // Global parameters to support faster operations for certain functions.
    std::mutex globalsAccess;
    float* legendreValuesStored;
    float storedTime;
    /** Function updates the legendreValuesStored to match given time.
     *
     *@param[in] t Time to update.
     *
     */
    void updateEngineerValuesStored(const float t);
    std::vector<std::shared_ptr<io::Frame2DI<float>>> framesStored;
    uint16_t storedZ;
    /** Function updates the framesStored to match given z frame.
     *
     *@param[in] z Frame to update.
     *
     */
    void updateFramesStored(const uint16_t z);
};

    EngineerSeriesEvaluator::EngineerSeriesEvaluator(std::string sampledBasisFunctions,
                            std::vector<std::string> &coefficientVolumeFiles,
                            float intervalStart,
                            float intervalEnd)
    : intervalStart(intervalStart)
    , intervalEnd(intervalEnd)
{
    io::DenFileInfo bfi(sampledBasisFunctions);
    samplingPointsCount = bfi.dimx();
    degree = coefficientVolumeFiles.size();
    if(degree < 1)
    {
        io::throwerr("There must be at least one basis function to capture the time behavior but there is 0.");
    }
    io::DenFileInfo di(coefficientVolumeFiles[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    if(coefficientVolumeFiles.size() != bfi.dimz())
    {
        io::throwerr("Number of files with coefficients have to be equal to basis functions, "
                     "but is %d and degree=%d.",
                     coefficientVolumeFiles.size(), degree);
    }
    std::shared_ptr<io::Frame2DReaderI<float>> pr;
    for(std::size_t i = 0; i != degree; i++)
    {
        pr = std::make_shared<io::DenFrame2DReader<float>>(coefficientVolumeFiles[i]);
        coefficientVolumes.push_back(pr);
    }
    for(std::shared_ptr<io::Frame2DReaderI<float>>& f : coefficientVolumes)
    {
        if(f->dimx() != dimx || f->dimy() != dimy || f->dimz() != dimz)
        {
            io::throwerr("There are incompatible coefficients!");
        }
    }
    basisEvaluator = std::make_shared<util::StepFunction>(degree, sampledBasisFunctions, intervalStart,
                                               intervalEnd);
    legendreValuesStored = new float[degree];
    legendreValuesIntervalStart = new float[degree];
    basisEvaluator->valuesAt(intervalStart, legendreValuesStored);
    basisEvaluator->valuesAt(intervalStart, legendreValuesIntervalStart);
    storedTime = intervalStart;
    storedZ = 0;
    for(uint32_t i = 0; i != degree; i++)
    {
        framesStored.push_back(coefficientVolumes[i]->readFrame(storedZ));
    }
    valuesAtStart = new float[dimx * dimy](); // Constructor is not filling this array
}

EngineerSeriesEvaluator::~EngineerSeriesEvaluator()
{
    if(legendreValuesStored != nullptr)
    {
        delete[] legendreValuesStored;
    }
    if(legendreValuesIntervalStart != nullptr)
    {
        delete[] legendreValuesIntervalStart;
    }
    if(valuesAtStart != nullptr)
    {
        delete[] valuesAtStart;
    }
}

void EngineerSeriesEvaluator::timeDiscretization(const uint32_t granularity, float* timePoints)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    for(uint32_t i = 0; i != granularity; i++)
    {
        timePoints[i] = float(time);
        time += increment;
    }
}

bool EngineerSeriesEvaluator::isTimeDiscretizedEvenly() { return true; }

void EngineerSeriesEvaluator::timeSeriesIn(
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

float EngineerSeriesEvaluator::valueAt(const uint16_t x,
                                       const uint16_t y,
                                       const uint16_t z,
                                       const float t)
{
    float val0 = valueAt_intervalStart(x, y, z);
    float v = valueAt_withoutOffset(x, y, z, t);
    return std::max(float(0), v - val0);
}

void EngineerSeriesEvaluator::updateEngineerValuesStored(const float t)
{
    if(storedTime != t)
    {
        basisEvaluator->valuesAt(t, legendreValuesStored);
        storedTime = t;
    }
}

void EngineerSeriesEvaluator::updateFramesStored(const uint16_t z)
{

    if(storedZ != z)
    {
        framesStored.clear();
        for(uint32_t i = 0; i != degree; i++)
        {
            framesStored.push_back(coefficientVolumes[i]->readFrame(z));
        }
        storedZ = z;
    }
}

float EngineerSeriesEvaluator::valueAt_intervalStart(const uint16_t x,
                                                     const uint16_t y,
                                                     const uint16_t z)
{

    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z);
    float val = 0.0;
    for(uint32_t i = 0; i != degree; i++)
    {
        val += legendreValuesIntervalStart[i] * framesStored[i]->get(x, y);
    }
    return val;
}

float EngineerSeriesEvaluator::valueAt_withoutOffset(const uint16_t x,
                                                     const uint16_t y,
                                                     const uint16_t z,
                                                     const float t)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateEngineerValuesStored(t);
    updateFramesStored(z);
    float val = 0.0;
    for(uint32_t i = 0; i != degree; i++)
    {
        val += legendreValuesStored[i] * framesStored[i]->get(x, y);
    }
    return val;
}

void EngineerSeriesEvaluator::frameAt(const uint16_t z, const float t, float* val)
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
            for(uint32_t d = 0; d != degree; d++)
            {
                valuesAtStart[y * dimx + x]
                    += legendreValuesIntervalStart[d] * framesStored[d]->get(x, y);
            }
        }
    }
    updateEngineerValuesStored(t);
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != degree; d++)
            {
                val[y * dimx + x] += legendreValuesStored[d] * framesStored[d]->get(x, y);
            }
            val[y * dimx + x] = std::max(float(0), val[y * dimx + x] - valuesAtStart[y * dimx + x]);
        }
    }
}

void EngineerSeriesEvaluator::frameAt_customOffset(const uint16_t z,
                                                   const float t,
                                                   float* offset,
                                                   float* val)
{
    std::fill_n(val, dimx * dimy, float(0.0));
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z);
    updateEngineerValuesStored(t);
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != degree; d++)
            {
                val[y * dimx + x] += legendreValuesStored[d] * framesStored[d]->get(x, y);
            }
            val[y * dimx + x] = std::max(float(0), val[y * dimx + x] - offset[y * dimx + x]);
        }
    }
}
void EngineerSeriesEvaluator::frameAt_intervalStart(const uint16_t z, float* val)
{

    std::fill_n(val, dimx * dimy, float(0.0));
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z);

    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != degree; d++)
            {
                val[y * dimx + x] += legendreValuesIntervalStart[d] * framesStored[d]->get(x, y);
            }
        }
    }
}
void EngineerSeriesEvaluator::frameTimeSeries(const uint16_t z,
                                              const uint32_t granularity,
                                              float* val)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    frameAt_intervalStart(z, val);
    for(uint32_t i = 1; i < granularity; i++)
    {
        frameAt_customOffset(z, time, val, &val[i * dimx * dimy]);
        time += increment;
    }
    std::fill_n(val, dimx * dimy, float(0.0)); // At time zero is concentration zero
}
} // namespace CTL::util
