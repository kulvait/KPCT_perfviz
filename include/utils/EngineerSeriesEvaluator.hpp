#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "FUN/StepFunction.hpp"
#include "utils/Attenuation4DEvaluatorI.hpp"

namespace KCT::util {

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
                            std::vector<std::string>& coefficientVolumeFiles,
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
    void frameTimeSeries(const uint16_t z, const uint32_t granularity, float* val) override;

    /**Function to evaluate the value of Engineer polynomial without constant at given point
     *(x,y,z,intervalStart).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     */
    float valueAt_intervalStart(const uint16_t x, const uint16_t y, const uint16_t z);

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

    /**Function to evaluate the value of Engineer polynomial without constant at given frame
     *(z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[out] val Prealocated array to put the values at particular times.
     */
    void frameAt_intervalStart(const uint16_t z, float* val);

    uint32_t coefficientCount;
    uint16_t samplingPointsCount;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> coefficientVolumes;
    uint16_t dimx, dimy, dimz;
    std::shared_ptr<util::VectorFunctionI> basisEvaluator;
    float* valuesAtStart;
    float* basisCoefficientsStart;

    // Global parameters to support faster operations for certain functions.
    std::mutex globalsAccess;
    float* basisCoefficientsStored;
    float storedTime;
    /** Function updates the basisCoefficientsStored to match given time.
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
                                                 std::vector<std::string>& coefficientVolumeFiles,
                                                 float intervalStart,
                                                 float intervalEnd)
    : Attenuation4DEvaluatorI(intervalStart, intervalEnd)
{
    std::string ERR;
    io::DenFileInfo bfi(sampledBasisFunctions);
    samplingPointsCount = bfi.dimx();
    coefficientCount = coefficientVolumeFiles.size();
    if(coefficientCount < 1)
    {
        io::throwerr("There must be at least one basis function to capture the time behavior but "
                     "there is 0.");
    }
    io::DenFileInfo di(coefficientVolumeFiles[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    if(coefficientCount > bfi.dimz())
    {
        ERR = io::xprintf("Number of files with coefficients have to be at most equal as the "
                          "number of the basis functions encoded in file, but is %d and "
                          "coefficientCount=%d.",
                          coefficientVolumeFiles.size(), coefficientCount);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    std::shared_ptr<io::Frame2DReaderI<float>> pr;
    for(std::size_t i = 0; i != coefficientCount; i++)
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
    basisEvaluator = std::make_shared<util::StepFunction>(sampledBasisFunctions, coefficientCount,
                                                          intervalStart, intervalEnd);
    basisCoefficientsStored = new float[coefficientCount];
    basisCoefficientsStart = new float[coefficientCount];
    basisEvaluator->valuesAt(intervalStart, basisCoefficientsStored);
    basisEvaluator->valuesAt(intervalStart, basisCoefficientsStart);
    storedTime = intervalStart;
    storedZ = 0;
    for(uint32_t i = 0; i != coefficientCount; i++)
    {
        framesStored.push_back(coefficientVolumes[i]->readFrame(storedZ));
    }
    valuesAtStart = new float[dimx * dimy](); // Constructor is not filling this array
}

EngineerSeriesEvaluator::~EngineerSeriesEvaluator()
{
    if(basisCoefficientsStored != nullptr)
    {
        delete[] basisCoefficientsStored;
    }
    if(basisCoefficientsStart != nullptr)
    {
        delete[] basisCoefficientsStart;
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
    return std::max(v - val0, 0.0f);
}

void EngineerSeriesEvaluator::updateEngineerValuesStored(const float t)
{
    if(storedTime != t)
    {
        basisEvaluator->valuesAt(t, basisCoefficientsStored);
        storedTime = t;
    }
}

void EngineerSeriesEvaluator::updateFramesStored(const uint16_t z)
{

    if(storedZ != z)
    {
        framesStored.clear();
        for(uint32_t i = 0; i != coefficientCount; i++)
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
    for(uint32_t i = 0; i != coefficientCount; i++)
    {
        val += basisCoefficientsStart[i] * framesStored[i]->get(x, y);
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
    for(uint32_t i = 0; i != coefficientCount; i++)
    {
        val += basisCoefficientsStored[i] * framesStored[i]->get(x, y);
    }
    return val;
}

void EngineerSeriesEvaluator::frameAt(const uint16_t z, const float t, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z);
    std::fill_n(valuesAtStart, dimx * dimy, float(0.0));
    std::fill_n(val, dimx * dimy, float(0.0));
    // Values at intervalStart
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != coefficientCount; d++)
            {
                valuesAtStart[y * dimx + x]
                    += basisCoefficientsStart[d] * framesStored[d]->get(x, y);
            }
        }
    }
    updateEngineerValuesStored(t);
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != coefficientCount; d++)
            {
                val[y * dimx + x] += basisCoefficientsStored[d] * framesStored[d]->get(x, y);
            }
            val[y * dimx + x] = std::max(float(0), val[y * dimx + x] - valuesAtStart[y * dimx + x]);
        }
    }
}
void EngineerSeriesEvaluator::volumeAt(const float t,
                                       std::shared_ptr<io::AsyncFrame2DWritterI<float>> volume,
                                       const bool subtractZeroVolume)
{
    io::BufferedFrame2D<float> frame(float(0), dimx, dimy);
    io::BufferedFrame2D<float> frame_zero(float(0), dimx, dimy);
    std::shared_ptr<io::Frame2DI<float>> frame_constant;
    float val;
    std::unique_lock<std::mutex> lock(globalsAccess);
    for(uint32_t z = 0; z != dimz; z++)
    {
        updateFramesStored(z);
        if(subtractZeroVolume)
        {
            for(uint32_t y = 0; y != dimy; y++)
            {
                for(uint32_t x = 0; x != dimx; x++)
                {
                    val = 0.0f;
                    for(uint32_t d = 0; d != coefficientCount; d++)
                    {
                        val += basisCoefficientsStart[d] * framesStored[d]->get(x, y);
                    }
                    frame_zero.set(val, x, y);
                }
            }
        }
        updateEngineerValuesStored(t);
        for(uint32_t y = 0; y != dimy; y++)
        {
            for(uint32_t x = 0; x != dimx; x++)
            {
                val = 0.0f;
                for(uint32_t d = 0; d != coefficientCount; d++)
                {
                    val += basisCoefficientsStored[d] * framesStored[d]->get(x, y);
                }
                if(subtractZeroVolume)
                {
                    frame.set(val - frame_zero.get(x, y), x, y);
                } else
                {
                    val += framesStored[0]->get(x, y);
                    frame.set(val, x, y);
                }
            }
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
            for(uint32_t d = 0; d != coefficientCount; d++)
            {
                val[y * dimx + x] += basisCoefficientsStored[d] * framesStored[d]->get(x, y);
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
            for(uint32_t d = 0; d != coefficientCount; d++)
            {
                val[y * dimx + x] += basisCoefficientsStart[d] * framesStored[d]->get(x, y);
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
        time += increment;
        frameAt_customOffset(z, time, val, &val[i * dimx * dimy]);
    }
    std::fill_n(val, dimx * dimy, float(0.0)); // At time zero is concentration zero
}
} // namespace KCT::util
