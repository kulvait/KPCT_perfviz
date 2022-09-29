#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "FUN/ChebyshevPolynomialsExplicit.hpp"
#include "FUN/LegendrePolynomialsExplicit.hpp"
#include "PROG/KCTException.hpp"
#include "utils/Attenuation4DEvaluatorI.hpp"

namespace KCT::util {

enum polynomialType { Legendre = 0, Chebyshev = 1 };

class PolynomialSeriesEvaluator : public Attenuation4DEvaluatorI
{
public:
    /**Evaluation of the attenuation values based on the Legendre coeficients fiting.
     *
     *@param[in] degree Polynomial degree, should be equal to coefficientVolumes.size() - 1 since
     *the first coeficient represent constant, that is polynomial of zeroth order.
     *@param[in] coefficientVolumeFiles Files with fitted coefficient volumes of related Legendre
     *polynomials.
     *@param[in] intervalStart Start time.
     *@param[in] intervalEnd End time.
     */
    PolynomialSeriesEvaluator(uint32_t degree,
                              std::vector<std::string> coefficientVolumeFiles,
                              float intervalStart,
                              float intervalEnd,
                              bool negativeAsZero = true,
                              polynomialType pt = polynomialType::Legendre);

    /**Destructor of PolynomialSeriesEvaluator class
     *
     */
    ~PolynomialSeriesEvaluator();

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
    void timeSeriesIn(const uint32_t x,
                      const uint32_t y,
                      const uint32_t z,
                      const uint32_t granularity,
                      float* aif) override;

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

    /**Function to evaluate the value of polynomial without constant at given point
     *(x,y,z,intervalStart).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     */
    float valueAt_intervalStart(const uint32_t x, const uint32_t y, const uint32_t z);

private:
    /**Function to evaluate the value of polynomial without constant at given point
     *(x,y,z,t).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     */
    float valueAt_withoutOffset(const uint32_t x, const uint32_t y, const uint32_t z, float t);

    /**Function to evaluate the value of polynomial without constant at given frame
     *(z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     *@param[in] offset Specified offset to subtract and crop at 0.
     *@param[out] val Prealocated array to put the values at particular times.
     */
    void frameAt_customOffset(const uint32_t z, const float t, float* offset, float* val);

    /**Function to evaluate the value of  polynomial without constant at given frame
     *(z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[out] val Prealocated array to put the values at particular times.
     */
    void frameAt_intervalStart(const uint32_t z, float* val);

    uint32_t degree;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> coefficientVolumes;
    std::shared_ptr<io::Frame2DReaderI<float>> constantCoefficientVolume;
    uint32_t dimx, dimy, dimz;
    std::shared_ptr<util::VectorFunctionI> polynomialBasisEvaluator;
    float* valuesAtStart;
    float* polynomialValuesAtIntervalStart;

    // Global parameters to support faster operations for certain functions.
    std::mutex globalsAccess;
    float* polynomialValuesAtStoredTime;
    float storedTime;
    /** Function updates the polynomialValuesAtStoredTime to match given time.
     *
     *@param[in] t Time to update.
     *
     */
    void updatePolynomialValuesStoredToNewTimepoint(const float t);
    std::vector<std::shared_ptr<io::Frame2DI<float>>> framesStored;
    std::shared_ptr<io::Frame2DI<float>> frameZeroStored = nullptr;
    uint32_t storedZ;
    /** Function updates the framesStored to match given z frame.
     *
     *@param[in] z Frame to update.
     *
     */
    void updateFramesStored(const uint32_t z, const bool updateFrameZero);
    bool negativeAsZero;
    polynomialType pt;
};

PolynomialSeriesEvaluator::PolynomialSeriesEvaluator(
    uint32_t degree,
    std::vector<std::string> coefficientVolumeFiles,
    float intervalStart,
    float intervalEnd,
    bool negativeAsZero,
    polynomialType pt)
    : Attenuation4DEvaluatorI(intervalStart, intervalEnd)
    , degree(degree)
    , negativeAsZero(negativeAsZero)
    , pt(pt)
{
    std::string err;
    if(coefficientVolumeFiles.size() != degree + 1)
    {
        err = io::xprintf(
            "Number of files with polynomial coefficients have to be equal to degree + 1, "
            "but is %d and degree=%d.",
            coefficientVolumeFiles.size(), degree);
        KCTERR(err);
    }
    if(degree < 1)
    {
        err = io::xprintf(
            "There must be at least linear polynomial to capture time behavior but we "
            "have polynomial degree 0");
        KCTERR(err);
    }
    std::shared_ptr<io::Frame2DReaderI<float>> pr;
    for(std::size_t i = 0; i != degree; i++)
    {
        pr = std::make_shared<io::DenFrame2DReader<float>>(coefficientVolumeFiles[i + 1]);
        coefficientVolumes.push_back(pr);
    }
    constantCoefficientVolume
        = std::make_shared<io::DenFrame2DReader<float>>(coefficientVolumeFiles[0]);
    io::DenFileInfo di(coefficientVolumeFiles[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();

    for(std::shared_ptr<io::Frame2DReaderI<float>>& f : coefficientVolumes)
    {
        if(f->dimx() != dimx || f->dimy() != dimy || f->dimz() != dimz)
        {
            err = io::xprintf("There are incompatible coefficients!");
            KCTERR(err);
        }
    }
    if(this->pt == polynomialType::Legendre)
    {
        polynomialBasisEvaluator = std::make_shared<util::LegendrePolynomialsExplicit>(
            degree, intervalStart, intervalEnd, true, 1);
    } else
    {
        polynomialBasisEvaluator = std::make_shared<util::ChebyshevPolynomialsExplicit>(
            degree, intervalStart, intervalEnd, true, 1);
    }
    polynomialValuesAtStoredTime = new float[degree];
    polynomialValuesAtIntervalStart = new float[degree];
    polynomialBasisEvaluator->valuesAt(intervalStart, polynomialValuesAtStoredTime);
    polynomialBasisEvaluator->valuesAt(intervalStart, polynomialValuesAtIntervalStart);
    storedTime = intervalStart;
    storedZ = 0;
    for(uint32_t i = 0; i != degree; i++)
    {
        framesStored.push_back(coefficientVolumes[i]->readFrame(storedZ));
    }
    frameZeroStored = constantCoefficientVolume->readFrame(storedZ);
    valuesAtStart = new float[dimx * dimy](); // Constructor is not filling this array
}

PolynomialSeriesEvaluator::~PolynomialSeriesEvaluator()
{
    if(polynomialValuesAtStoredTime != nullptr)
    {
        delete[] polynomialValuesAtStoredTime;
    }
    if(polynomialValuesAtIntervalStart != nullptr)
    {
        delete[] polynomialValuesAtIntervalStart;
    }
    if(valuesAtStart != nullptr)
    {
        delete[] valuesAtStart;
    }
}

void PolynomialSeriesEvaluator::timeDiscretization(const uint32_t granularity, float* timePoints)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    for(uint32_t i = 0; i != granularity; i++)
    {
        timePoints[i] = float(time);
        time += increment;
    }
}

bool PolynomialSeriesEvaluator::isTimeDiscretizedEvenly() { return true; }

void PolynomialSeriesEvaluator::timeSeriesIn(
    const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t granularity, float* val)
{
    double time = intervalStart;
    double increment = (intervalEnd - intervalStart) / double(granularity - 1);
    for(uint32_t i = 0; i != granularity; i++)
    {
        val[i] = valueAt(x, y, z, time);
        time += increment;
    }
}

float PolynomialSeriesEvaluator::valueAt(const uint32_t x,
                                         const uint32_t y,
                                         const uint32_t z,
                                         const float t)
{
    float val0 = valueAt_intervalStart(x, y, z);
    float v = valueAt_withoutOffset(x, y, z, t);
    if(negativeAsZero)
        return std::max(float(0), v - val0);
    else
        return v - val0;
}

void PolynomialSeriesEvaluator::updatePolynomialValuesStoredToNewTimepoint(const float t)
{
    if(storedTime != t)
    {
        polynomialBasisEvaluator->valuesAt(t, polynomialValuesAtStoredTime);
        storedTime = t;
    }
}

void PolynomialSeriesEvaluator::updateFramesStored(const uint32_t z, const bool updateFrameZero)
{

    if(storedZ != z)
    {
        framesStored.clear();
        for(uint32_t i = 0; i != degree; i++)
        {
            framesStored.push_back(coefficientVolumes[i]->readFrame(z));
        }
        if(updateFrameZero)
        {
            frameZeroStored = constantCoefficientVolume->readFrame(storedZ);
        }
        storedZ = z;
    }
}

float PolynomialSeriesEvaluator::valueAt_intervalStart(const uint32_t x,
                                                       const uint32_t y,
                                                       const uint32_t z)
{

    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z, false);
    float val = 0.0;
    for(uint32_t i = 0; i != degree; i++)
    {
        val += polynomialValuesAtIntervalStart[i] * framesStored[i]->get(x, y);
    }
    return val;
}

float PolynomialSeriesEvaluator::valueAt_withoutOffset(const uint32_t x,
                                                       const uint32_t y,
                                                       const uint32_t z,
                                                       const float t)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updatePolynomialValuesStoredToNewTimepoint(t);
    updateFramesStored(z, false);
    float val = 0.0;
    for(uint32_t i = 0; i != degree; i++)
    {
        val += polynomialValuesAtStoredTime[i] * framesStored[i]->get(x, y);
    }
    return val;
}

void PolynomialSeriesEvaluator::frameAt(const uint32_t z, const float t, float* val)
{
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z, false);
    std::fill_n(valuesAtStart, dimx * dimy, float(0.0));
    std::fill_n(val, dimx * dimy, float(0.0));
    // Initialize values of polynomials at the time intervalStart
    for(uint32_t y = 0; y != dimy; y++)
    {
        for(uint32_t x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != degree; d++)
            {
                valuesAtStart[y * dimx + x]
                    += polynomialValuesAtIntervalStart[d] * framesStored[d]->get(x, y);
            }
        }
    }
    updatePolynomialValuesStoredToNewTimepoint(t);
    for(uint32_t y = 0; y != dimy; y++)
    {
        for(uint32_t x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != degree; d++)
            {
                val[y * dimx + x] += polynomialValuesAtStoredTime[d] * framesStored[d]->get(x, y);
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

void PolynomialSeriesEvaluator::frameAt_customOffset(const uint32_t z,
                                                     const float t,
                                                     float* offset,
                                                     float* val)
{
    std::fill_n(val, dimx * dimy, float(0.0));
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z, false);
    updatePolynomialValuesStoredToNewTimepoint(t);
    for(uint32_t y = 0; y != dimy; y++)
    {
        for(uint32_t x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != degree; d++)
            {
                val[y * dimx + x] += polynomialValuesAtStoredTime[d] * framesStored[d]->get(x, y);
            }
            if(negativeAsZero)
            {
                val[y * dimx + x] = std::max(0.0f, val[y * dimx + x] - offset[y * dimx + x]);
            } else
            {
                val[y * dimx + x] = val[y * dimx + x] - offset[y * dimx + x];
            }
        }
    }
}

void PolynomialSeriesEvaluator::frameAt_intervalStart(const uint32_t z, float* val)
{

    std::fill_n(val, dimx * dimy, float(0.0));
    std::unique_lock<std::mutex> lock(globalsAccess);
    updateFramesStored(z, false);

    for(uint32_t y = 0; y != dimy; y++)
    {
        for(uint32_t x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != degree; d++)
            {
                val[y * dimx + x]
                    += polynomialValuesAtIntervalStart[d] * framesStored[d]->get(x, y);
            }
        }
    }
}

void PolynomialSeriesEvaluator::volumeAt(const float t,
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
        if(subtractZeroVolume)
        {
            updateFramesStored(z, false);
            for(uint32_t y = 0; y != dimy; y++)
            {
                for(uint32_t x = 0; x != dimx; x++)
                {
                    val = 0.0f;
                    for(uint32_t d = 0; d != degree; d++)
                    {
                        val += polynomialValuesAtIntervalStart[d] * framesStored[d]->get(x, y);
                    }
                    frame_zero.set(val, x, y);
                }
            }
        } else
        {
            updateFramesStored(z, true);
        }
        updatePolynomialValuesStoredToNewTimepoint(t);
        for(uint32_t y = 0; y != dimy; y++)
        {
            for(uint32_t x = 0; x != dimx; x++)
            {
                val = 0.0f;
                for(uint32_t d = 0; d != degree; d++)
                {
                    val += polynomialValuesAtStoredTime[d] * framesStored[d]->get(x, y);
                }
                if(subtractZeroVolume)
                {
                    if(negativeAsZero)
                    {
                        frame.set(std::max(0.0f, val - frame_zero.get(x, y)), x, y);
                    } else
                    {
                        frame.set(val - frame_zero.get(x, y), x, y);
                    }
                } else
                {
                    val += frameZeroStored->get(x, y);
                    frame.set(val, x, y);
                }
            }
        }
        volume->writeFrame(frame, z);
    }
}

void PolynomialSeriesEvaluator::frameTimeSeries(const uint32_t z,
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
    std::fill_n(val, dimx * dimy, 0.0f); // At time zero is concentration zero
}
} // namespace KCT::util
