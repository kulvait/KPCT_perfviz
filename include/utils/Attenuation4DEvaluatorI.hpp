#pragma once

namespace KCT::util {

class Attenuation4DEvaluatorI
{
public:
    Attenuation4DEvaluatorI(float intervalStart, float intervalEnd)
        : intervalStart(intervalStart)
        , intervalEnd(intervalEnd)
    {
        if(!(intervalStart < intervalEnd))
        {
            std::string msg = io::xprintf(
                "Start of interval %f needs to be strictly before its end %f but its not!",
                intervalStart, intervalEnd);
            LOGE << msg;
            throw std::runtime_error(msg);
        }
    }

    /** Virtual destructor
     *
     * Virtual destructor is important not to produce warnings, see
     * https://stackoverflow.com/a/10024842
     */
    virtual ~Attenuation4DEvaluatorI() = default;

    /**Function to obtain time discretization.
     *
     *This function evaluates the time instants in which other functions of the implementing class
     *fills the arrays with values in that times.
     *@param[in] granularity Number of time points to fill in timePoints array with.
     *@param[out] timePoints Time discretization.
     */
    virtual void timeDiscretization(const uint32_t granularity, float* timePoints) = 0;

    /**Test if the time is discretized evenly.
     *
     * This property is required for assembling pseudoinverse matrix.
     */
    virtual bool isTimeDiscretizedEvenly() = 0;

    /**Function to evaluate time series in given point.
     *
     *@param[in] vx Zero based x coordinate of the volume.
     *@param[in] vy Zero based y coordinate of the volume.
     *@param[in] vz Zero based z coordinate of the volume.
     *@param[in] granularity Number of time points to fill in aif array with.
     *@param[out] aif Prealocated array to put the values at particular times.
     */
    virtual void timeSeriesIn(const uint32_t vx,
                              const uint32_t vy,
                              const uint32_t vz,
                              const uint32_t granularity,
                              float* aif)
        = 0;

    /**Function to evaluate the value of Legendre polynomial without constant at given point
     *(x,y,z,t).
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     *@param[out] aif Prealocated array to put the values at particular times.
     */
    virtual float valueAt(const uint32_t x, const uint32_t y, const uint32_t z, const float t) = 0;

    /**Function to evaluate the value of attenuation for the whole frame (z,t).
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] t Time of evaluation.
     *@param[out] val Prealocated array of size dimx*dimy to put the values of the frame at time t
     *at frame z.
     */
    virtual void frameAt(const uint32_t z, const float t, float* val) = 0;

    /**Function to evaluate the value of attenuation for the whole volume at point t.
     *
     * @param[in] t Time of evaluation.
     * @param[out] volume Writter to write volume to
     * @param[in] subtractZeroVolume If the values at zero shall be zeros and all other volumes
     * shall be evaluated with respect to this. Good for representing TACs.
     */
    virtual void volumeAt(const float t,
                          std::shared_ptr<io::AsyncFrame2DWritterI<float>> volume,
                          const bool subtractZeroVolume = true)
        = 0;

    /**Function to evaluate time series of frames of attenuation coefficients.
     *
     *@param[in] vz Zero based z coordinate of the volume.
     *@param[in] granularity Number of time points to fill in aif array with.
     *@param[out] val Prealocated array to put the values at particular times of the size
     *granularity*dimx*dimy.
     */
    virtual void frameTimeSeries(const uint32_t vz, const uint32_t granularity, float* val) = 0;

protected:
    float intervalStart;
    float intervalEnd;
};
} // namespace KCT::util
