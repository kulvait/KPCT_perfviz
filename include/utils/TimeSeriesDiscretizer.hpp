#pragma once

// External libraries
#include "ctpl_stl.h" //Threadpool

#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "utils/Attenuation4DEvaluatorI.hpp"

namespace CTL::util {

class TimeSeriesDiscretizer
{
public:
    TimeSeriesDiscretizer(std::shared_ptr<util::Attenuation4DEvaluatorI> attenuationEvaluator,
                          uint16_t dimx,
                          uint16_t dimy,
                          uint16_t dimz,
                          float intervalStart,
                          float intervalEnd,
                          float secLength,
                          uint16_t threads)
        : attenuationEvaluator(attenuationEvaluator)
        , dimx(dimx)
        , dimy(dimy)
        , dimz(dimz)
        , intervalStart(intervalStart)
        , intervalEnd(intervalEnd)
        , secLength(secLength)
        , threads(threads)
    {
        if(threads < 1)
        {
            io::throwerr("The number of threads %d must be positive", threads);
        }
        this->threads = threads;
        if(secLength > 0.0)
        {
            this->secLength = secLength;
        } else
        {
            io::throwerr("Length of the second %f needs to be positive!", secLength);
        }
    }

    /** Computes the second from zero in which attenuation is maximal in given point.
     *
     *@param[in] x Zero based x coordinate of the volume.
     *@param[in] y Zero based y coordinate of the volume.
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] granularity How much time instants should be used to discretize time interval.
     */
    float getPeakIn(int x, int y, int z, int granularity)
    {
        float time = intervalStart;
        float dt = (intervalEnd - intervalStart) / float(granularity - 1);
        float maxval, val, maxtime;
        maxval = std::numeric_limits<float>::min();
        for(int i = 0; i != granularity; i++)
        {
            val = attenuationEvaluator->valueAt(x, y, z, time);
            if(val > maxval)
            {
                maxval = val;
                maxtime = time / secLength;
            }
            time += dt;
        }
        return maxtime / secLength;
    }

    /** Writes the frame z of the second from zero in which attenuation is maximal in given point.
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] granularity How much time instants should be used to discretize time interval.
     */
    void writePeakFrame(int z, int granularity, std::shared_ptr<io::AsyncFrame2DWritterI<float>> w)
    {
        float time = intervalStart;
        float dt = (intervalEnd - intervalStart) / float(granularity - 1);
        float *maxval, *val;
        maxval = new float[dimx * dimy];
        val = new float[dimx * dimy];
        attenuationEvaluator->frameAt(z, intervalStart, maxval);
        io::BufferedFrame2D<float> pt(float(intervalStart / secLength), dimx,
                                      dimy); // Init buffer by time at the begining
        time += dt;
        for(int i = 1; i < granularity; i++)
        {
            attenuationEvaluator->frameAt(z, time, val);
            for(int x = 0; x != dimx; x++)
            {
                for(int y = 0; y != dimy; y++)
                {
                    if(val[x + dimx * y] > maxval[x + dimx * y])
                    {
                        maxval[x + dimx * y] = val[x + dimx * y];
                        pt.set(float(time / secLength), x, y);
                    }
                }
            }
            time += dt;
        }
        w->writeFrame(pt, z);
        delete[] maxval;
        delete[] val;
        LOGD << io::xprintf("Written %d TTP frame.", z);
    }

    /** Computes the second from zero in which attenuation is maximal.
     *
     *@param[in] granularity How much time instants should be used to discretize time interval.
     *@param[in] w The result should be written into the volume through this frame writter
     *interface.
     */
    void computeTTP(int granularity, std::shared_ptr<io::AsyncFrame2DWritterI<float>> w)
    {
        if(threads == 0)
        {
            LOGD << io::xprintf("Function computeTTP working synchronously without threading.");
            for(int z = 0; z != dimz; z++)
            {
                writePeakFrame(z, granularity, w); // For testing normal
            }
        } else
        {
            LOGD << io::xprintf("Function computeTTP working asynchronously on %d threads.",
                                threads);
            ctpl::thread_pool* threadpool = new ctpl::thread_pool(threads);
            for(int z = 0; z != dimz; z++)
            {
                threadpool->push(
                    [&, this, z, granularity, w](int id) { writePeakFrame(z, granularity, w); });
            }
            threadpool->stop(true);
            delete threadpool;
        }
    }

    /**Writes the z frames of perfusion parameters integral of deconvolution vector elements,
     *maximal element of deconvolution vector and their division into the files.
     *
     *@param[in] z Zero based z coordinate of the volume.
     *@param[in] granularity How much time instants should be used to discretize time interval.
     *@param[in] convolutionInverse Inverse convolution matrix of the size granularity \times
     *granularity.
     *@param[in] cbf_w Writter to write maximal element of deconvolution vector in.
     *@param[in] cbv_w Writter to write deconvolution vector integral .
     *@param[in] mtt_w Writter to write division product of deconvolution integral and deconvolution
     *maximum.
     */
    void writePerfusionFrames(int z,
                              int granularity,
                              float* convolutionInverse,
                              std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w,
                              std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w,
                              std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w)
    {
        LOGD << io::xprintf("Estimating perfusion parameters for frame %d.", z);
        float* values = new float[dimx * dimy * granularity];
        float* convol = new float[dimx * dimy * granularity]();
        float* maxval_cbf = new float[dimx * dimy];
        float* sum_cbv = new float[dimx * dimy]();
        float* div_mtt = new float[dimx * dimy];
        std::fill(maxval_cbf, &maxval_cbf[dimx * dimy], std::numeric_limits<float>::min());
        double time = intervalStart;
        double dt = (intervalEnd - intervalStart) / double(granularity - 1);
        for(int i = 0; i != granularity; i++)
        {
            attenuationEvaluator->frameAt(z, time, &values[i * dimx * dimy]);
            time += dt;
        }
        for(int x = 0; x != dimx; x++)
        {
            for(int y = 0; y != dimy; y++)
            {
                for(int i = 0; i != granularity; i++)
                {
                    for(int j = 0; j != granularity; j++)
                    {
                        convol[i * dimx * dimy + y * dimx + x]
                            += convolutionInverse[i * granularity + j]
                            * values[j * dimx * dimy + y * dimx + x];
                    }
                    if(convol[i * dimx * dimy + y * dimx + x] / dt > maxval_cbf[x + dimx * y])
                    {
                        maxval_cbf[x + dimx * y] = convol[i * dimx * dimy + y * dimx + x] / dt;
                    }
                    sum_cbv[x + dimx * y] += convol[i * dimx * dimy + y * dimx + x];
                }
                div_mtt[x + dimx * y] = sum_cbv[x + dimx * y] / maxval_cbf[x + dimx * y];
                div_mtt[x + dimx * y] /= secLength; // Scaling to seconds
                sum_cbv[x + dimx * y] *= 100; // Scaling to mL/100g
                maxval_cbf[x + dimx * y] *= secLength;
                maxval_cbf[x + dimx * y] *= 60;
                maxval_cbf[x + dimx * y] *= 100; // Scaling to mL/100g/min
            }
        }
        io::BufferedFrame2D<float> cbf(maxval_cbf, dimx, dimy);
        cbf_w->writeFrame(cbf, z);
        io::BufferedFrame2D<float> cbv(sum_cbv, dimx, dimy);
        cbv_w->writeFrame(cbv, z);
        io::BufferedFrame2D<float> mtt(div_mtt, dimx, dimy);
        mtt_w->writeFrame(mtt, z);
        delete[] maxval_cbf;
        delete[] sum_cbv;
        delete[] div_mtt;
        delete[] values;
        delete[] convol;
    }

    /**Writes three perfusion parameters into the files.
     *
     *@param[in] granularity How much time instants should be used to discretize time interval.
     *@param[in] convolutionInverse Inverse convolution matrix of the size granularity \times
     *granularity.
     *@param[in] cbf_w Writter to write maximal element of deconvolution vector in.
     *@param[in] cbv_w Writter to write deconvolution vector integral .
     *@param[in] mtt_w Writter to write division product of deconvolution integral and deconvolution
     *maximum.
     */
    void computePerfusionParameters(int granularity,
                                    float* convolutionInverse,
                                    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w,
                                    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w,
                                    std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w)
    {

        if(threads == 0)
        {
            LOGD << io::xprintf(
                "Function computePerfusionParameters working synchronously without threading.");
            for(int z = 0; z != dimz; z++)
            {
                writePerfusionFrames(z, granularity, convolutionInverse, cbf_w, cbv_w, mtt_w);
            }
        } else
        {
            LOGD << io::xprintf(
                "Function computePerfusionParameters working asynchronously on %d threads.",
                threads);
            ctpl::thread_pool* threadpool = new ctpl::thread_pool(threads);
            for(int z = 0; z != dimz; z++)
            {
                threadpool->push([&, this, z, convolutionInverse, granularity, cbf_w, cbv_w,
                                  mtt_w](int id) {
                    writePerfusionFrames(z, granularity, convolutionInverse, cbf_w, cbv_w, mtt_w);
                });
            }
            threadpool->stop(true);
            delete threadpool;
        }
    }

private:
    std::shared_ptr<util::Attenuation4DEvaluatorI> attenuationEvaluator;
    uint16_t dimx, dimy, dimz;
    float intervalStart;
    float intervalEnd;
    float secLength;
    uint16_t threads;
};

} // namespace CTL::util
