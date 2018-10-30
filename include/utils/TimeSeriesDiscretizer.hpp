#pragma once

// External libraries
#include "ctpl_stl.h" //Threadpool

#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "SVD/TikhonovInverse.hpp"

namespace CTL::utils {

class TimeSeriesDiscretizer
{
public:
    TimeSeriesDiscretizer(std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator,
                          std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> coefs,
                          float secLength,
                          int threads)
    {
        this->baseFunctionsEvaluator = baseFunctionsEvaluator;
        this->coefs = coefs;
        this->intervalStart = baseFunctionsEvaluator->getStart();
        this->intervalEnd = baseFunctionsEvaluator->getEnd();
        this->baseSize = baseFunctionsEvaluator->getDimension();
        this->dimx = coefs[0]->dimx();
        this->dimy = coefs[0]->dimy();
        this->dimz = coefs[0]->dimz();
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

    float evaluateFunctionAt(int x, int y, int z, double t)
    {
        if(z != zStored)
        {
            frames.clear();
            for(int i = 0; i != baseSize; i++)
            {
                frames.push_back(coefs[i]->readFrame(z));
            }
            zStored = z;
        }
        float baseval[baseSize];
        baseFunctionsEvaluator->valuesAt(t, baseval);
        float val = 0.0;
        for(int i = 0; i != baseSize; i++)
        {
            val += baseval[i] * frames[i]->get(x, y);
        }
        return val;
    }
    /**Evaluate function on the level of frame and write values into the array.
     *
     * z ... frame coordinate
     */
    void evaluateFunction(int z, double t, float* values)
    {
        float baseval[baseSize];
        std::fill(values, values + (dimx * dimy), float(0));
        baseFunctionsEvaluator->valuesAt(t, baseval);
        std::shared_ptr<io::Frame2DI<float>> f;
        for(int b = 0; b != baseSize; b++)
        {
            f = coefs[b]->readFrame(z);
            for(int x = 0; x != dimx; x++)
            {
                for(int y = 0; y != dimy; y++)
                {
                    values[dimx * y + x] += baseval[b] * f->get(x, y);
                }
            }
        }
    }

    void fillTimeValues(int x, int y, int z, int granularity, float* values)
    {
        double time = intervalStart;
        double increment = (intervalEnd - intervalStart) / double(granularity - 1);
        for(int i = 0; i != granularity; i++)
        {
            values[i] = evaluateFunctionAt(x, y, z, time);
            time += increment;
        }
    }

    void fillTimeFrame(int z, int granularity, float* values)
    {
        double time = intervalStart;
        double increment = (intervalEnd - intervalStart) / double(granularity - 1);
        for(int i = 0; i != granularity; i++)
        {
            evaluateFunction(z, time, &values[i * dimx * dimy]);
            time += increment;
        }
    }

    /**Get peak time in seconds from 0.
     *
     */
    float getPeakTime(int x, int y, int z, int granularity)
    {
        float time = intervalStart;
        float dt = (intervalEnd - intervalStart) / float(granularity - 1);
        float maxval, val, maxtime;
        maxval = std::numeric_limits<float>::min();
        for(int i = 0; i != granularity; i++)
        {
            val = evaluateFunctionAt(x, y, z, time);
            if(val > maxval)
            {
                maxval = val;
                maxtime = time / secLength;
            }
            time += dt;
        }
        return maxtime / secLength;
    }

    /** Write one slice of TTP values.
     *
     *Values are in seconds from 0.
     *
     */
    void writePeakSlice(int z, int granularity, std::shared_ptr<io::AsyncFrame2DWritterI<float>> w)
    {
        float time = intervalStart;
        float dt = (intervalEnd - intervalStart) / float(granularity - 1);
        LOGD << io::xprintf("Writing %d TTP frame to file, dt is %fs.", z, dt / secLength);
        float *maxval, *val;
        maxval = new float[dimx * dimy];
        val = new float[dimx * dimy];
        evaluateFunction(z, time, val);
        std::fill(maxval, &maxval[dimx * dimy], time / secLength); // Misuse of maxval
        io::BufferedFrame2D<float> pt(maxval, dimx, dimy); // Init buffer by time at the begining
        std::memcpy(maxval, val,
                    dimx * dimy * sizeof(float)); // Maximum value is value at the begining
        time += dt;
        for(int i = 1; i < granularity; i++)
        {
            evaluateFunction(z, time, val);
            for(int x = 0; x != dimx; x++)
            {
                for(int y = 0; y != dimy; y++)
                {
                    if(val[x + dimx * y] > maxval[x + dimx * y])
                    {
                        maxval[x + dimx * y] = val[x + dimx * y];
                        pt.set(time / secLength, x, y);
                    }
                }
            }
            time += dt;
        }
        w->writeFrame(pt, z);
        delete[] maxval;
        delete[] val;
    }

    void fillConvolutionMatrix(int x, int y, int z, int granularity, float* A)
    {
        float* aif = new float[granularity];
        fillTimeValues(x, y, z, granularity, aif);
        for(int i = 0; i != granularity; i++)
            for(int j = 0; j != granularity; j++)
            {
                if(j > i)
                {
                    A[i * granularity + j] = 0.0;
                } else
                {
                    A[i * granularity + j] = aif[i - j];
                }
            }

        delete[] aif;
    }

    void computeTTP(int granularity, std::shared_ptr<io::AsyncFrame2DWritterI<float>> w)
    {
        /*        float* b = new float[dimx * dimy];
                io::BufferedFrame2D<float> f(b, dimx, dimy);
                for(int z = 0; z != dimz; z++)
                {
                    for(int x = 0; x != dimx; x++)
                    {
                        for(int y = 0; y != dimy; y++)
                        {
                            float pt = getPeakTime(x, y, z, granularity);
                            f.set(pt, x, y);
                        }
                    }
                    w->writeFrame(f, z);
                }
                delete[] b;
        */
        // New implementation
        LOGD << io::xprintf("Called computeTTP dimz is %d and threads is %d.", dimz, threads);
        ctpl::thread_pool* threadpool = new ctpl::thread_pool(threads);
        for(int z = 0; z != dimz; z++)
        {
            threadpool->push([&, this, z](int id) { writePeakSlice(z, granularity, w); });
            // writePeakSlice(z, granularity, w);//For testing normal
        }
        threadpool->stop(true);
        delete threadpool;
    }

    void multiplyWithVector(float* A, float* v, int n, float* out)
    {
        for(int i = 0; i != n; i++)
        {
            out[i] = 0.0;
            for(int k = 0; k != n; k++)
            {
                out[i] += A[i * n + k] * v[k];
            }
        }
    }

    /**Compute slice of convolved parameters
     *
     */
    void computeConvolved(int z,
                          float* convolutionInverse,
                          int granularity,
                          std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w,
                          std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w,
                          std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w)
    {
        LOGD << io::xprintf("Computation on %d.", z);
        float* values = new float[dimx * dimy * granularity];
        float* convol = new float[dimx * dimy * granularity];
        float* maxval_cbf = new float[dimx * dimy];
        float* sum_cbv = new float[dimx * dimy];
        float* div_mtt = new float[dimx * dimy];
        std::fill(maxval_cbf, &maxval_cbf[dimx * dimy], std::numeric_limits<float>::min());
        std::fill(sum_cbv, &sum_cbv[dimx * dimy], float(0));
        std::fill(convol, &convol[dimx * dimy * granularity], float(0));
        double time = intervalStart;
        double dt = (intervalEnd - intervalStart) / double(granularity - 1);
        for(int i = 0; i != granularity; i++)
        {
            evaluateFunction(z, time, &values[i * dimx * dimy]);
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

    void computeConvolvedParameters(float* convolutionInverse,
                                    int granularity,
                                    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w,
                                    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w,
                                    std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w)
    {
        /*
        float* b = new float[dimx * dimy];
        float* values = new float[granularity];
        float* convol = new float[granularity];
        float dt = (intervalEnd - intervalStart) / float(granularity - 1);
        io::BufferedFrame2D<float> cbf(b, dimx, dimy);
        io::BufferedFrame2D<float> cbv(b, dimx, dimy);
        io::BufferedFrame2D<float> mtt(b, dimx, dimy);
        for(int z = 0; z != dimz; z++)
        {
            for(int x = 0; x != dimx; x++)
            {
                for(int y = 0; y != dimy; y++)
                {
                    fillTimeValues(x, y, z, granularity, values);
                    multiplyWithVector(convolutionInverse, values, granularity, convol);
                    float sum = 0;
                    float maxval = std::numeric_limits<float>::min();
                    for(int i = 0; i != granularity; i++)
                    {

                        maxval = (convol[i] < maxval ? maxval : convol[i]);
                        sum += convol[i];
                    }
                    maxval /= dt;
                    cbf.set(maxval, x, y);
                    cbv.set(sum, x, y);
                    mtt.set(sum / maxval, x, y);
                }
            }
            cbf_w->writeFrame(cbf, z);
            cbv_w->writeFrame(cbv, z);
            mtt_w->writeFrame(mtt, z);
        }
        delete[] b;
        delete[] values;
        delete[] convol;
*/

        ctpl::thread_pool* threadpool = new ctpl::thread_pool(threads);
        for(int z = 0; z != dimz; z++)
        {
            threadpool->push([&, this, z](int id) {
                computeConvolved(z, convolutionInverse, granularity, cbf_w, cbv_w, mtt_w);
            });
            // computeConvolved(z, convolutionInverse, granularity, cbf_w, cbv_w, mtt_w);//For
            // testing normal
        }
        threadpool->stop(true);
        delete threadpool;
    }

private:
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> coefs;
    double intervalStart, intervalEnd;
    int zStored = -1;
    int baseSize;
    int threads;
    int dimx;
    int dimy;
    int dimz;
    float secLength;
    std::vector<std::shared_ptr<io::Frame2DI<float>>> frames;
};

} // namespace CTL::utils
