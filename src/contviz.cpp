// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

#include "ARGPARSE/parseArgs.h"
//#include "strtk/strtk.hpp"
//#include <string>
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "FUN/LegendrePolynomialsDerivatives.hpp"
#include "FUN/LegendrePolynomialsExplicit.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "stringFormatter.h"
#include "utils/TimeSeriesDiscretizer.hpp"
#include <regex>

using namespace CTL;

/// Arguments of the main function.
struct Arguments
{
    /// Function to parse function parameters.
    int parseArguments(int argc, char* argv[]);

    /// Folder to which output data after the linear regression.
    std::string outputFolder;

    /// Projection files in a DEN format to use for linear regression.
    /// Together with that should also come the gradient data.
    std::vector<std::string> fittedCoeficients;

    /// Gradients of the coeficients in x, y and z direction
    std::vector<std::string> gradX, gradY, gradZ;

    /// Number of threads
    int threads = 1;

    /// Controls the size of the time interval [ms] that should be identified with 0.0.
    float startTime = 4117, endTime = 56000;

    /// Granularity of the time is number of time points analyzed by i.e. convolution
    int granularity = 100;

    // Length of one second in the units of the domain
    float secLength = 1000;

    // Frames to process
    std::string frameSpecs = "";
    std::vector<int> frames;
};

int Arguments::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Visualization of perfusion parameters CT based on Legendre fiting." };
    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use. Defaults to 1.")
        ->check(CLI::Range(1, 65535));
    app.add_option("-i,--start-time", startTime,
                   "Start of the interval in miliseconds of the support of the functions of time "
                   "[defaults to 4117, 247*16.6].")
        ->check(CLI::Range(0.0, 100000.0));
    app.add_option("-e,--end-time", endTime,
                   "End of the interval in miliseconds of the support of the functions of time "
                   "[defaults to 56000, duration of 9 sweeps].")
        ->check(CLI::Range(0.0, 100000.0));
    app.add_option("-g,--granularity", granularity,
                   "Granularity of the time is number of time points to which time interval is "
                   "discretized. Defaults to 100.")
        ->check(CLI::Range(1, 1000000));
    app.add_option("-c,--sec-length", secLength,
                   "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));

    app.add_option("output_folder", outputFolder,
                   "Folder to which output data after the processing. Each z component processed "
                   "will be in separate DEN file.")
        ->required()
        ->check(CLI::ExistingDirectory);
    app.add_option("-f,--frames", frames,
                   "Specify only particular slices/frames in a z direction to process. You can "
                   "input range i.e. 0-20 or "
                   "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("fitted_coeficients", fittedCoeficients,
                   "Legendre coeficients fited by the algorithm. Orderred from the zeroth "
                   "coeficient that corresponds to the constant that needs to be offseted to match "
                   "zero concentration at the start of the time. These must be also the "
                   "corresponding files that represents gradients of these coefficients in x, y "
                   "and z direction.")
        ->required()
        ->check(CLI::ExistingFile);

    try
    {
        app.parse(argc, argv);
        io::DenFileInfo di(fittedCoeficients[0]);
        int dimz = di.dimz();
        frames = util::processFramesSpecification(frameSpecs, dimz);
        if(!(startTime < endTime))
        {
            io::throwerr("Start time %f must preceed end time %f.", startTime, endTime);
        }
        if(secLength == 0.0)
        {
            io::throwerr("Length of the second is not positive!");
        }
        std::string output_x, output_y, output_z;
        for(std::size_t i = 0; i != fittedCoeficients.size(); i++)
        {

            std::string prefix = fittedCoeficients[i].substr(0, fittedCoeficients[i].find(".", 0));
            output_x = io::xprintf("%s_x.den", prefix.c_str());
            output_y = io::xprintf("%s_y.den", prefix.c_str());
            output_z = io::xprintf("%s_z.den", prefix.c_str());
            if(!io::fileExists(output_x) || !io::fileExists(output_y) || !io::fileExists(output_z))
            {
                LOGE << io::xprintf(
                    "Some of the gradient files required %s, %s or %s does not exist!",
                    output_x.c_str(), output_y.c_str(), output_z.c_str());
                return 1;
            }
            gradX.push_back(output_x);
            gradY.push_back(output_y);
            gradZ.push_back(output_z);
        }
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        if(exitcode == 0) // Help message was printed
        {
            return 1;
        } else
        {
            LOGE << "Parse error catched";
            return -1;
        }
    }
    return 0;
}

void evaluateFrame(std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> coef,
                   std::shared_ptr<util::VectorFunctionI> b,
                   uint16_t z,
                   float time,
                   float* val)
{
    uint32_t degree = b->getDimension();
    float* coefficientsAtTime = new float[b->getDimension()];
    uint16_t dimx = coef[0]->dimx();
    uint16_t dimy = coef[0]->dimy();
    std::vector<std::shared_ptr<io::Frame2DI<float>>> frames;
    for(std::shared_ptr<io::Frame2DReaderI<float>> x : coef)
    {
        frames.push_back(x->readFrame(z));
    }
    b->valuesAt(time, coefficientsAtTime);
    std::fill_n(val, dimx * dimy, float(0.0));
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != degree; d++)
            {
                val[y * dimx + x] += coefficientsAtTime[d] * frames[d]->get(x, y);
            }
        }
    }
    delete[] coefficientsAtTime;
}

void evaluateFrameShiftedToZero(std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> coef,
                                std::shared_ptr<util::VectorFunctionI> b,
                                uint16_t z,
                                float time,
                                float* val)
{
    uint32_t degree = b->getDimension();
    float* valuesAtTime = new float[b->getDimension()];
    float* valuesAtZero = new float[b->getDimension()];
    uint16_t dimx = coef[0]->dimx();
    uint16_t dimy = coef[0]->dimy();
    std::vector<std::shared_ptr<io::Frame2DI<float>>> frames;
    for(std::shared_ptr<io::Frame2DReaderI<float>> x : coef)
    {
        frames.push_back(x->readFrame(z));
    }
    b->valuesAt(time, valuesAtTime);
    b->valuesAt(b->getStart(), valuesAtZero);
    std::fill_n(val, dimx * dimy, float(0.0));
    for(int y = 0; y != dimy; y++)
    {
        for(int x = 0; x != dimx; x++)
        {
            for(uint32_t d = 0; d != degree; d++)
            {
                // Need to offset the gradients by the value at zero
                val[y * dimx + x] += (valuesAtTime[d] - valuesAtZero[d]) * frames[d]->get(x, y);
            }
        }
    }
    delete[] valuesAtTime;
    delete[] valuesAtZero;
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; // debug, info, ...
    std::string csvLogFile = io::xprintf(
        "/tmp/%s.csv", io::getBasename(std::string(argv[0])).c_str()); // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << io::xprintf("START %s", argv[0]);
    Arguments a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult != 0)
    {
        if(parseResult > 0)
        {
            return 0; // Exited sucesfully, help message printed
        } else
        {
            return -1; // Exited somehow wrong
        }
    }
    int baseSize = a.fittedCoeficients.size();
    io::DenFileInfo di(a.fittedCoeficients[0]);
    uint16_t dimx = di.dimx();
    uint16_t dimy = di.dimy();
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator
        = std::make_shared<util::LegendrePolynomialsExplicit>(baseSize - 1, a.startTime, a.endTime);
    std::shared_ptr<util::VectorFunctionI> baseFunctionsDerivatives
        = std::make_shared<util::LegendrePolynomialsDerivatives>(baseSize - 1, a.startTime,
                                                                 a.endTime);
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> fittedCoeficients;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> fittedCoeficients_x;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> fittedCoeficients_y;
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> fittedCoeficients_z;
    // Fill this array only by values without offset.
    std::shared_ptr<io::Frame2DReaderI<float>> pr, px, py, pz;
    for(int i = 0; i != baseSize; i++)
    {
        pr = std::make_shared<io::DenFrame2DReader<float>>(a.fittedCoeficients[i]);
        px = std::make_shared<io::DenFrame2DReader<float>>(a.gradX[i]);
        py = std::make_shared<io::DenFrame2DReader<float>>(a.gradY[i]);
        pz = std::make_shared<io::DenFrame2DReader<float>>(a.gradZ[i]);
        fittedCoeficients.push_back(pr);
        fittedCoeficients_x.push_back(px);
        fittedCoeficients_y.push_back(py);
        fittedCoeficients_z.push_back(pz);
    }
    /*
        util::TimeSeriesDiscretizer ct(baseFunctionsDerivatives, fittedCoeficients, arg.secLength,
                                        arg.threads);
        util::TimeSeriesDiscretizer cx(baseFunctionsEvaluator, fittedCoeficients_x, arg.secLength,
                                        arg.threads);
        utils::TimeSeriesDiscretizer cy(baseFunctionsEvaluator, fittedCoeficients_y, arg.secLength,
                                        arg.threads);
        utils::TimeSeriesDiscretizer cz(baseFunctionsEvaluator, fittedCoeficients_z, arg.secLength,
                                        arg.threads);
    */

    // Vizualization
    int granularity = a.granularity;
    float* val_t = new float[dimx * dimy * granularity];
    float* val_x = new float[dimx * dimy * granularity];
    float* val_y = new float[dimx * dimy * granularity];
    float* val_z = new float[dimx * dimy * granularity];
    float* val = new float[dimx * dimy * granularity];
    std::unique_ptr<io::FrameMemoryViewer2D<float>> f;
    for(int z : a.frames)
    {
        std::string outputFile = io::xprintf("%s/velocity_%05d.den", a.outputFolder.c_str(), z);
        std::string outputMeanFile
            = io::xprintf("%s/meanvelocity_%05d.den", a.outputFolder.c_str(), z);
        std::unique_ptr<io::AsyncFrame2DWritterI<float>> velocity
            = std::make_unique<io::DenAsyncFrame2DWritter<float>>(outputFile, dimx, dimy,
                                                                  granularity);
        std::unique_ptr<io::AsyncFrame2DWritterI<float>> meanVelocity
            = std::make_unique<io::DenAsyncFrame2DWritter<float>>(outputMeanFile, dimx, dimy, 1);
        double dt = (a.endTime - a.startTime) / double(granularity - 1);
        double time = a.startTime;
        for(int i = 0; i != granularity; i++)
        {
            /// We put frame values with fixed z for time derivatives and for components of
            /// gradients
            evaluateFrame(fittedCoeficients, baseFunctionsDerivatives, z, time,
                          &val_t[i * dimx * dimy]);
            evaluateFrame(fittedCoeficients_x, baseFunctionsEvaluator, z, time,
                          &val_x[i * dimx * dimy]);
            evaluateFrame(fittedCoeficients_y, baseFunctionsEvaluator, z, time,
                          &val_y[i * dimx * dimy]);
            evaluateFrame(fittedCoeficients_z, baseFunctionsEvaluator, z, time,
                          &val_z[i * dimx * dimy]);
            time += dt;
        }
        float gradient = 0.0;
        for(int i = 0; i != dimx * dimy * granularity; i++)
        {
            gradient = std::sqrt(val_x[i] * val_x[i] + val_y[i] * val_y[i] + val_z[i] * val_z[i]);
            val[i] = std::abs(val_t[i]) / gradient;
        }
        for(int i = 0; i != granularity; i++)
        {
            f = std::make_unique<io::FrameMemoryViewer2D<float>>(&val[i * dimx * dimy], dimx, dimy);
            velocity->writeFrame(*f, i);
        }
        // Now construct lower bound on \NORM{\vec{v}}_2
        int framesize = dimx * dimy;
        std::fill(val, val + framesize, float(0));
        for(int i = 0; i != granularity; i++)
        {
            for(int j = 0; j != framesize; j++)
            {
                val[j] += std::abs(val_t[framesize * i + j]);
            }
        }

        std::fill(val_t, val_t + framesize, float(0));
        for(int i = 0; i != granularity; i++)
        {
            for(int j = 0; j != framesize; j++)
            {
                int ind = framesize * i + j;
                val_t[j]
                    += val_x[ind] * val_x[ind] + val_y[ind] * val_y[ind] + val_z[ind] * val_z[ind];
            }
        }
        for(int j = 0; j != framesize; j++)
        {
            val[j] = val[j] / std::sqrt(val_t[j]);
        }

        f = std::make_unique<io::FrameMemoryViewer2D<float>>(val, dimx, dimy);
        meanVelocity->writeFrame(*f, 0);
    }
    delete[] val_t;
    delete[] val_x;
    delete[] val_y;
    delete[] val_z;
    delete[] val;
    LOGI << io::xprintf("END %s", argv[0]);
}
