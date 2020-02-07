// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "FUN/FourierSeries.hpp"
#include "Frame2DReaderI.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "stringFormatter.h"
#include "utils/Attenuation4DEvaluatorI.hpp"
#include "utils/FourierSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

#ifdef DEBUG
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
#endif

using namespace CTL;

/// Arguments of the main function.
struct Arguments
{
    /// Function to parse function parameters.
    int parseArguments(int argc, char* argv[]);

    /// Folder to which output data after the linear regression.
    std::string outputFile;

    /// Reconstructed files in a DEN format that represents Fourier coefficients
    std::vector<std::string> fittedCoefficients;

    /// Number of threads
    uint16_t threads = 0;

    /**
     * The first sweep and the last sweep should be identified with the ends of the interval.
     * startTime default is the end of the first sweep and endTime is the start of the last sweep
     * Data are from the experiments. Controls interval [ms].
     */
    float startTime = 4145, endTime = 43699;

    /// Coordinates of arthery input function
    uint16_t ifx, ify, ifz;

    /// Granularity of the time is number of time points analyzed by i.e. convolution
    uint32_t granularity = 20;

    // Length of one second in the units of the domain
    float secLength = 1000;

    bool allowNegativeValues = false;

    // Tikhonov regularization parameter
    float lambdaRel = 0.2;

    // If only ttp should be computed
    bool onlyttp = false;

    /**Vizualize base functions.
     *
     *If set vizualize base functions using Python.
     */
    bool vizualize = false;
    bool onlyaif = false;
    /**
     * @brief File to store AIF.
     */
    std::string storeAIF = "";
    bool halfPeriodicFunctions = false;
};

int Arguments::parseArguments(int argc, char* argv[])
{
    CLI::App app{
        "Fast deconvolution computation of the parameter KVA based on Fourier coeffieients."
    };
    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use. Defaults to 0 which means "
                   "sychronous execution.")
        ->check(CLI::Range(0, 65535));
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
                   "discretized. Defaults to 20.")
        ->check(CLI::Range(1, 1000000));
    app.add_option("-c,--sec-length", secLength,
                   "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
    app.add_option("--lambda-rel", lambdaRel,
                   "Tikhonov regularization parameter, defaults to 0.2.");
    app.add_flag("-v,--vizualize", vizualize, "Vizualize AIF and the basis.");
    app.add_option("--store-aif", storeAIF, "Store AIF into image file.");

    app.add_option("output_file", outputFile, "File KVA coefs.")->required();
    app.add_option("fitted_coeficients", fittedCoefficients,
                   "Fourier coeficients fited by the algorithm. Orderred from the first "
                   "coeficient that corresponds to the constant.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_flag("--allow-negative-values", allowNegativeValues, "Allow negative values.");
    app.add_flag("--only-aif", onlyaif, "Compute only AIF.");
    app.add_flag("--half-periodic-functions", halfPeriodicFunctions,
                 "Use Fourier basis and include half periodic functions.");

    try
    {
        app.parse(argc, argv);
        if(!(startTime < endTime))
        {
            io::throwerr("Start time %f must preceed end time %f.", startTime, endTime);
        }
        if(secLength == 0.0)
        {
            io::throwerr("Length of the second is not positive!");
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
    uint16_t dimx, dimy, dimz;
    io::DenFileInfo di(a.fittedCoefficients[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    LOGI << io::xprintf("Start time is %f and end time is %f", a.startTime, a.endTime);
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration
        = std::make_shared<util::FourierSeriesEvaluator>(a.fittedCoefficients, a.startTime,
                                                         a.endTime, !a.allowNegativeValues, a.halfPeriodicFunctions);
    // Vizualization
    float* convolutionMatrix = new float[a.granularity * a.granularity];
    float* aif = new float[a.granularity]();
    aif[0] = 0.5;
    aif[1] = 1.0;
    aif[2] = 0.5;
    utils::TikhonovInverse::precomputeConvolutionMatrix(a.granularity, aif, convolutionMatrix);
    if(a.vizualize || !a.storeAIF.empty())
    {
        util::FourierSeries b(a.fittedCoefficients.size(), a.startTime, a.endTime, 1);
        if(a.vizualize && !a.onlyaif)
        {
            b.plotFunctions();
        }
        std::vector<double> taxis;
        float* _taxis = new float[a.granularity];
        concentration->timeDiscretization(a.granularity, _taxis);
        std::vector<double> plotme;
        for(uint32_t i = 0; i != a.granularity; i++)
        {
            plotme.push_back(aif[i]);
            taxis.push_back(_taxis[i]);
        }
        plt::plot(taxis, plotme);
        if(a.vizualize)
        {
            plt::show();
        }
        if(!a.storeAIF.empty())
        {
            plt::save(a.storeAIF);
        }
        delete[] _taxis;
    }
    if(a.onlyaif)
    {
        return 0;
    }
    bool truncatedInstead = false;
    float lambdaRel = 0.2;
    utils::TikhonovInverse ti(lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, a.granularity);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> kva
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.outputFile, dimx, dimy, dimz);
    //    a.lambdaRel = 0.0;
    // Test what is the projection of convolutionMatrix to the last element of aif

    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, a.startTime, a.endTime,
                                    a.secLength, a.threads);

    LOGD << "KVA computation.";
    float* values = new float[dimx * dimy * a.granularity];
    // float* convol = new float[dimx * dimy * granularity]();
    float convol_i;
    for(int z = 0; z != dimz; z++)
    {
        concentration->frameTimeSeries(z, a.granularity, values);
        io::BufferedFrame2D<float> kvafr(std::numeric_limits<float>::lowest(), dimx, dimy);
        io::BufferedFrame2D<float> kvbfr(float(0), dimx, dimy);
        io::BufferedFrame2D<float> kvcfr(float(0), dimx, dimy);
        for(int x = 0; x != dimx; x++)
        {
            for(int y = 0; y != dimy; y++)
            {
                for(uint32_t i = 0; i != a.granularity; i++)
                {
                    convol_i = 0.0;
                    for(uint32_t j = 0; j != a.granularity; j++)
                    {
                        convol_i += convolutionMatrix[i * a.granularity + j]
                            * values[j * dimx * dimy + y * dimx + x];
                    }
                    if(convol_i * float(a.granularity - i - 1) / float(a.granularity - 1) > kvafr.get(x, y))
                    {
                        kvafr.set(convol_i * float(a.granularity - i - 1) / float(a.granularity - 1), x, y);
                    }
                    kvbfr.set(convol_i + kvbfr.get(x, y), x, y);
                    kvcfr.set(values[i * dimx * dimy + y * dimx + x] + kvcfr.get(x, y), x, y);
                }
            }
        }
        kva->writeFrame(kvafr, z);
        //    kvb->writeFrame(kvbfr, z);
        //    kvc->writeFrame(kvcfr, z);
    }
    delete[] values;
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
