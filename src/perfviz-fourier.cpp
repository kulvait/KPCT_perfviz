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
    std::string outputFolder;

    /// Projection files in a DEN format to use for linear regression.
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
    uint32_t granularity = 100;

    // Length of one second in the units of the domain
    float secLength = 1000;

    // Write only ttp
    bool onlyttp = false;
    bool allowNegativeValues = false;

    // Tikhonov regularization parameter
    float lambdaRel = 0.2;

#ifdef DEBUG
    /**Vizualize base functions.
     *
     *If set vizualize base functions using Python.
     */
    bool vizualize = false;
#endif
};

int Arguments::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Visualization of perfusion parameters CT based on Legendre fiting." };
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
                   "discretized. Defaults to 100.")
        ->check(CLI::Range(1, 1000000));
    app.add_option("-c,--sec-length", secLength,
                   "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
    app.add_option("--lambda-rel", lambdaRel,
                   "Tikhonov regularization parameter, defaults to 0.2.");
    app.add_option("ifx", ifx, "Pixel based x coordinate of arthery input function")->required();
    app.add_option("ify", ify, "Pixel based y coordinate of arthery input function")->required();
    app.add_option("ifz", ifz, "Pixel based z coordinate of arthery input function")->required();

    app.add_option("output_folder", outputFolder,
                   "Folder to which output data after the linear regression.")
        ->required()
        ->check(CLI::ExistingDirectory);
    app.add_option("fitted_coeficients", fittedCoefficients,
                   "Fourier coeficients fited by the algorithm. Orderred from the first "
                   "coeficient that corresponds to the constant.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_flag("--only-ttp", onlyttp, "Report TTP only.");
    app.add_flag("--allow-negative-values", allowNegativeValues, "Allow negative values.");
#ifdef DEBUG
    app.add_flag("-v,--vizualize", vizualize, "Vizualize engineered basis.");
#endif

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
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration
        = std::make_shared<util::FourierSeriesEvaluator>(a.fittedCoefficients.size(),
                                                         a.fittedCoefficients, a.startTime,
                                                         a.endTime, !a.allowNegativeValues);
    // Vizualization
    float* convolutionMatrix = new float[a.granularity * a.granularity];
    float* aif = new float[a.granularity];
    concentration->timeSeriesIn(a.ifx, a.ify, a.ifz, a.granularity, aif);
    utils::TikhonovInverse::precomputeConvolutionMatrix(a.granularity, aif, convolutionMatrix);
#ifdef DEBUG // Ploting AIF
    if(a.vizualize)
    {
        util::FourierSeries b(a.fittedCoefficients.size(), a.startTime, a.endTime, 1);
        b.plotFunctions();
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
        plt::show();
        delete[] _taxis;
    }
#endif
    bool truncatedInstead = false;
    //    a.lambdaRel = 0.0;
    utils::TikhonovInverse ti(a.lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, a.granularity);
    // Test what is the projection of convolutionMatrix to the last element of aif

    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, a.startTime, a.endTime,
                                    a.secLength, a.threads);
    if(a.vizualize)
    {
        tsd.visualizeConvolutionKernel(a.ifx, a.ify, a.ifz, a.granularity, convolutionMatrix);
    }
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> ttp_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/TTP.den", a.outputFolder.c_str()), dimx, dimy, dimz);
    LOGD << "TTP computation.";
    tsd.computeTTP(a.granularity, ttp_w, aif);
    if(!a.onlyttp)
    {
        LOGD << "CBV, CBF and MTT computation.";
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/CBF.den", a.outputFolder.c_str()), dimx, dimy, dimz);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/CBV.den", a.outputFolder.c_str()), dimx, dimy, dimz);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/MTT.den", a.outputFolder.c_str()), dimx, dimy, dimz);
        tsd.computePerfusionParameters(a.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
