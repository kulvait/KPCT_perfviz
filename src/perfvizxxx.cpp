// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "FUN/StepFunction.hpp"
#include "Frame2DReaderI.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "stringFormatter.h"
#include "utils/TimeSeriesDiscretizer.hpp"

using namespace CTL;

/// Arguments of the main function.
struct Arguments
{
    /// Function to parse function parameters.
    int parseArguments(int argc, char* argv[]);

    /// Folder to which output data after the linear regression.
    std::string outputFolder;

    /// Projection files in a DEN format to use for linear regression.
    std::vector<std::string> fittedCoeficients;

    /// Number of threads
    int threads = 1;

    /// Controls the size of the time interval [ms] that should be identified with 0.0.
    float startTime = 4117, endTime = 56000;

    /// Coordinates of arthery input function
    uint16_t ifx, ify, ifz;

    /// Granularity of the time is number of time points analyzed by i.e. convolution
    int granularity = 100;

    // Length of one second in the units of the domain
    float secLength = 1000;
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
    app.add_option("ifx", ifx, "Pixel based x coordinate of arthery input function")->required();
    app.add_option("ify", ify, "Pixel based y coordinate of arthery input function")->required();
    app.add_option("ifz", ifz, "Pixel based z coordinate of arthery input function")->required();

    app.add_option("output_folder", outputFolder,
                   "Folder to which output data after the linear regression.")
        ->required()
        ->check(CLI::ExistingDirectory);
    app.add_option("fitted_coeficients", fittedCoeficients,
                   "Legendre coeficients fited by the algorithm. Orderred from the first "
                   "coeficient that corresponds to the constant.")
        ->required()
        ->check(CLI::ExistingFile);

    try
    {
        app.parse(argc, argv);
        io::DenFileInfo di(fittedCoeficients[0]);
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
/*
    int baseSize = a.fittedCoeficients.size();
    io::DenFileInfo di(a.fittedCoeficients[0]);
    int dimx = di.dimx();
    int dimy = di.dimy();
    int dimz = di.dimz();
    double christina[] = {
        0.1694,  0.1696,  0.1698,  0.1699,  0.1705,  0.1716,  0.1746,  0.1825,  0.1953,  0.2115,
        0.2201,  0.2194,  0.2157,  0.2077,  0.2020,  0.1955,  0.1915,  0.1887,  0.1867,  0.1860,
        0.1857,  0.1854,  0.1853,  0.1849,  0.1848,  0.1841,  0.1837,  0.1830,

        0.2305,  0.2344,  0.2232,  0.2301,  0.2216,  0.2074,  0.1744,  0.0770,  -0.0700, -0.2687,
        -0.3926, -0.3968, -0.3546, -0.2439, -0.1514, -0.0679, -0.0141, 0.0231,  0.0559,  0.0617,
        0.0706,  0.0690,  0.0761,  0.0760,  0.0719,  0.0874,  0.0902,  0.0966,

        0.0863,  0.0882,  0.0685,  0.0937,  0.0785,  0.1173,  0.1596,  0.2843,  0.4406,  0.4470,
        0.2234,  0.1195,  -0.2047, -0.1982, -0.3100, -0.2145, -0.2327, -0.1463, -0.1655, -0.0716,
        -0.1163, -0.0538, -0.0740, -0.0876, -0.0916, -0.0927, -0.0662, -0.0679
    };
    int valuesPerFunction = 28;
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator
        = std::make_shared<util::StepFunction>(christina, 3, valuesPerFunction, a.startTime, a.endTime);
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> fittedCoeficients;
    // Fill this array only by values without offset.
    std::shared_ptr<io::Frame2DReaderI<float>> pr;
    for(int i = 1; i != baseSize; i++)
    {
        pr = std::make_shared<io::DenFrame2DReader<float>>(a.fittedCoeficients[i]);
        fittedCoeficients.push_back(pr);
    }
    utils::TimeSeriesDiscretizer tsd(baseFunctionsEvaluator, fittedCoeficients, a.secLength,
                                     a.threads);

    // Vizualization
    int granularity = a.granularity;
    float* convolutionMatrix = new float[granularity * granularity];
    float* aif = new float[granularity];
    tsd.fillTimeValues(a.ifx, a.ify, a.ifz, granularity, aif);
    tsd.fillConvolutionMatrix(a.ifx, a.ify, a.ifz, granularity, convolutionMatrix);
    bool truncatedInstead = false;
    float lambdaRel = 0.2;
    utils::TikhonovInverse ti(lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, granularity);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> ttp_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/TTP.den", a.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/CBF.den", a.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/CBV.den", a.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/MTT.den", a.outputFolder.c_str()), dimx, dimy, dimz);
    LOGD << "Starting TTP computation.";
    tsd.computeTTP(granularity, ttp_w);
    LOGD << "Evaluating perfusion parameters CBV, CBF and MTT.";
    tsd.computeConvolvedParameters(convolutionMatrix, granularity, cbf_w, cbv_w, mtt_w);
    LOGD << "End of computation.";

    delete[] convolutionMatrix;
    delete[] aif;
*/
}
