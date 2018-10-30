// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "stringFormatter.h"
#include "FUN/LegendrePolynomialsExplicit.hpp"
#include "SVD/TikhonovInverse.hpp"
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
    int ifx, ify, ifz;

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
        int dimx = di.getNumCols();
        int dimy = di.getNumRows();
        int dimz = di.getNumSlices();
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
    // Logging setup
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/perfvizLog.csv"; // Set "" to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGD << "Logging!";

    // Command line parsing
    Arguments arg;
    int parseResult = arg.parseArguments(argc, argv);
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
    LOGD << "Parsing arguments!";
    int baseSize = arg.fittedCoeficients.size();
    io::DenFileInfo di(arg.fittedCoeficients[0]);
    int dimx = di.getNumCols();
    int dimy = di.getNumRows();
    int dimz = di.getNumSlices();
    double dt = (arg.startTime - arg.endTime) / double(arg.granularity - 1);
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator
        = std::make_shared<util::LegendrePolynomialsExplicit>(baseSize - 1, arg.startTime,
                                                              arg.endTime, 1);
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> fittedCoeficients;
    // Fill this array only by values without offset.
    std::shared_ptr<io::Frame2DReaderI<float>> pr;
    for(int i = 1; i != baseSize; i++)
    {
        pr = std::make_shared<io::DenFrame2DReader<float>>(arg.fittedCoeficients[i]);
        fittedCoeficients.push_back(pr);
    }
    utils::TimeSeriesDiscretizer tsd(baseFunctionsEvaluator, fittedCoeficients, arg.secLength,
                                     arg.threads);

    // Vizualization
    int granularity = arg.granularity;
    float* convolutionMatrix = new float[arg.granularity * arg.granularity];
    float* aif = new float[granularity];
    tsd.fillTimeValues(arg.ifx, arg.ify, arg.ifz, granularity, aif);
    tsd.fillConvolutionMatrix(arg.ifx, arg.ify, arg.ifz, granularity, convolutionMatrix);
    bool truncatedInstead = false;
    float lambdaRel = 0.2;
    utils::TikhonovInverse ti(lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, granularity);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> ttp_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/TTP.den", arg.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/CBF.den", arg.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/CBV.den", arg.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/MTT.den", arg.outputFolder.c_str()), dimx, dimy, dimz);
    LOGD << "Starting TTP computation.";
    tsd.computeTTP(granularity, ttp_w);
    LOGD << "Evaluating perfusion parameters CBV, CBF and MTT.";
    tsd.computeConvolvedParameters(convolutionMatrix, granularity, cbf_w, cbv_w, mtt_w);
    LOGD << "End of computation.";

    delete[] convolutionMatrix;
    delete[] aif;
}
