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
#include "SVD/TikhonovInverse.hpp"
#include "stringFormatter.h"
#include "utils/ReconstructedSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

#if DEBUG
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

    /// Static reconstructions in a DEN format to use for linear regression.
    std::vector<std::string> coefficientVolumeFiles;

    /// Number of threads
    uint16_t threads = 0;

    /// Times between starts of consecutive sweeps [ms].
    float sweepTime = 6206;

    /// Offset at the beginning.
    float sweepOffset = 2050;

    /// Coordinates of arthery input function
    uint16_t ifx, ify, ifz;

    /// Granularity of the time is number of time points analyzed by i.e. convolution
    uint32_t granularity = 100;

    // Length of one second in the units of the domain
    float secLength = 1000;

    uint32_t baseSize;

    bool onlyttp = false;

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
    CLI::App app{ "Visualization of perfusion parameters CT based on fiting of splines to a static "
                  "reconstructions." };
    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use. Defaults to 0 which means "
                   "sychronous execution.")
        ->check(CLI::Range(0, 65535));
    app.add_option("-i,--sweep-time", sweepTime,
                   "Sweep time between the starts of two consecutive acquisitions in miliseconds."
                   "[defaults to 6206, 247*16.6+2089].")
        ->check(CLI::Range(0.0, 100000.0));
    app.add_option("-e,--sweep-offset", sweepOffset,
                   "Offset at the beginning and at the end of aquisition in seconds/1000."
                   "[defaults to 2050, 247*16.6/2].")
        ->check(CLI::Range(0.0, 100000.0));
    app.add_option("-g,--granularity", granularity,
                   "Granularity of the time is number of time points to which time interval is "
                   "discretized. Defaults to 100.")
        ->check(CLI::Range(1, 1000000));
    app.add_option("-c,--sec-length", secLength,
                   "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
#if DEBUG
    app.add_flag("-v,--vizualize", vizualize, "Vizualize engineered basis.");
#endif
    app.add_flag("--ttp", onlyttp, "Compute only ttp.");
    app.add_option("ifx", ifx, "Pixel based x coordinate of arthery input function")->required();
    app.add_option("ify", ify, "Pixel based y coordinate of arthery input function")->required();
    app.add_option("ifz", ifz, "Pixel based z coordinate of arthery input function")->required();

    app.add_option("output_folder", outputFolder,
                   "Folder to which output data of perfusion coefficients.")
        ->required()
        ->check(CLI::ExistingDirectory);
    app.add_option("static_reconstructions", coefficientVolumeFiles,
                   "Coeficients of the basis functions fited by the algorithm. Orderred in the "
                   "same order as the basis is sampled in a DEN file.")
        ->required()
        ->check(CLI::ExistingFile);

    try
    {
        app.parse(argc, argv);
        if(coefficientVolumeFiles.size() != 10)
        {
            std::string err = io::xprintf("Number of sweeps is %d which is unusual number.",
                                          coefficientVolumeFiles.size());
            LOGW << err;
        }
        if(coefficientVolumeFiles.size() < 2)
        {
            std::string err
                = io::xprintf("Small number of input files %d.", coefficientVolumeFiles.size());
            LOGE << err;
            io::throwerr(err);
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
    io::DenFileInfo di(a.coefficientVolumeFiles[0]);
    int dimx = di.dimx();
    int dimy = di.dimy();
    int dimz = di.dimz();
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration
        = std::make_shared<util::ReconstructedSeriesEvaluator>(a.coefficientVolumeFiles,
                                                               a.sweepTime, a.sweepOffset);
    // Vizualization
    float* convolutionMatrix = new float[a.granularity * a.granularity];
    float* aif = new float[a.granularity];
    concentration->timeSeriesIn(a.ifx, a.ify, a.ifz, a.granularity, aif);
    utils::TikhonovInverse::precomputeConvolutionMatrix(a.granularity, aif, convolutionMatrix);
#if DEBUG // Ploting AIF
    if(a.vizualize)
    {
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
    float lambdaRel = 0.2;
    utils::TikhonovInverse ti(lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, a.granularity);
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
    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, a.sweepOffset,
                                    a.sweepOffset
                                        + a.sweepTime * (a.coefficientVolumeFiles.size() - 1),
                                    a.secLength, a.threads);
    LOGD << "TTP computation.";
    tsd.computeTTP(a.granularity, ttp_w);
    LOGD << "CBV, CBF and MTT computation.";
    if(!a.onlyttp)
    {
        tsd.computePerfusionParameters(a.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
