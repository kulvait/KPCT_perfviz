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
#include "matplotlibcpp.h"
#include "stringFormatter.h"
#include "utils/ReconstructedSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

namespace plt = matplotlibcpp;

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
    float sweepTime = 5316;

    /// Offset at the beginning.
    float sweepOffset = 2072.5;

    /// Coordinates of arthery input function
    uint16_t ifx, ify, ifz;

    /// Granularity of the time is number of time points analyzed by i.e. convolution
    uint32_t granularity = 100;

    // Length of one second in the units of the domain
    float secLength = 1000;

    uint32_t baseSize;

    // If only ttp should be computed
    bool onlyttp = false;

    /**Vizualize base functions.
     *
     *If set vizualize base functions using Python.
     */
    bool vizualize = false;
    bool onlyaif = false;
    float water_value = -0.027;
    bool allowNegativeValues = false;
    /**
     * @brief File to store AIF.
     */
    std::string storeAIF = "";
};

int Arguments::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Visualization of perfusion parameters CT based on fiting of splines to a static "
                  "reconstructions." };
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
    app.add_option("-i,--sweep-time", sweepTime,
                   "Sweep time between the starts of two consecutive acquisitions in miliseconds."
                   "[defaults to 5316].")
        ->check(CLI::Range(0.0, 100000.0));
    app.add_option("-e,--sweep-offset", sweepOffset,
                   "Offset at the beginning and at the end of aquisition in seconds/1000."
                   "[defaults to 2072.5].")
        ->check(CLI::Range(0.0, 100000.0));
    app.add_option("-g,--granularity", granularity,
                   "Granularity of the time is number of time points to which time interval is "
                   "discretized. Defaults to 100.")
        ->check(CLI::Range(1, 1000000));
    app.add_option("-c,--sec-length", secLength,
                   "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
    app.add_flag("-v,--vizualize", vizualize, "Vizualize AIF.");
    app.add_option("--water-value", water_value,
                   "If the AIF vizualization should be in HU, use this water_value, default is negative value to show normal values, reasonable value 0.027.");
    app.add_flag("--allow-negative-values", allowNegativeValues, "Allow negative values.");
    app.add_option("--store-aif", storeAIF, "Store AIF into image file.");
    app.add_flag("--only-aif", onlyaif, "Vizualize only aif.");
    app.add_flag("--only-ttp", onlyttp, "Compute only ttp.");

    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use. Defaults to 0 which means "
                   "sychronous execution.")
        ->check(CLI::Range(0, 65535));

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
    std::shared_ptr<util::ReconstructedSeriesEvaluator> _concentration
        = std::dynamic_pointer_cast<util::ReconstructedSeriesEvaluator>(concentration);
    if(a.allowNegativeValues)
    {
        _concentration->timeSeriesNativeNoOffsetNoTruncationIn(a.ifx, a.ify, a.ifz, a.granularity,
                                                               aif);
        for(uint32_t i = 0; i != a.granularity; i++)
        {
            aif[i] = aif[i] - aif[0];
        }
    } else
    {
        concentration->timeSeriesIn(a.ifx, a.ify, a.ifz, a.granularity, aif);
    }
    utils::TikhonovInverse::precomputeConvolutionMatrix(a.granularity, aif, convolutionMatrix);
    if(a.vizualize || !a.storeAIF.empty())
    {
        float* _taxis = new float[a.granularity]();
        float* aif_native = new float[a.granularity]();
        concentration->timeDiscretization(a.granularity, _taxis);
        _concentration->timeSeriesNativeNoOffsetNoTruncationIn(a.ifx, a.ify, a.ifz, a.granularity,
                                                               aif_native);
        std::vector<double> taxis;
        std::vector<double> plotme;
        std::vector<double> taxis_scatter = _concentration->nativeTimeDiscretization();
        std::vector<double> plotme_scatter = _concentration->nativeValuesIn(a.ifx, a.ify, a.ifz);
        if(a.water_value > 0) // Put it to Hounsfield units
        {
            for(uint32_t i = 0; i != a.granularity; i++)
            {
                plotme.push_back(1000 * (aif_native[i] - a.water_value) / a.water_value);
            }
            for(uint32_t i = 0; i != plotme_scatter.size(); i++)
            {
                plotme_scatter[i] = 1000 * (plotme_scatter[i] - a.water_value) / a.water_value;
            }
        } else
        {
            for(uint32_t i = 0; i != a.granularity; i++)
            {
                plotme.push_back(aif_native[i]);
            }
        }
        if(!a.allowNegativeValues) // Truncate but preserve initial attenuation
        {
            for(uint32_t i = 0; i != a.granularity; i++)
            {
                plotme[i] = std::max(plotme[0], plotme[i]);
            }
        }
        for(uint32_t i = 0; i != a.granularity; i++)
        {
            taxis.push_back(_taxis[i] / 1000.0);
        }
        for(uint32_t i = 0; i != taxis_scatter.size(); i++)
        {
            taxis_scatter[i] = taxis_scatter[i] / 1000;
        }
        plt::title(io::xprintf("Time attenuation curve x=%d, y=%d, z=%d.", a.ifx, a.ify, a.ifz));
        plt::named_plot("Spline fit approximation", taxis, plotme);
        std::map<std::string, std::string> pltargs;
        pltargs.insert(std::pair<std::string, std::string>("Color", "Orange"));
        plt::scatter(taxis_scatter, plotme_scatter, 90.0, pltargs);
        plt::xlabel("Time [s]");
        if(a.water_value > 0)
        {
            plt::ylabel("Attenuation [HU]");
        } else
        {
            plt::ylabel("Attenuation");
        }
        plt::legend();
        if(a.vizualize)
        {
            plt::show();
        }
        if(!a.storeAIF.empty())
        {
            plt::save(a.storeAIF);
        }
        delete[] _taxis;
        delete[] aif_native;
    }
    if(a.onlyaif)
    {
        return 0;
    }
    bool truncatedInstead = false;
    float lambdaRel = 0.075;
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
    tsd.computeTTP(a.granularity, ttp_w, aif);
    if(!a.onlyttp)
    {
        LOGD << "CBV, CBF and MTT computation.";
        tsd.computePerfusionParameters(a.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
