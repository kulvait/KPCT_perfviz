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
#include "utils/CTEvaluator.hpp"
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

    /// Additional information about time offsets and other data
    std::vector<std::string> tickFiles;

    /// Number of threads
    uint16_t threads = 0;

    /// Coordinates of arthery input function
    uint16_t ifx, ify, ifz;

    /// Granularity of the time is number of time points analyzed by i.e. convolution
    uint32_t granularity = 100;

    // Length of one second in the units of the domain
    float secLength = 1.0;

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
    /**
     * @brief File to store AIF.
     */
    std::string storeAIF = "";
};

int Arguments::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Visualization of perfusion parameters CT based on fiting of splines to a static "
                  "reconstructions of CT data." };
    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use. Defaults to 0 which means "
                   "sychronous execution.")
        ->check(CLI::Range(0, 65535));
    app.add_option("-g,--granularity", granularity,
                   "Granularity of the time is number of time points to which time interval is "
                   "discretized. Defaults to 100.")
        ->check(CLI::Range(1, 1000000));
    app.add_option("-c,--sec-length", secLength,
                   "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
    app.add_flag("-v,--vizualize", vizualize, "Vizualize AIF and the basis.");
    app.add_option("--store-aif", storeAIF, "Store AIF into image file.");
    app.add_flag("--only-aif", onlyaif, "Compute only AIF.");
    app.add_flag("--only-ttp", onlyttp, "Compute only TTP.");
    app.add_option("--water-value", water_value,
                   "If the AIF vizualization should be in HU, use this water_value.");
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
        if(coefficientVolumeFiles.size() != 30 && coefficientVolumeFiles.size() != 15)
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
        for(std::string f : coefficientVolumeFiles)
        {
            std::string tickFile = f.substr(0, f.find_last_of(".")) + ".tick";
            if(!io::fileExists(tickFile))
            {
                std::string err
                    = io::xprintf("Small number of input files %d.", coefficientVolumeFiles.size());
                LOGE << err;
                throw new std::runtime_error(err);

            } else
            {
                tickFiles.push_back(tickFile);
            }
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
    uint16_t dimx = di.dimx();
    uint16_t dimy = di.dimy();
    uint16_t dimz = di.dimz();
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration
        = std::make_shared<util::CTEvaluator>(a.coefficientVolumeFiles, a.tickFiles);
    // Vizualization
    float* convolutionMatrix = new float[a.granularity * a.granularity];
    float* aif = new float[a.granularity];
    concentration->timeSeriesIn(a.ifx, a.ify, a.ifz, a.granularity, aif);
    utils::TikhonovInverse::precomputeConvolutionMatrix(a.granularity, aif, convolutionMatrix);
    if(a.vizualize || !a.storeAIF.empty())
    {
        std::vector<double> taxis;
        float* _taxis = new float[a.granularity];
        concentration->timeDiscretization(a.granularity, _taxis);
        std::vector<double> plotme;
        for(uint32_t i = 0; i != a.granularity; i++)
        {
            if(a.water_value > 0)
            {
                plotme.push_back(aif[i] * 1000 / a.water_value);
            } else
            {
                plotme.push_back(aif[i]);
            }
            taxis.push_back(_taxis[i]);
        }
        plt::title(io::xprintf("AIF x=%d, y=%d, z=%d", a.ifx, a.ify, a.ifz));
        plt::ylabel("Attenuation");
        plt::xlabel("Time [s]");
        plt::named_plot("Fit", taxis, plotme);
        std::shared_ptr<util::CTEvaluator> conct
            = std::dynamic_pointer_cast<util::CTEvaluator>(concentration);
        plt::plot(taxis, plotme);
        std::vector<double> taxis_scatter = conct->nativeTimeDiscretization(a.ifz);
        std::vector<double> plotme_scatter = conct->nativeValuesIn(a.ifx, a.ify, a.ifz);
        if(a.water_value > 0)
        {
            for(uint32_t i = 0; i != plotme_scatter.size(); i++)
            {
                plotme_scatter[i] = plotme_scatter[i] * 1000 / a.water_value;
            }
        }
        plt::named_plot("Original", taxis_scatter, plotme_scatter);
        // plt::legend();
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
    // lambdaRel = 0.075;
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
    // Computation of the start and end of the interval

    std::shared_ptr<io::Frame2DReaderI<float>> startData
        = std::make_shared<io::DenFrame2DReader<float>>(a.tickFiles[0]);
    std::shared_ptr<io::Frame2DReaderI<float>> endData
        = std::make_shared<io::DenFrame2DReader<float>>(a.tickFiles[a.tickFiles.size() - 1]);
    std::shared_ptr<io::Frame2DI<float>> fs, fe;
    fs = startData->readFrame(0);
    fe = endData->readFrame(0);
    // Fifth element of frame is time
    float intervalStart, intervalEnd;
    intervalStart = fs->get(4, 0);
    intervalEnd = fe->get(4, 0);
    float start, end;
    for(std::size_t z = 0; z != dimz; z++)
    {
        fs = startData->readFrame(z);
        fe = endData->readFrame(z);
        start = fs->get(4, 0);
        end = fe->get(4, 0);
        if(start > intervalStart)
        {
            intervalStart = start;
        }
        if(end < intervalEnd)
        {
            intervalEnd = end;
        }
    }
    LOGD << io::xprintf("Computed interval start is %f and interval end is %f and secLength is %f",
                        intervalStart, intervalEnd, a.secLength);
    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, intervalStart, intervalEnd,
                                    a.secLength, a.threads);
    LOGD << "TTP computation.";
    // tsd.computeTTP(a.granularity, ttp_w, aif);//Peak of aif is minimum index in output
    tsd.computeTTP(a.granularity, ttp_w, nullptr);
    if(!a.onlyttp)
    {
        LOGD << "CBV, CBF and MTT computation.";
        tsd.computePerfusionParameters(a.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
