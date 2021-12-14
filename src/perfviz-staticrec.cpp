// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "CSVWriter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PerfusionVizualizationArguments.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "gitversion/version.h"
#include "matplotlibcpp.h"
#include "stringFormatter.h"
#include "utils/ReconstructedSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

namespace plt = matplotlibcpp;

using namespace KCT;
using namespace KCT::util;

/// Arguments of the main function.
class Args : public ArgumentsThreading, public PerfusionVizualizationArguments
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName)
        , PerfusionVizualizationArguments(argc, argv, prgName){};
    /// Function to parse function parameters.
    int parseArguments(int argc, char* argv[]);

    /// Folder to which output data after the linear regression.
    std::string outputFolder;

    /// Static reconstructions in a DEN format to use for linear regression.
    std::vector<std::string> coefficientVolumeFiles;

    /// Coordinates of arthery input function
    uint32_t ifx, ify, ifz;

    /**Vizualize base functions.
     *
     *If set vizualize base functions using Python.
     */
    bool allowNegativeValues = false;
};

void Args::defineArguments()
{
    cliApp->add_option("ifx", ifx, "Pixel based x coordinate of arthery input function")
        ->required();
    cliApp->add_option("ify", ify, "Pixel based y coordinate of arthery input function")
        ->required();
    cliApp->add_option("ifz", ifz, "Pixel based z coordinate of arthery input function")
        ->required();
    cliApp
        ->add_option("output_folder", outputFolder,
                     "Folder to which output data of perfusion coefficients.")
        ->required();
    cliApp
        ->add_option("static_reconstructions", coefficientVolumeFiles,
                     "Coeficients of the basis functions fited by the algorithm. Orderred in the "
                     "same order as the basis is sampled in a DEN file.")
        ->required()
        ->check(CLI::ExistingFile);

    CLI::Option_group* flow_og = cliApp->add_option_group("Program flow parameters");
    addThreadingArgs(flow_og);
    cliApp->add_flag("--allow-negative-values", allowNegativeValues, "Allow negative values.");

    addSweepArgs(false);
    addGranularity();
    addSecLength();
    addSettingsArgs();
    addVizualizationArgs(false);
}

int Args::postParse()
{
    std::string err;
    if(coefficientVolumeFiles.size() != 10)
    {
        err = io::xprintf("Number of sweeps is %d which is unusual number.",
                          coefficientVolumeFiles.size());
        LOGW << err;
    }
    if(outputFolder.compare("-") == 0)
    {
        stopAfterVizualization = true;
        LOGI << "No directory specified so that after visualization program ends.";
    } else
    {
        if(!io::isDirectory(outputFolder))
        {
            err = io::xprintf("The path %s does not encode a valid directory!",
                              outputFolder.c_str());
            LOGE << err;
            return -1;
        }
    }
    sweepCount = coefficientVolumeFiles.size();
    if(sweepCount < 2)
    {
        err = io::xprintf("Small number of input files %d.", coefficientVolumeFiles.size());
        LOGE << err;
        return 1;
    }
    if(secLength == 0.0)
    {
        err = "Length of the second is not positive!";
        LOGE << err;
        return 1;
    }
    if(vizualize)
    {
        showBasis = true;
        showAIF = true;
    }
    if(showBasis || showAIF)
    {
        vizualize = true;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    std::string prgInfo
        = "Visualization of perfusion parameters based on the spline fiting to the static "
          "reconstructions from C-Arm CT acquisition.";
    if(version::MODIFIED_SINCE_COMMIT == true)
    {
        prgInfo = io::xprintf("%s Dirty commit %s", prgInfo.c_str(), version::GIT_COMMIT_ID);
    } else
    {
        prgInfo = io::xprintf("%s Git commit %s", prgInfo.c_str(), version::GIT_COMMIT_ID);
    }
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.coefficientVolumeFiles[0]);
    int dimx = di.dimx();
    int dimy = di.dimy();
    int dimz = di.dimz();
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration
        = std::make_shared<util::ReconstructedSeriesEvaluator>(ARG.coefficientVolumeFiles,
                                                               ARG.sweepTime, ARG.sweepOffset);
    // Vizualization
    float* convolutionMatrix = new float[ARG.granularity * ARG.granularity];
    float* aif = new float[ARG.granularity];
    std::shared_ptr<util::ReconstructedSeriesEvaluator> _concentration
        = std::dynamic_pointer_cast<util::ReconstructedSeriesEvaluator>(concentration);
    if(ARG.allowNegativeValues)
    {
        _concentration->timeSeriesNativeNoOffsetNoTruncationIn(ARG.ifx, ARG.ify, ARG.ifz,
                                                               ARG.granularity, aif);
        for(uint32_t i = 0; i != ARG.granularity; i++)
        {
            aif[i] = aif[i] - aif[0];
        }
    } else
    {
        concentration->timeSeriesIn(ARG.ifx, ARG.ify, ARG.ifz, ARG.granularity, aif);
    }
    utils::TikhonovInverse::precomputeConvolutionMatrix(ARG.granularity, aif, convolutionMatrix);
    if(ARG.vizualize || !ARG.aifImageFile.empty() || !ARG.aifCsvFile.empty())
    {
        // See
        // https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
        if(!ARG.vizualize)
        {
            plt::backend("Agg");
        }
        float* _taxis = new float[ARG.granularity]();
        float* aif_native = new float[ARG.granularity]();
        concentration->timeDiscretization(ARG.granularity, _taxis);
        _concentration->timeSeriesNativeNoOffsetNoTruncationIn(ARG.ifx, ARG.ify, ARG.ifz,
                                                               ARG.granularity, aif_native);
        std::vector<double> taxis;
        std::vector<double> plotme;
        std::vector<double> taxis_scatter = _concentration->nativeTimeDiscretization();
        std::vector<double> plotme_scatter
            = _concentration->nativeValuesIn(ARG.ifx, ARG.ify, ARG.ifz);
        if(ARG.water_value > 0) // Put it to Hounsfield units
        {
            for(uint32_t i = 0; i != ARG.granularity; i++)
            {
                plotme.push_back(1000 * (aif_native[i] - ARG.water_value) / ARG.water_value);
            }
            for(uint32_t i = 0; i != plotme_scatter.size(); i++)
            {
                plotme_scatter[i] = 1000 * (plotme_scatter[i] - ARG.water_value) / ARG.water_value;
            }
        } else
        {
            for(uint32_t i = 0; i != ARG.granularity; i++)
            {
                plotme.push_back(aif_native[i]);
            }
        }
        if(!ARG.allowNegativeValues) // Truncate but preserve initial attenuation
        {
            for(uint32_t i = 0; i != ARG.granularity; i++)
            {
                plotme[i] = std::max(plotme[0], plotme[i]);
            }
        }
        for(uint32_t i = 0; i != ARG.granularity; i++)
        {
            taxis.push_back(_taxis[i] / 1000.0);
        }
        for(uint32_t i = 0; i != taxis_scatter.size(); i++)
        {
            taxis_scatter[i] = taxis_scatter[i] / 1000;
        }
        plt::title(
            io::xprintf("Time attenuation curve x=%d, y=%d, z=%d.", ARG.ifx, ARG.ify, ARG.ifz));
        plt::named_plot("Spline fit approximation", taxis, plotme);
        std::map<std::string, std::string> pltargs;
        pltargs.insert(std::pair<std::string, std::string>("Color", "Orange"));
        plt::scatter(taxis_scatter, plotme_scatter, 90.0, pltargs);
        plt::xlabel("Time [s]");
        if(ARG.water_value > 0)
        {
            plt::ylabel("Attenuation [HU]");
        } else
        {
            plt::ylabel("Attenuation");
        }
        plt::legend();
        if(!ARG.aifImageFile.empty())
        {
            plt::save(ARG.aifImageFile);
        }
        if(ARG.vizualize)
        {
            plt::show();
        }
        if(!ARG.aifCsvFile.empty())
        {
            io::CSVWriter csv(ARG.aifCsvFile, "\t", true);
            csv.writeLine(io::xprintf("perfviz-staticrec generated AIF (x,y,z)=(%d, %d, %d) with "
                                      "sweepTime=%f and sweepOffset=%f",
                                      ARG.ifx, ARG.ify, ARG.ifz, ARG.sweepTime, ARG.sweepOffset));
            csv.writeVector("time_staticrec_midsweep", taxis_scatter);
            csv.writeVector("value_staticrec__midsweep", plotme_scatter);
            csv.writeVector("time_staticrec_interp", taxis);
            csv.writeVector("value_staticrec_interp", plotme);
            csv.close();
        }
        delete[] _taxis;
        delete[] aif_native;
    }
    if(ARG.stopAfterVizualization)
    {
        return 0;
    }
    bool truncatedInstead = false;
    float lambdaRel = 0.075;
    utils::TikhonovInverse ti(lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, ARG.granularity);
    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, ARG.sweepOffset,
                                    ARG.sweepOffset
                                        + ARG.sweepTime * (ARG.coefficientVolumeFiles.size() - 1),
                                    ARG.secLength, ARG.threads);
    LOGD << "TTP computation.";
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> ttp_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/TTP.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
    tsd.computeTTP(ARG.granularity, ttp_w, aif);
    if(!ARG.stopAfterTTP)
    {
        LOGD << "CBV, CBF and MTT computation.";
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/CBF.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/CBV.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/MTT.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
        LOGI << io::xprintf("CBF time is set as %f.", ARG.cbf_time);
        tsd.computePerfusionParameters(ARG.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w,
                                       ARG.cbf_time);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
