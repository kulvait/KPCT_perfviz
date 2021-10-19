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
#include "rawop.h"
#include "stringFormatter.h"
#include "utils/CTEvaluator.hpp"
#include "utils/FourierSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

#if DEBUG
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
#endif

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

    /// Folder to which output data after the linear regression.
    std::string outputFolder;

    /// Static reconstructions in a DEN format to use for linear regression.
    std::vector<std::string> coefficientVolumeFiles;

    /// Additional information about time offsets and other data
    std::vector<std::string> tickFiles;

    /// Coordinates of arthery input function
    uint32_t ifx, ify, ifz;

    uint32_t baseSize;

    // If only ttp should be computed
    bool onlyttp = false;

    /**Vizualize base functions.
     *
     *If set vizualize base functions using Python.
     */
    std::string vizualizeFourier;
    bool allowNegativeValues = false;
    float cbfTime = 5.0;
};

void Args::defineArguments()
{
    std::string optstr;
    cliApp->add_option("ifx", ifx, "Voxel x coordinate of the arthery input function.")->required();
    cliApp->add_option("ify", ify, "Voxel y coordinate of the arthery input function.")->required();
    cliApp->add_option("ifz", ifz, "Voxel z coordinate of the arthery input function.")->required();
    cliApp
        ->add_option("output_folder", outputFolder,
                     "Folder to which output data after the linear regression, specify - for no "
                     "computation of perfusion parameters.")
        ->required();
    cliApp
        ->add_option("static_reconstructions", coefficientVolumeFiles,
                     "Input volumes in a DEN format to represent CT volumes of perfusion scan. "
                     "These files must have .den suffix and corresponding files with .tick suffix "
                     "must exist. Tick files encode timings of the acquisition.")

        ->required()
        ->check(CLI::ExistingFile);
    addGranularity();
    addSecLength();
    addThreadingArgs();
    addVizualizationArgs(false);
    og_vizualization->add_option(
        "--vizualize-fourier", vizualizeFourier,
        "Prefix of harmonic coefficients to vizualize along the data. It will be added "
        "0.den to k.den when all integers from 0 to k are present.");
    cliApp->add_flag("--allow-negative-values", allowNegativeValues,
                     "AIF is usually truncated by 0 and does not allow negative values.");

    optstr
        = io::xprintf("The CBF value should be value at the t=0 of the deconvolution function. For "
                      "improved stability we put a maximum of this value over the interval [0.0, "
                      "cbfTime] in seconds. Defaults to %0.2fs.",
                      cbfTime);
    cliApp->add_option("--cbf-time", cbfTime, optstr);
}

int Args::postParse()
{
    std::string err;
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
    if(coefficientVolumeFiles.size() != 30 && coefficientVolumeFiles.size() != 15)
    {
        err = io::xprintf("Number of sweeps is %d which is unusual number.",
                          coefficientVolumeFiles.size());
        LOGW << err;
    }
    if(coefficientVolumeFiles.size() < 2)
    {
        err = io::xprintf("Small number of input files %d.", coefficientVolumeFiles.size());
        LOGE << err;
        return -1;
    }
    if(secLength == 0.0)
    {
        err = io::xprintf("Length of the second is not positive!");
        LOGE << err;
        return -1;
    }
    for(std::string f : coefficientVolumeFiles)
    {
        std::string tickFile = f.substr(0, f.find_last_of(".")) + ".tick";
        if(!io::isRegularFile(tickFile))
        {
            std::string err = io::xprintf("Tick file %s does not exists.", tickFile.c_str());
            LOGE << err;
            throw std::runtime_error(err);
        } else
        {
            tickFiles.push_back(tickFile);
        }
    }
    return 0;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    std::string prgInfo
        = "Visualization of perfusion parameters based on the CT perfusion acquisition given in "
          "the form of reconstructed volumes and corresponding tick files.";
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
    uint16_t dimx = di.dimx();
    uint16_t dimy = di.dimy();
    uint16_t dimz = di.dimz();
    // Computation of the start and end of the interval
    std::shared_ptr<io::Frame2DReaderI<float>> startData
        = std::make_shared<io::DenFrame2DReader<float>>(ARG.tickFiles[0]);
    std::shared_ptr<io::Frame2DReaderI<float>> endData
        = std::make_shared<io::DenFrame2DReader<float>>(ARG.tickFiles[ARG.tickFiles.size() - 1]);
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
                        intervalStart, intervalEnd, ARG.secLength);
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration
        = std::make_shared<util::CTEvaluator>(ARG.coefficientVolumeFiles, ARG.tickFiles);
    std::shared_ptr<util::CTEvaluator> _concentration
        = std::dynamic_pointer_cast<util::CTEvaluator>(concentration);
    // Vizualization
    float* convolutionMatrix = new float[ARG.granularity * ARG.granularity];
    float* aif = new float[ARG.granularity];
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
        float* _taxis = new float[ARG.granularity]();
        float* aif_native = new float[ARG.granularity]();
        concentration->timeDiscretization(ARG.granularity, _taxis);
        _concentration->timeSeriesNativeNoOffsetNoTruncationIn(ARG.ifx, ARG.ify, ARG.ifz,
                                                               ARG.granularity, aif_native);
        std::vector<double> taxis;
        std::vector<double> plotme;
        std::vector<double> plotme_fourier;
        std::vector<double> taxis_scatter = _concentration->nativeTimeDiscretization(ARG.ifz);
        std::vector<double> plotme_scatter
            = _concentration->nativeValuesIn(ARG.ifx, ARG.ify, ARG.ifz);
        if(ARG.vizualizeFourier != "")
        {
            std::vector<std::string> fourierVolumeFiles;
            uint32_t k = 0;
            std::string f = io::xprintf("%s%d.den", ARG.vizualizeFourier.c_str(), k);
            while(io::isRegularFile(f))
            {
                fourierVolumeFiles.emplace_back(f);
                k++;
                f = io::xprintf("%s%d.den", ARG.vizualizeFourier.c_str(), k);
            }
            std::shared_ptr<util::FourierSeriesEvaluator> concentrationFourier
                = std::make_shared<util::FourierSeriesEvaluator>(fourierVolumeFiles, intervalStart,
                                                                 intervalEnd, false, false);
            float* aif_fourier = new float[ARG.granularity];
            auto offsetReader
                = std::make_shared<io::DenFrame2DReader<float>>(fourierVolumeFiles[0]);
            auto aifframe = offsetReader->readFrame(ARG.ifz);
            float aifofset = aifframe->get(ARG.ifx, ARG.ify)
                + concentrationFourier->valueAt_intervalStart(ARG.ifx, ARG.ify, ARG.ifz);
            concentrationFourier->timeSeriesIn(ARG.ifx, ARG.ify, ARG.ifz, ARG.granularity,
                                               aif_fourier);
            for(uint32_t i = 0; i != ARG.granularity; i++)
            {
                float v = aif_fourier[i] + aifofset;
                if(ARG.water_value > 0)
                {
                    float hu = 1000 * (v / ARG.water_value - 1.0);
                    plotme_fourier.push_back(hu);
                } else
                {
                    plotme_fourier.push_back(v);
                }
            }
            delete[] aif_fourier;
        }
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
            taxis.push_back(_taxis[i]);
        }
        // See
        // https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
        if(!ARG.vizualize)
        {
            plt::backend("Agg");
        }
        plt::title(
            io::xprintf("Time attenuation curve x=%d, y=%d, z=%d.", ARG.ifx, ARG.ify, ARG.ifz));
        if(ARG.vizualizeFourier != "")
        {
            plt::named_plot("Harmonic approximation", taxis, plotme_fourier);
        }
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
        if(ARG.showAIF)
        {
            plt::show();
        }
        if(!ARG.aifImageFile.empty())
        {
            plt::save(ARG.aifImageFile);
        }
        if(!ARG.aifCsvFile.empty())
        {
            io::CSVWriter csv(ARG.aifCsvFile, "\t", true);
            csv.writeLine(io::xprintf("perfviz-ctrec generated AIF (x,y,z)=(%d, %d, %d)", ARG.ifx,
                                      ARG.ify, ARG.ifz));
            csv.writeVector("time_ct_bp", taxis_scatter);
            csv.writeVector("value_ct_bp", plotme_scatter);
            csv.writeVector("time_ct_interp", taxis);
            csv.writeVector("value_ct_interp", plotme);
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
    float lambdaRel = 0.2;
    // lambdaRel = 0.075;
    // lambdaRel = 0.15;
    utils::TikhonovInverse ti(lambdaRel, truncatedInstead);
    ti.computePseudoinverse(convolutionMatrix, ARG.granularity);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> ttp_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/TTP.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbf_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/CBF.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> cbv_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/CBV.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> mtt_w
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
            io::xprintf("%s/MTT.den", ARG.outputFolder.c_str()), dimx, dimy, dimz);
    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, intervalStart, intervalEnd,
                                    ARG.secLength, ARG.threads);
    LOGD << "TTP computation.";
    // tsd.computeTTP(ARG.granularity, ttp_w, aif);//Peak of aif is minimum index in output
    tsd.computeTTP(ARG.granularity, ttp_w, nullptr);
    if(!ARG.onlyttp)
    {
        LOGD << "CBV, CBF and MTT computation.";
        tsd.computePerfusionParameters(ARG.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w,
                                       ARG.cbfTime);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
