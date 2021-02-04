// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "FUN/LegendrePolynomialsExplicit.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PerfusionVizualizationArguments.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "stringFormatter.h"
#include "utils/Attenuation4DEvaluatorI.hpp"
#include "utils/PolynomialSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"
#include "utils/ReconstructedSeriesEvaluator.hpp"

#ifdef DEBUG
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
#endif

using namespace CTL;
using namespace CTL::util;

/// Arguments of the main function.
struct Args : public ArgumentsThreading, public PerfusionVizualizationArguments
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

    /// Projection files in a DEN format to use for linear regression.
    std::vector<std::string> fittedCoefficients;

    /// Coordinates of arthery input function
    uint16_t ifx, ify, ifz;

    // Specify basis
    bool allowNegativeValues = false;
    bool chebyshev = false;
    bool legendre = false;

    // Tikhonov regularization parameter
    float lambdaRel = 0.2;
};

void Args::defineArguments()
{
    cliApp->add_option("ifx", ifx, "Voxel x coordinate of the arthery input function.")->required();
    cliApp->add_option("ify", ify, "Voxel y coordinate of the arthery input function.")->required();
    cliApp->add_option("ifz", ifz, "Voxel z coordinate of the arthery input function.")->required();
    cliApp
        ->add_option("output_folder", outputFolder,
                     "Folder to which output data after the linear regression. It needs to be "
                     "directory or symbol - to indicate that no computation should be performed.")
        ->required();
    cliApp
        ->add_option("fitted_coeficients", fittedCoefficients,
                     "Basis coeficients fited by the algorithm. Orderred from the first "
                     "coeficient that corresponds to the constant.")
        ->required()
        ->check(CLI::ExistingFile);
    CLI::Option_group* pol_og = cliApp->add_option_group("Polynomial type");
    pol_og->add_flag("--chebyshev", chebyshev, "Use Chebyshev polynomials.");
    pol_og->add_flag("--legendre", legendre, "Use Legendre polynomials.");
    pol_og->require_option(1);
    cliApp->add_flag("--allow-negative-values", allowNegativeValues, "Allow negative values.");
    addIntervalArgs(true);
    addVizualizationArgs(true);
    CLI::Option_group* flow_og = cliApp->add_option_group("Program flow parameters");
    addThreadingArgs(flow_og);
    cliApp->add_option("--lambda-rel", lambdaRel,
                       "Tikhonov regularization parameter, defaults to 0.2.");
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
    if(!(startTime < endTime))
    {
        err = io::xprintf("Start time %f must preceed end time %f.", startTime, endTime);
        LOGE << err;
        return -1;
    }
    if(secLength == 0.0)
    {
        err = io::xprintf("Length of the second is not positive!");
        LOGE << err;
        return -1;
    }
    if(vizualize)
    {
        showBasis = true;
        showAIF = true;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv,
             "Visualization of the CT perfusion parameters using deconvolution based technique "
             "based on fitting of the orthogonal polynomials.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    uint16_t dimx, dimy, dimz;
    io::DenFileInfo di(ARG.fittedCoefficients[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    util::polynomialType pt;
    if(ARG.chebyshev)
    {
        pt = util::polynomialType::Chebyshev;
    } else if(ARG.legendre)
    {
        pt = util::polynomialType::Legendre;
    }
    std::shared_ptr<util::PolynomialSeriesEvaluator> concentration
        = std::make_shared<util::PolynomialSeriesEvaluator>(
            ARG.fittedCoefficients.size() - 1, ARG.fittedCoefficients, ARG.startTime, ARG.endTime,
            !ARG.allowNegativeValues, pt);
    // Vizualization
    auto offsetReader = std::make_shared<io::DenFrame2DReader<float>>(ARG.fittedCoefficients[0]);
    auto aifframe = offsetReader->readFrame(ARG.ifz);
    float aifofset = aifframe->get(ARG.ifx, ARG.ify)
        + concentration->valueAt_intervalStart(ARG.ifx, ARG.ify, ARG.ifz);
    float* convolutionMatrix = new float[ARG.granularity * ARG.granularity];
    float* aif = new float[ARG.granularity];
    concentration->timeSeriesIn(ARG.ifx, ARG.ify, ARG.ifz, ARG.granularity, aif);
    utils::TikhonovInverse::precomputeConvolutionMatrix(ARG.granularity, aif, convolutionMatrix);
    if(ARG.showBasis || !ARG.basisImageFile.empty())
    {
        std::shared_ptr<util::VectorFunctionI> b;
        if(ARG.chebyshev)
        {
            b = std::make_shared<util::ChebyshevPolynomialsExplicit>(
                ARG.fittedCoefficients.size() - 1, ARG.startTime, ARG.endTime, true, 1);
        } else
        {
            b = std::make_shared<util::LegendrePolynomialsExplicit>(
                ARG.fittedCoefficients.size() - 1, ARG.startTime, ARG.endTime, true, 1);
        }
        if(ARG.showBasis)
        {
            b->plotFunctions();
        }
        if(!ARG.basisImageFile.empty())
        {
            b->storeFunctions(ARG.basisImageFile);
        }
    }

    if(ARG.showAIF || !ARG.aifImageFile.empty())
    {
        std::string polynomialType;
        if(ARG.chebyshev)
        {
            polynomialType = "Chebyshev";
        } else
        {
            polynomialType = "Legendre";
        }
        plt::title(io::xprintf("Time attenuation curve, TST %s, x=%d, y=%d, z=%d.", polynomialType.c_str(), ARG.ifx,
                               ARG.ify, ARG.ifz));
        plt::xlabel("Time [s]");
        if(ARG.water_value > 0)
        {
            plt::ylabel("Attenuation [HU]");
        } else
        {
            plt::ylabel("Attenuation");
        }
        std::vector<double> taxis_scatter;
        std::vector<double> plotme_scatter;
        if(ARG.staticReconstructionDir != "")
        {
            std::shared_ptr<util::ReconstructedSeriesEvaluator> _concentration
                = std::make_shared<util::ReconstructedSeriesEvaluator>(
                    ARG.staticReconstructionDir, ARG.sweepCount, ARG.sweepTime, ARG.sweepOffset);
            taxis_scatter = _concentration->nativeTimeDiscretization();
            plotme_scatter = _concentration->nativeValuesIn(ARG.ifx, ARG.ify, ARG.ifz);
            if(ARG.water_value > 0)
            {
                for(uint32_t i = 0; i != plotme_scatter.size(); i++)
                {
                    float v = plotme_scatter[i];
                    float hu = 1000 * (v / ARG.water_value - 1.0);
                    plotme_scatter[i] = hu;
                    taxis_scatter[i] = taxis_scatter[i] / 1000;
                }
            }
        }
        std::vector<double> taxis;
        float* _taxis = new float[ARG.granularity];
        concentration->timeDiscretization(ARG.granularity, _taxis);
        std::vector<double> plotme;
        for(uint32_t i = 0; i != ARG.granularity; i++)
        {
            float v = aif[i] + aifofset;
            if(ARG.water_value > 0)
            {
                float hu = 1000 * (v / ARG.water_value - 1.0);
                plotme.push_back(hu);
            } else
            {
                plotme.push_back(v);
            }
            taxis.push_back(_taxis[i] / 1000);
        }
        plt::plot(taxis, plotme);
        if(ARG.staticReconstructionDir != "")
        {
            std::map<std::string, std::string> pltargs;
            pltargs.insert(std::pair<std::string, std::string>("Color", "Orange"));
            plt::scatter(taxis_scatter, plotme_scatter, 90.0, pltargs);
        }
        if(ARG.showAIF)
        {
            plt::show();
        }
        if(!ARG.aifImageFile.empty())
        {
            plt::save(ARG.aifImageFile);
        }
        delete[] _taxis;
    }
    if(ARG.stopAfterVizualization)
    {
        return 0;
    }
    bool truncatedInstead = false;
    float lambdaRel = 0.2;
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
    util::TimeSeriesDiscretizer tsd(concentration, dimx, dimy, dimz, ARG.startTime, ARG.endTime,
                                    ARG.secLength, ARG.threads);
    LOGD << "TTP computation.";
    tsd.computeTTP(ARG.granularity, ttp_w, aif);
    if(!ARG.stopAfterTTP)
    {
        LOGD << "CBV, CBF and MTT computation.";
        tsd.computePerfusionParameters(ARG.granularity, convolutionMatrix, cbf_w, cbv_w, mtt_w);
    }
    delete[] convolutionMatrix;
    delete[] aif;
    LOGI << io::xprintf("END %s", argv[0]);
}
