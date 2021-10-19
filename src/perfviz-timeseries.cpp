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
#include "PROG/ArgumentsBasisSpecification.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PerfusionVizualizationArguments.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "stringFormatter.h"
#include "utils/Attenuation4DEvaluatorI.hpp"
#include "utils/PolynomialSeriesEvaluator.hpp"
#include "utils/FourierSeriesEvaluator.hpp"
#include "utils/EngineerSeriesEvaluator.hpp"
#include "utils/ReconstructedSeriesEvaluator.hpp"
#include "utils/TimeSeriesDiscretizer.hpp"

#ifdef DEBUG
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
#endif

using namespace KCT;
using namespace KCT::util;

/// Arguments of the main function.
class Args : public ArgumentsThreading,
             public PerfusionVizualizationArguments,
             public ArgumentsBasisSpecification
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName)
        , PerfusionVizualizationArguments(argc, argv, prgName)
        , ArgumentsBasisSpecification(argc, argv, prgName){};

    /// Folder to which output data after the linear regression.
    std::string outputFolder;

    /// Projection files in a DEN format to use for linear regression.
    std::vector<std::string> volumeCoefficients;
    int granularity = 100;
    bool allowNegativeValues = true;
};

void Args::defineArguments()
{
    std::string optstr;
    cliApp
        ->add_option("output_folder", outputFolder,
                     "Folder to which output data after the linear regression. It needs to be "
                     "directory or symbol - to indicate that no computation should be performed.")
        ->required()
        ->check(CLI::ExistingDirectory);
    cliApp
        ->add_option("fitted_coeficients", volumeCoefficients,
                     "Basis coeficients fited by the algorithm. Orderred from the first "
                     "coeficient that corresponds to the constant.")
        ->required()
        ->check(CLI::ExistingFile);
    optstr = io::xprintf("Number of volumes to produce, [defaults to %d].", granularity);
    cliApp->add_option("--granularity", granularity, optstr)->check(CLI::Range(1, 2147483647));
    optstr = io::xprintf("Allow values below baseline value that translate to a negative TACs "
                         "values, [defaults to %s].",
                         allowNegativeValues ? "true" : "false");
    cliApp->add_flag("--allow-negative-values", allowNegativeValues, optstr);
    bool includeBasisSize = false;
    bool includeBasisSetSelectionArgs = true;
    bool includeFittingOptions = false; // We don't do any fitting here
    addBasisSpecificationArgs(includeBasisSize, includeBasisSetSelectionArgs,
                              includeFittingOptions);
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
             "Create time series according to a given model of dimension reduction from the "
             "perfusion data respectively corresponding coefficient volumes.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.volumeCoefficients[0]);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t dimz = di.dimz();
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration = nullptr;
    if(ARG.useLegendrePolynomials)
    {
        uint32_t degree = ARG.volumeCoefficients.size() - 1;
        bool negativeAsZero = !ARG.allowNegativeValues;
        concentration = std::make_shared<util::PolynomialSeriesEvaluator>(
            degree, ARG.volumeCoefficients, 0.0f, (float)(ARG.granularity - 1), negativeAsZero,
            polynomialType::Legendre);
    } else if(ARG.useChebyshevPolynomials)
    {
        uint32_t degree = ARG.volumeCoefficients.size() - 1;
        bool negativeAsZero = !ARG.allowNegativeValues;
        concentration = std::make_shared<util::PolynomialSeriesEvaluator>(
            degree, ARG.volumeCoefficients, 0.0f, (float)(ARG.granularity - 1), negativeAsZero,
            polynomialType::Chebyshev);
    } else if(ARG.useFourierBasis)
    {
        bool negativeAsZero = !ARG.allowNegativeValues;
        concentration = std::make_shared<util::FourierSeriesEvaluator>(
            ARG.volumeCoefficients, 0.0f, (float)(ARG.granularity - 1), negativeAsZero,
            ARG.halfPeriodicFunctions);
    } else if(!ARG.engineerBasis.empty())
    {
        concentration = std::make_shared<util::EngineerSeriesEvaluator>(
            ARG.engineerBasis, ARG.volumeCoefficients, 0.0f, (float)(ARG.granularity - 1));
    } else
    {
        LOGE << "Type of action not specified.";
        return -1;
    }

    bool subtractZeroVolume = false;
    for(int i = 0; i != ARG.granularity; i++)
    {
        std::string volumei = io::xprintf("%s/Volume_%03d.den", ARG.outputFolder.c_str(), i);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> volumei_w
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(volumei, dimx, dimy, dimz);
        concentration->volumeAt((float)i, volumei_w, subtractZeroVolume);
    }
    LOGI << io::xprintf("END %s", argv[0]);
}
