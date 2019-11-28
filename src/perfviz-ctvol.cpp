// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser

#include "ftpl.h" //Threadpool

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
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

    /// Number of sweeps
    uint32_t sweepCount = 10;

    /// Angles per sweep
    uint32_t anglesPerSweep = 248;

    /** Frame Time. (0018, 1063) Nominal time (in msec) per individual frame.
     *
     *The model assumes that there is delay between two consecutive frames of the frame_time.
     *First frame is aquired directly after pause. From DICOM it is 16.6666667ms. From
     *experiment 16.8ms.
     */
    float frameTime_ms = 16.8;

    /**
     *Size of pause between sweeps [ms].
     *
     *Computed from DICOM files as 2088.88889ms. Based on experiment, it is 1171ms.
     */
    float pauseSize_ms = 1171;

    /// Start of C-Arm acquisition [ms].
    float startOffset_ms = 0.0;
    float CTDataStartOffset_ms, CTDataCenterOffset_ms;

    /** Time conversion constant
     *
     *Length of the second for the unit of C-Arm CT acquisition that will be converted into time
     *utit of CT run. The computation will be performed as
     *ct_time [s] = carm_time [ms] / sec_length
     */
    float secLength = 1000.0;

    /// Number of threads, zero for no threading
    uint16_t threads = 0;

    uint16_t dimx, dimy, dimz;

    double totalSweepTime_ms, startCTData_s, endCTData_s, startcarm_s, endcarm_s;
};

int Arguments::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "From the volumes of the CT data together with tick files with the time "
                  "informations exports volumes to be used for the time points of C-Arm CT "
                  "protocol." };
    app.add_option("output_folder", outputFolder,
                   "Folder to which output data of perfusion coefficients.")
        ->required()
        ->check(CLI::ExistingDirectory);
    app.add_option("static_reconstructions", coefficientVolumeFiles,
                   "Volume files equipped with the tick files in a DEN format.")
        ->required()
        ->check(CLI::ExistingFile);
    CLI::Option_group* carm_cli
        = app.add_option_group("C-Arm CT parameters", "Parameters to tune C-Arm CT acquisition.");
    carm_cli->add_option("--sweep-count", sweepCount, "Number of sweeps. Default 10.")
        ->check(CLI::Range(0, 100));
    carm_cli
        ->add_option("-a,--angles-per-sweep", anglesPerSweep,
                     "Number of frames acquired per one sweep, defaults to 248")
        ->check(CLI::Range(1, 10000));
    carm_cli
        ->add_option("-f,--frame-time", frameTime_ms,
                     "Frame Time. (0018, 1063) Nominal time (in msec) per individual frame (slice) "
                     "[ms]. Might be supplied for fine tuning of the algorithm. [default is "
                     "16.8]")
        ->check(CLI::Range(0.01, 10000.0));
    carm_cli
        ->add_option("-s,--pause-size", pauseSize_ms,
                     "Size of pause [ms]. This might be supplied for fine tuning of the algorithm."
                     "[default is 1171] ")
        ->check(CLI::Range(0.01, 100000.0));
    CLI::Option_group* interval_cli = app.add_option_group(
        "Interval adjustments", "Parameters to align CT with the C-Arm CT interval.");
    interval_cli
        ->add_option("--sec-length", secLength,
                     "Units of C-Arm CT interval are usually miliseconds while timing of CT is in "
                     "seconds. How many units of the C-Arm interval [ms] are equal to one unit of "
                     "the CT interval [s]. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
    CLI::Option* startOffset_option
        = interval_cli
              ->add_option("--start-offset", startOffset_ms,
                           "The timings for C-Arm and CT aquisition might differ. This offset [ms] "
                           "controls how the zero time of the acquisition of the C-Arm data is "
                           "shifted relative to the zero time of the CT data. Note that the zero "
                           "time might not be represented in the volume data. [defaults to 0.0]")
              ->check(CLI::Range(-1000000.0, 1000000.0));
    CLI::Option* CTDataStartOffset_option
        = interval_cli
              ->add_option(
                  "--ct-start-offset", CTDataStartOffset_ms,
                  "The timings for C-Arm and CT aquisition might differ. This offset when "
                  "specified [ms] controls how the acquisition of the C-Arm data is shifted "
                  "relative to the latest time of the first CT stack volume [by default, "
                  "--start-offset 0.0 is used].")
              ->check(CLI::Range(-1000000.0, 1000000.0));
    CLI::Option* CTDataCenterOffset_option
        = interval_cli
              ->add_option("--interval-center-offset", CTDataCenterOffset_ms,
                           "The timings for C-Arm and CT aquisition might differ. By specifiing "
                           "this parameter, you set the offset [ms] of the middles of the C-Arm CT "
                           "interval with respect to CT interval [by default, --start-offset 0.0 "
                           "is used].")
              ->check(CLI::Range(-1000000.0, 1000000.0));
    startOffset_option->excludes(CTDataStartOffset_option);
    startOffset_option->excludes(CTDataCenterOffset_option);
    CTDataStartOffset_option->excludes(startOffset_option);
    CTDataStartOffset_option->excludes(CTDataCenterOffset_option);
    CTDataCenterOffset_option->excludes(startOffset_option);
    CTDataCenterOffset_option->excludes(CTDataStartOffset_option);
    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use. Defaults to 0 which means "
                   "sychronous execution.")
        ->check(CLI::Range(0, 65535));
    try
    {
        app.parse(argc, argv);
        if(coefficientVolumeFiles.size() != 29)
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
        io::DenFileInfo di(coefficientVolumeFiles[0]);
        dimx = di.dimx();
        dimy = di.dimy();
        dimz = di.dimz();
        for(std::string f : coefficientVolumeFiles)
        {
            io::DenFileInfo df(f);
            if(dimx != df.dimx() || dimy != df.dimy() || dimz != df.dimz())
            {
                std::string err = io::xprintf("Dimension check for the file %s fails.", f.c_str());
                LOGE << err;
                throw new std::runtime_error(err);
            }
            std::string tickFile = f.substr(0, f.find_last_of(".")) + ".tick";
            if(!io::fileExists(tickFile))
            {
                std::string err = io::xprintf("There is no tick file %s.", tickFile.c_str());
                LOGE << err;
                throw new std::runtime_error(err);

            } else
            {
                tickFiles.push_back(tickFile);
            }
        }
        totalSweepTime_ms = (anglesPerSweep - 1) * frameTime_ms + pauseSize_ms;
        std::shared_ptr<io::Frame2DReaderI<float>> startData
            = std::make_shared<io::DenFrame2DReader<float>>(tickFiles[0]);
        std::shared_ptr<io::Frame2DReaderI<float>> endData
            = std::make_shared<io::DenFrame2DReader<float>>(tickFiles[tickFiles.size() - 1]);
        std::shared_ptr<io::Frame2DI<float>> fs, fe;
        fs = startData->readFrame(0);
        fe = endData->readFrame(0);
        // Fifth element of frame is time
        startCTData_s = fs->get(4, 0);
        endCTData_s = fe->get(4, 0);
        float start, end;
        for(std::size_t z = 0; z != dimz; z++)
        {
            fs = startData->readFrame(z);
            fe = endData->readFrame(z);
            start = fs->get(4, 0);
            end = fe->get(4, 0);
            if(start > startCTData_s)
            {
                startCTData_s = start;
            }
            if(end < endCTData_s)
            {
                endCTData_s = end;
            }
        }
        if(endCTData_s - startCTData_s <= 0.0)
        {
            LOGE << io::xprintf("CT data provided inconsistent start %f and end %f stamps!",
                                startCTData_s, endCTData_s);
            return -1;
        }
        if(CTDataStartOffset_option->count() > 0)
        {
            startOffset_ms = startCTData_s * secLength + CTDataStartOffset_ms;
        } else if(CTDataCenterOffset_option->count() > 0)
        {
            double halfCarmInterval_ms
                = (totalSweepTime_ms * (sweepCount - 1) + (anglesPerSweep - 1) * frameTime_ms) / 2;
            double halfCTDataInterval_ms = (endCTData_s - startCTData_s) * secLength / 2;
            startOffset_ms = startCTData_s * secLength + halfCTDataInterval_ms - halfCarmInterval_ms
                + CTDataCenterOffset_ms;
        }
        startcarm_s = startOffset_ms / secLength;
        endcarm_s = (startOffset_ms + totalSweepTime_ms * (sweepCount - 1)
                     + (anglesPerSweep - 1) * frameTime_ms)
            / secLength;
        // startct is the latest time from the first stack
        // endct is the earliest time from the last stack
        LOGI << io::xprintf("C-Arm data interval is [%f, %f]", startcarm_s, endcarm_s);
        LOGI << io::xprintf("CT data coverred interval is [%f, %f]", startCTData_s, endCTData_s);

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

int writeVolume(
    int i,
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration,
    double t,
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> volumeWritter) // For testing normal
{
    uint16_t dimx = volumeWritter->dimx();
    uint16_t dimy = volumeWritter->dimy();
    uint16_t dimz = volumeWritter->dimz();
    float* buffer = new float[dimx * dimy];
    std::unique_ptr<io::Frame2DI<float>> memoryViewer
        = std::make_unique<io::FrameMemoryViewer2D<float>>(buffer, dimx, dimy);
    for(uint32_t z = 0; z != dimz; z++)
    {
        concentration->frameAt(z, t, buffer);
        volumeWritter->writeFrame(*memoryViewer, z);
    }
    delete[] buffer;
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

    std::shared_ptr<io::AsyncFrame2DWritterI<float>> volumeWritter;

    ftpl::thread_pool* threadpool = nullptr;
    if(a.threads == 0)
    {
        LOGD << io::xprintf("Computing perfusion volumes synchronously without threading.");
    } else
    {
        LOGD << io::xprintf("Computing perfusion volumes on %d threads.", a.threads);
        threadpool = new ftpl::thread_pool(a.threads);
    }
    double t = 0.0;
    std::shared_ptr<util::Attenuation4DEvaluatorI> concentration;
    for(uint32_t sweepid = 0; sweepid != a.sweepCount; sweepid++)
    {
        if(threadpool != nullptr)
        {
            threadpool->init();
            threadpool->resize(a.threads);
        }
        for(uint32_t angleid = 0; angleid != a.anglesPerSweep; angleid++)
        {
            concentration = std::make_shared<util::CTEvaluator>(a.coefficientVolumeFiles,
                                                                a.tickFiles, false, false, -1024.0);
            // Timing information about CT acquisition is provided by means of tick files
            t = a.startOffset_ms + sweepid * a.totalSweepTime_ms + angleid * a.frameTime_ms;
            t /= a.secLength;
            if(t < a.startCTData_s || t > a.endCTData_s)
            {
                LOGI << io::xprintf("Time %fs out of range [%fs, %fs], sweep=%02d, angle=%03d.", t,
                                    a.startCTData_s, a.endCTData_s, sweepid, angleid);
            }
            if(angleid == 0)
            {
                LOGD << io::xprintf("Creating Volume%02d_%03d.den at time %0.2f", sweepid, angleid,
                                    t);
            }
            volumeWritter = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                io::xprintf("%s/Volume%02d_%03d.den", a.outputFolder.c_str(), sweepid, angleid),
                a.dimx, a.dimy, a.dimz);
            if(threadpool != nullptr)
            {
                threadpool->push([&, concentration, t, volumeWritter](int id) {
                    writeVolume(id, concentration, t, volumeWritter);
                });
            } else
            {
                writeVolume(0, concentration, t, volumeWritter); // For testing normal
            }
        }
        if(threadpool != nullptr)
        {
            threadpool->stop(true); // Wait for threads
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
        threadpool = nullptr;
    }
    LOGI << io::xprintf("END %s", argv[0]);
}
