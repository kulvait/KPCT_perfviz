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

    /**
     *Size of pause between sweeps [ms].
     *
     *Computed from DICOM files as 2088.88889ms. Based on experiment, it is 1171ms.
     */
    float pauseSize = 1171;

    /** Frame Time. (0018, 1063) Nominal time (in msec) per individual frame.
     *
     *The model assumes that there is delay between two consecutive frames of the frame_time.
     *First frame is aquired directly after pause. From DICOM it is 16.6666667ms. From
     *experiment 16.8ms.
     */
    float frameTime = 16.8;

    /// Angles per sweep
    uint32_t anglesPerSweep = 248;

    /// Start of C-Arm acquisition [ms].
    float startOffset = 0.0;

    // C-Arm CT and CT intervals should be ofsetted in a way that the half of the interval matches
    // for both
    bool intervalCenterOffset = false;

    /** Time conversion constant
     *
     *Length of the second for the unit of C-Arm CT acquisition that will be converted into time
     *utit of CT run. The computation will be performed as
     *ct_time [s] = carm_time [ms] / sec_length
     */
    float secLength = 1000.0;

    /// Number of threads, zero for no threading
    uint16_t threads = 0;

    /// Angles per sweep
    uint32_t sweepCount = 10;

    uint16_t dimx, dimy, dimz;

    double totalSweepTime, startct, endct, startcarm, endcarm;
};

int Arguments::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Produces volumes of perfusion data for the time points of C-Arm CT protocol for "
                  "classical CT data." };
    app.add_option("output_folder", outputFolder,
                   "Folder to which output data of perfusion coefficients.")
        ->required()
        ->check(CLI::ExistingDirectory);
    app.add_option("static_reconstructions", coefficientVolumeFiles,
                   "Coeficients of the basis functions fited by the algorithm. Orderred in the "
                   "same order as the basis is sampled in a DEN file.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-s,--pause-size", pauseSize,
                   "Size of pause [ms]. This might be supplied for fine tuning of the algorithm."
                   "[default is 1171] ")
        ->check(CLI::Range(0.01, 100000.0));
    app.add_option("-f,--frame-time", frameTime,
                   "Frame Time. (0018, 1063) Nominal time (in msec) per individual frame (slice) "
                   "[ms]. Might be supplied for fine tuning of the algorithm. [default is "
                   "16.8]")
        ->check(CLI::Range(0.01, 10000.0));
    app.add_option("-a,--angles-per-sweep", anglesPerSweep,
                   "Number of frames acquired per one sweep, defaults to 248")
        ->check(CLI::Range(1, 10000));
    CLI::Option* cli_startOffset
        = app.add_option(
                 "-i,--start-offset", startOffset,
                 "The timings for C-Arm and CT aquisition might differ. This offset [ms] "
                 "controls how the acquisition of the C-Arm data is shifted relative to the "
                 "latest time of the first CT stack volume [defaults to 0.0].")
              ->check(CLI::Range(-1000000.0, 1000000.0));
    CLI::Option* cli_centralTime
        = app.add_flag("--interval-center-offset", intervalCenterOffset,
                       "The timings for C-Arm and CT aquisition might differ. When this setting is "
                       "enabled, then the middle of the CT interval will match the middle of the "
                       "C-Arm CT interval [defaults to false].");
    cli_centralTime->excludes(cli_startOffset);
    cli_startOffset->excludes(cli_centralTime);
    app.add_option("-c,--sec-length", secLength,
                   "Length of one second in the units of the domain. Defaults to 1000.")
        ->check(CLI::Range(0.0, 1000000.0));
    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use. Defaults to 0 which means "
                   "sychronous execution.")
        ->check(CLI::Range(0, 65535));
    app.add_option("--sweep-count", sweepCount, "Number of sweeps. Default 10.")
        ->check(CLI::Range(0, 100));
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
        totalSweepTime = (anglesPerSweep - 1) * frameTime + pauseSize;
        std::shared_ptr<io::Frame2DReaderI<float>> startData
            = std::make_shared<io::DenFrame2DReader<float>>(tickFiles[0]);
        std::shared_ptr<io::Frame2DReaderI<float>> endData
            = std::make_shared<io::DenFrame2DReader<float>>(tickFiles[tickFiles.size() - 1]);
        std::shared_ptr<io::Frame2DI<float>> fs, fe;
        fs = startData->readFrame(0);
        fe = endData->readFrame(0);
        // Fifth element of frame is time
        startct = fs->get(4, 0);
        endct = fe->get(4, 0);
        float start, end;
        for(std::size_t z = 0; z != dimz; z++)
        {
            fs = startData->readFrame(z);
            fe = endData->readFrame(z);
            start = fs->get(4, 0);
            end = fe->get(4, 0);
            if(start > startct)
            {
                startct = start;
            }
            if(end < endct)
            {
                endct = end;
            }
        }
        if(intervalCenterOffset)
        {
            double halfcarm = totalSweepTime * 9 + (anglesPerSweep - 1) * frameTime;
            double halfct = endct - startct;
            startOffset = halfct - halfcarm;
        }
        startcarm = (startct * secLength + startOffset) / secLength;
        endcarm = (startct * secLength + startOffset + totalSweepTime * 9
                   + (anglesPerSweep - 1) * frameTime)
            / secLength;
        // startct is the latest time from the first stack
        // endct is the earliest time from the last stack
        LOGI << io::xprintf("C-Arm data interval is [%f, %f]", startcarm, endcarm);
        LOGI << io::xprintf("CT data interval is [%f, %f]", startct, endct);

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
        threadpool->init();
        threadpool->resize(a.threads);
        for(uint32_t angleid = 0; angleid != a.anglesPerSweep; angleid++)
        {
            concentration = std::make_shared<util::CTEvaluator>(a.coefficientVolumeFiles,
                                                                a.tickFiles, false, false, 0.0);
            t = a.startct * a.secLength + a.startOffset + sweepid * a.totalSweepTime
                + angleid * a.frameTime;
            t /= a.secLength;
            if(t < a.startct || t > a.endct)
            {
                LOGW << io::xprintf("Time %f is out of the range [%f, %f]", t, a.startct, a.endct);
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
