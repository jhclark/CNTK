//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;
using namespace std::placeholders;

extern bool Is1bitSGDAvailable();

namespace
{
    struct FeedForwardClassifier
    {
        size_t inputDim;
        size_t ouputDim;
        Variable features;
        Variable labels;
        FunctionPtr output;
        FunctionPtr trainingLoss;
        FunctionPtr prediction;
    };

    const std::wstring g_featureStreamName = L"features";
    const std::wstring g_labelsStreamName = L"labels";
    const std::wstring g_inputFile = L"SimpleDataTrain_cntk_text.txt";
    const size_t minibatchSize = 25;
    const size_t outputFrequencyInMinibatches = 20;
    const size_t numSamplesPerSweep = 10000;
    const size_t numSweepsToTrainWith = 2;
    const size_t numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;
    const size_t totalNumberOfSamples = numSamplesPerSweep * numSweepsToTrainWith;
    const size_t warmStart = 1000;

    void LoopBasedOnMinibatches(const std::wstring& name, const DeviceDescriptor& device, std::function<DistributedTrainerPtr()> factory, const FeedForwardClassifier& classifier, std::vector<double>* pTrainCE)
    {
        printf("Training loop thru minibatches with %ls.\n", name.c_str());
        auto distributedTrainer = factory();

        auto minibatchSource = TextFormatMinibatchSource(g_inputFile, { { g_featureStreamName, classifier.inputDim }, { g_labelsStreamName, classifier.ouputDim } }, MinibatchSource::InfinitelyRepeat, true, (distributedTrainer ? distributedTrainer->GetDistributedAfterSampleCount() : MinibatchSource::InfiniteSamples));
        auto featureStreamInfo = minibatchSource->StreamInfo(g_featureStreamName);
        auto labelStreamInfo = minibatchSource->StreamInfo(g_labelsStreamName);

        double learningRatePerSample = 0.02;

        Trainer trainer(classifier.output, classifier.trainingLoss, classifier.prediction, { SGDLearner(classifier.output->Parameters(), LearningRatePerSampleSchedule(learningRatePerSample)) }, distributedTrainer);
        for (size_t i = 0; i < numMinibatchesToTrain; ++i)
        {
            auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
            trainer.TrainMinibatch({ { classifier.features, minibatchData[featureStreamInfo].m_data }, { classifier.labels, minibatchData[labelStreamInfo].m_data } }, device);
            if (i % 300 == 0)
                trainer.SaveCheckpoint(L"test.tmp");
            PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
            if (pTrainCE)
                pTrainCE->push_back(trainer.PreviousMinibatchLossAverage());
        }
    }

    void LoopBasedOnSamples(const std::wstring& name, const DeviceDescriptor& device, std::function<DistributedTrainerPtr()> factory, const FeedForwardClassifier& classifier, std::vector<double>* pTrainCE)
    {
        printf("Training loop thru samples with %ls.\n", name.c_str());
        auto distributedTrainer = factory();

        auto minibatchSource = TextFormatMinibatchSource(g_inputFile, { { g_featureStreamName, classifier.inputDim }, { g_labelsStreamName, classifier.ouputDim } }, MinibatchSource::InfinitelyRepeat, true, (distributedTrainer ? distributedTrainer->GetDistributedAfterSampleCount() : MinibatchSource::InfiniteSamples));
        auto featureStreamInfo = minibatchSource->StreamInfo(g_featureStreamName);
        auto labelStreamInfo = minibatchSource->StreamInfo(g_labelsStreamName);

        double learningRatePerSample = 0.02;

        Trainer trainer(classifier.output, classifier.trainingLoss, classifier.prediction, { SGDLearner(classifier.output->Parameters(), LearningRatePerSampleSchedule(learningRatePerSample)) }, distributedTrainer);
        size_t checkpointFrequency = 7000;
        size_t count = 0, index = 0;
        bool updated = true;
        while (count < totalNumberOfSamples && updated)
        {
            auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
            updated = trainer.TrainMinibatch({ { classifier.features, minibatchData[featureStreamInfo].m_data }, { classifier.labels, minibatchData[labelStreamInfo].m_data } }, device);
            count += trainer.PreviousMinibatchSampleCount();
            if (count % checkpointFrequency == 0)
                trainer.SaveCheckpoint(L"test.tmp");
            PrintTrainingProgress(trainer, index++, outputFrequencyInMinibatches);
            if (pTrainCE)
                pTrainCE->push_back(trainer.PreviousMinibatchLossAverage());
        }
    }

    FeedForwardClassifier BuildFeedForwardClassifer(const DeviceDescriptor& device)
    {
        const size_t inputDim = 2;
        const size_t numOutputClasses = 2;
        const size_t hiddenLayerDim = 50;
        const size_t numHiddenLayers = 2;

        auto minibatchSource = TextFormatMinibatchSource(g_inputFile, { { g_featureStreamName, inputDim }, { g_labelsStreamName, numOutputClasses } }, MinibatchSource::FullDataSweep, false);
        auto featureStreamInfo = minibatchSource->StreamInfo(g_featureStreamName);
        auto labelStreamInfo = minibatchSource->StreamInfo(g_labelsStreamName);

        std::unordered_map<StreamInformation, std::pair<NDArrayViewPtr, NDArrayViewPtr>> inputMeansAndInvStdDevs = { { featureStreamInfo, { nullptr, nullptr } } };
        ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, inputMeansAndInvStdDevs);

        auto nonLinearity = std::bind(Sigmoid, _1, L"Sigmoid");
        auto input = InputVariable({ inputDim }, DataType::Float, g_featureStreamName);
        auto normalizedinput = PerDimMeanVarianceNormalize(input, inputMeansAndInvStdDevs[featureStreamInfo].first, inputMeansAndInvStdDevs[featureStreamInfo].second);
        auto classifierOutput = FullyConnectedDNNLayer(normalizedinput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedInput"));
        for (size_t i = 1; i < numHiddenLayers; ++i)
            classifierOutput = FullyConnectedDNNLayer(classifierOutput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedHidden"));

        auto outputTimesParam = Parameter({ numOutputClasses, hiddenLayerDim }, DataType::Float, UniformInitializer(CNTK::DefaultParamInitScale, 1), device, L"outputTimesParam");
        auto outputBiasParam = Parameter({ numOutputClasses }, DataType::Float, UniformInitializer(CNTK::DefaultParamInitScale, 1), device, L"outputBiasParam");
        classifierOutput = Plus(outputBiasParam, Times(outputTimesParam, classifierOutput), L"classifierOutput");

        auto labels = InputVariable({ numOutputClasses }, DataType::Float, g_labelsStreamName);
        auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
        auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

        return FeedForwardClassifier{ inputDim, numOutputClasses, input, labels, classifierOutput, trainingLoss, prediction };
    }
}

void TestFrameMode()
{
    // Create a set of trainers.
    std::map<std::wstring, std::function<DistributedTrainerPtr()>> trainers;
    trainers[L"simple"] = []() { return CreateDataParallelDistributedTrainer(MPICommunicator(), false); };

    if (Is1bitSGDAvailable())
    {
        trainers[L"1bitsgd"] = []() { return CreateQuantizedDataParallelDistributedTrainer(QuantizedMPICommunicator(true, true, 1), false, 0); };
        trainers[L"blockmomentum"] = []() { return CreateBlockMomentumDistributedTrainer(MPICommunicator(), 1024); };
    }

    // Create a set of devices.
    std::vector<DeviceDescriptor> devices;
    devices.push_back(DeviceDescriptor::CPUDevice());
    if (IsGPUAvailable())
        devices.push_back(DeviceDescriptor::GPUDevice(0));

    // Create different types of loops.
    std::vector<std::function<void(const std::wstring&, const DeviceDescriptor&, std::function<DistributedTrainerPtr()>, const FeedForwardClassifier&, std::vector<double>*)>> loops;
    loops.push_back(LoopBasedOnMinibatches);
    loops.push_back(LoopBasedOnSamples);

    // Trying all distribution methods on all available devices with different types of loops.
    auto sync = MPICommunicator();
    for (auto t : trainers)
    {
        for (auto device : devices)
        {
            for (auto loop : loops)
            {
                sync->Barrier();
                loop(t.first, device, t.second, BuildFeedForwardClassifer(device), nullptr);
            }
        }
    }
    sync->Barrier();

}

void TestAccuracy()
{
    // Create a set of devices.
    std::vector<DeviceDescriptor> devices;
    devices.push_back(DeviceDescriptor::CPUDevice());
    if (IsGPUAvailable())
        devices.push_back(DeviceDescriptor::GPUDevice(0));

    // generate ground truth for accuracy test
    std::vector<double> groundTruthTrainCE;
    LoopBasedOnSamples(
        L"GroundTruth",
        DeviceDescriptor::CPUDevice(),
        []() -> DistributedTrainerPtr { return nullptr; },
        BuildFeedForwardClassifer(DeviceDescriptor::CPUDevice()),
        &groundTruthTrainCE);

    // Test simple aggregation accuracy
    for (auto device : devices)
    {
        std::vector<double> trainCE;
        LoopBasedOnSamples(
            L"simple",
            device,
            []() { return CreateDataParallelDistributedTrainer(MPICommunicator(), false); },
            BuildFeedForwardClassifer(device),
            &trainCE);

        // check if CE matches ground truth
        for (int i = 0; i < numMinibatchesToTrain / outputFrequencyInMinibatches; i++)
        {
            FloatingPointCompare(trainCE[i], groundTruthTrainCE[i], "Simple aggregation CE deviated from non-distributed");
        }
    }

    // Test warm-start
    for (auto device : devices)
    {
        // test 1-bit SGD with warm start
        std::vector<double> trainCE;
        LoopBasedOnSamples(
            L"1bitsgd",
            device,
            []() { return CreateDataParallelDistributedTrainer(MPICommunicator(), false, warmStart); },
            BuildFeedForwardClassifer(device),
            &trainCE);

        // check if CE before warm start matches ground truth
        for (int i = 0; i < warmStart / outputFrequencyInMinibatches / minibatchSize; i++)
        {
            FloatingPointCompare(trainCE[i], groundTruthTrainCE[i], "Warm start CE deviated from non-distributed");
        }
    }
}