// Add to an iPhone host app AND keyboard extension target. Call on a serial worker.
// Tensor-level deployment probe; SentencePiece and search belong to the application.
import Foundation
import CoreML
import Darwin

struct ProbeMeasurement: Codable {
    let stage: String
    let elapsedMS: Double
    let footprintBeforeBytes: UInt64?
    let footprintAfterBytes: UInt64?
}

func physicalFootprint() -> UInt64? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let capacity = Int(count)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: capacity) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : nil
}

final class CoreMLProbe {
    private let model: MLModel
    let load: ProbeMeasurement
    private(set) var lastPrediction: ProbeMeasurement?

    // Supply Xcode's compiled .mlmodelc URL. Measure package compilation separately.
    init(compiledURL: URL, units: MLComputeUnits) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        let before = physicalFootprint()
        let start = ProcessInfo.processInfo.systemUptime
        model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        load = ProbeMeasurement(stage: "load_compiled", elapsedMS:
            (ProcessInfo.processInfo.systemUptime - start) * 1000,
            footprintBeforeBytes: before, footprintAfterBytes: physicalFootprint())
    }

    func predict(ids: [Int]) throws -> MLMultiArray {
        guard !ids.isEmpty, ids.count <= 128, ids.allSatisfy({ (0..<16384).contains($0) }) else {
            throw NSError(domain: "VimeMLProbe", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Expected 1..128 vocabulary IDs"])
        }
        let before = physicalFootprint()
        let start = ProcessInfo.processInfo.systemUptime
        let input = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        for (index, token) in ids.enumerated() { input[index] = NSNumber(value: token) }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        guard let logits = result.featureValue(for: "logits")?.multiArrayValue,
              logits.shape.map({ $0.intValue }) == [1, ids.count, 16384] else {
            throw NSError(domain: "VimeMLProbe", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Unexpected logits shape"])
        }
        lastPrediction = ProbeMeasurement(stage: "predict_with_input_and_output", elapsedMS:
            (ProcessInfo.processInfo.systemUptime - start) * 1000,
            footprintBeforeBytes: before, footprintAfterBytes: physicalFootprint())
        return logits
    }

    // Stable full-vocabulary log-softmax; no special-token filtering during reranking.
    static func suffixScore(logits: MLMultiArray, targets: [Int], scoreStart: Int) throws -> Double {
        guard logits.shape.count == 3, logits.shape[0].intValue == 1,
              logits.shape[1].intValue == targets.count, logits.shape[2].intValue == 16384,
              scoreStart >= 0, scoreStart < targets.count,
              targets.allSatisfy({ (0..<16384).contains($0) }) else {
            throw NSError(domain: "VimeMLProbe", code: 3)
        }
        var total = 0.0
        for position in scoreStart..<targets.count {
            // Index through MLMultiArray strides, including outputs with noncontiguous layout.
            func value(_ token: Int) -> Double {
                logits[[NSNumber(value: 0), NSNumber(value: position), NSNumber(value: token)]].doubleValue
            }
            var maximum = -Double.infinity
            for token in 0..<16384 { maximum = max(maximum, value(token)) }
            var sum = 0.0
            for token in 0..<16384 { sum += exp(value(token) - maximum) }
            let score = value(targets[position]) - maximum - log(sum)
            guard score.isFinite else { throw NSError(domain: "VimeMLProbe", code: 4) }
            total += score
        }
        return total
    }
}

struct DeviceFixtures: Decodable {
    struct Logits: Decodable { let id: String; let input_ids: [Int]; let valid_length: Int; let last_top1: Int }
    struct Candidate: Decodable {
        let context: String; let text: String; let input_ids: [Int]; let targets: [Int]
        let score_start: Int; let log_probability_sum: Double
    }
    let bundle_manifest_sha256: String
    let logits: [Logits]
    let candidates: [Candidate]
}

extension CoreMLProbe {
    // Keep returned arrays within per-case autorelease pools. Run off the keyboard UI thread.
    func checkFixtures(url: URL) throws -> [String: Double] {
        let fixtures = try JSONDecoder().decode(DeviceFixtures.self, from: Data(contentsOf: url))
        var top1Changes = 0.0
        var maximumScoreDelta = 0.0
        for item in fixtures.logits {
            try autoreleasepool {
                let logits = try predict(ids: item.input_ids)
                var best = 0
                for token in 1..<16384 {
                    let current = logits[[NSNumber(value: 0), NSNumber(value: item.valid_length - 1), NSNumber(value: token)]].doubleValue
                    let prior = logits[[NSNumber(value: 0), NSNumber(value: item.valid_length - 1), NSNumber(value: best)]].doubleValue
                    if current > prior { best = token }
                }
                if best != item.last_top1 { top1Changes += 1 }
            }
        }
        for item in fixtures.candidates {
            try autoreleasepool {
                let logits = try predict(ids: item.input_ids)
                let score = try Self.suffixScore(logits: logits, targets: item.targets, scoreStart: item.score_start)
                maximumScoreDelta = max(maximumScoreDelta, abs(score - item.log_probability_sum))
            }
        }
        return ["last_top1_changes": top1Changes, "max_candidate_sum_abs": maximumScoreDelta]
    }
}
